"""Explicit training entry; importing this module or preflight never trains."""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from recbole.trainer import Trainer
from recbole.utils import init_logger

from qwen_runtime import (ROOT, cleanup_distributed,
                          distributed_world_size, is_primary, load_experiment,
                          load_trained_checkpoint)
from qwen_schedule import epoch_learning_rate


def _value(config, key, default=None):
    value = config[key] if key in config else None
    return default if value is None else value


def primary_call(function, label):
    """Run an operation once and broadcast its result or failure to every rank."""
    if distributed_world_size() == 1:
        return function()
    payload = [None]
    if is_primary():
        try:
            payload[0] = {'ok': True, 'result': function()}
        except Exception as error:
            payload[0] = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
    torch.distributed.broadcast_object_list(payload, src=0)
    if not payload[0]['ok']:
        raise RuntimeError(f'{label} failed on rank 0: {payload[0]["error"]}')
    return payload[0]['result']


class _NullWriter:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class QwenTrainer(Trainer):
    def __init__(self, config, model):
        # RecBole's ensure_dir uses a check-then-mkdir sequence. Publish the
        # directory on rank 0 before any rank enters the parent initializer.
        primary_call(lambda: Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True),
                     'checkpoint directory initialization')
        # RecBole constructs a TensorBoard writer in its initializer. Only the
        # primary rank needs one; the real training loop is otherwise unchanged.
        if is_primary():
            super().__init__(config, model)
        else:
            import recbole.trainer.trainer as trainer_module
            original_writer = trainer_module.get_tensorboard
            trainer_module.get_tensorboard = lambda logger: _NullWriter()
            try:
                super().__init__(config, model)
            finally:
                trainer_module.get_tensorboard = original_writer
        self.saved_model_file = primary_call(lambda: self.saved_model_file,
                                             'checkpoint path synchronization')
        self.current_epoch_lr = float(config['learning_rate'])
        self.last_vector_quality = None

    def training_setup(self):
        world_size = distributed_world_size()
        global_batch = int(self.config['train_batch_size'])
        return {
            'world_size': world_size,
            'global_batch_size': global_batch,
            'per_rank_batch_size': global_batch // world_size,
            'learning_rate': float(self.config['learning_rate']),
            'lr_schedule': _value(self.config, 'lr_schedule', 'constant'),
            'warmup_epochs': int(_value(self.config, 'warmup_epochs', 0)),
            'min_learning_rate': float(_value(self.config, 'min_learning_rate', 0.0)),
            'epochs': int(self.config['epochs']),
            'current_epoch_lr': float(self.current_epoch_lr),
        }

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.current_epoch_index = epoch_idx
        self.current_epoch_lr = epoch_learning_rate(
            self.learning_rate, epoch_idx, self.epochs,
            schedule=_value(self.config, 'lr_schedule'),
            warmup_epochs=int(_value(self.config, 'warmup_epochs', 0)),
            min_learning_rate=float(_value(self.config, 'min_learning_rate', 0.0)),
        )
        for group in self.optimizer.param_groups:
            group['lr'] = self.current_epoch_lr
        if is_primary():
            self.logger.info('Epoch %d learning rate: %.8g', epoch_idx + 1, self.current_epoch_lr)
            self.tensorboard.add_scalar('Learning_rate', self.current_epoch_lr, epoch_idx)
        return super()._train_epoch(train_data, epoch_idx, loss_func=loss_func,
                                    show_progress=show_progress and is_primary())

    def _full_sort_batch_eval(self, batched_data):
        interaction, _, positive_u, positive_i = batched_data
        history_index = None
        if self.config['exclude_seen']:
            history = interaction[self.model.ITEM_SEQ].to(self.device)
            rows = torch.arange(history.shape[0], device=self.device).unsqueeze(1).expand_as(history)
            active = history.ne(0)
            history_index = rows[active], history[active]
        batched_data = interaction, history_index, positive_u, positive_i
        return super()._full_sort_batch_eval(batched_data)

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        destination = kwargs.get('saved_model_file', self.saved_model_file)
        def save():
            state = {
                'config': self.config, 'epoch': epoch, 'cur_step': self.cur_step,
                'best_valid_score': self.best_valid_score,
                'state_dict': self.model.state_dict(),
                'other_parameter': self.model.other_parameter(),
                'optimizer': self.optimizer.state_dict(),
                'qwen_provenance': self.model.qwen_provenance,
                'qwen_training_setup': self.training_setup(),
                'qwen_vector_validation': getattr(self, 'last_vector_quality', None),
            }
            # Publish a complete file so the background artifact copier never
            # observes a checkpoint halfway through torch.save.
            path = Path(destination)
            temporary = path.with_name(path.name + '.partial')
            try:
                torch.save(state, temporary, pickle_protocol=4)
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            if verbose:
                self.logger.info('Saved checkpoint: %s', destination)
            return destination
        primary_call(save, 'checkpoint save')
        if distributed_world_size() > 1:
            torch.distributed.barrier()

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        def evaluate_complete_split():
            if load_best_model:
                load_trained_checkpoint(self.model, model_file or self.saved_model_file)
            # Runtime provides complete unsharded validation/test loaders. Skip
            # RecBole's collective metric reducer on the sole evaluating rank.
            original_single_spec = self.config['single_spec']
            self.config['single_spec'] = True
            try:
                result = super(QwenTrainer, self).evaluate(
                    eval_data, load_best_model=False, show_progress=show_progress and is_primary())
                if _value(self.config, 'vector_quality_eval', False):
                    from qwen_vector_eval import evaluate_model_vectors
                    report = evaluate_model_vectors(
                        self.model, eval_data.dataset,
                        temperature=float(self.config['temperature']),
                        topk=tuple(self.config['topk']),
                        batch_size=int(self.config['eval_batch_size']))
                    summary = report['summary']
                    self.last_vector_quality = {
                        'summary': summary, 'geometry': report['geometry'],
                        'rows': len(eval_data.dataset),
                        'epoch_index': getattr(self, 'current_epoch_index', None),
                        'checkpoint_loaded': bool(load_best_model),
                    }
                    for key in ('vector_nll', 'positive_cosine_mean',
                                'margin_vs_hardest_negative_mean', 'pairwise_accuracy_mean',
                                'alignment_squared_l2_mean'):
                        result[key] = float(summary[key])
                    self.logger.info('Vector quality: %s', json.dumps(self.last_vector_quality))
                    epoch = getattr(self, 'current_epoch_index', 0)
                    for key in ('vector_nll', 'positive_cosine_mean',
                                'margin_vs_hardest_negative_mean', 'pairwise_accuracy_mean'):
                        self.tensorboard.add_scalar('Vector_quality/' + key, result[key], epoch)
                    with Path(self.saved_model_file).with_suffix('.vector_quality.jsonl').open(
                            'a', encoding='utf-8') as handle:
                        handle.write(json.dumps(self.last_vector_quality, ensure_ascii=False,
                                                allow_nan=False) + '\n')
                return result
            finally:
                self.config['single_spec'] = original_single_spec
        # Each rank's fit receives identical metrics, hence identical best-model
        # and early-stop decisions. A rank-0 exception is also sent to all ranks.
        return primary_call(evaluate_complete_split, 'evaluation')


def validate_training_launch(config_path, device=None):
    import yaml
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding='utf-8') as handle:
        configured = yaml.safe_load(handle)
    nproc = int(configured.get('nproc', 1))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != nproc:
        raise ValueError(f'This config requires torchrun --nproc-per-node={nproc}; got WORLD_SIZE={world_size}')
    if nproc > 1 and (str(device).startswith('cpu') or os.environ.get('QWEN_DDP_BACKEND') == 'gloo'):
        raise ValueError('Four-GPU training requires CUDA/NCCL; use qwen_ddp_preflight.py for CPU checks')
    if nproc > 1 and device is not None:
        raise ValueError('Do not pass --device for DDP; LOCAL_RANK selects each GPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='props/qwen_industrial_4gpu.yaml')
    parser.add_argument('--device', default=None)
    parser.add_argument('--start-training', action='store_true',
                        help='Explicitly enter trainer.fit; omit to print instructions only')
    args = parser.parse_args()
    if not args.start_training:
        print('No training started. Use qwen_preflight.py for checks; add --start-training to train.')
        return
    try:
        validate_training_launch(args.config, args.device)
    except ValueError as error:
        parser.error(str(error))
    try:
        config, dataset, (train, valid, test), model = load_experiment(args.config, args.device)
        if is_primary():
            init_logger(config)
        else:
            logging.getLogger().setLevel(logging.ERROR)
        logger = logging.getLogger()
        if is_primary():
            logger.info(config)
            logger.info(dataset)
        trainer = QwenTrainer(config, model)
        best_score, best_result = trainer.fit(
            train, valid, saved=True, verbose=is_primary(),
            show_progress=config['show_progress'] and is_primary())
        test_result = trainer.evaluate(
            test, load_best_model=True, show_progress=config['show_progress'] and is_primary())
        result = {'checkpoint': trainer.saved_model_file, 'best_valid_score': best_score,
                  'best_valid_result': best_result, 'test_result': test_result,
                  'metric_semantics': 'item ID ranking; original SID collisions retained',
                  'qwen_provenance': model.qwen_provenance,
                  'qwen_training_setup': trainer.training_setup(),
                  'vector_quality': getattr(trainer, 'last_vector_quality', None)}
        def save_result():
            Path(trainer.saved_model_file).with_suffix('.result.json').write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps(result, indent=2, ensure_ascii=False))
        primary_call(save_result, 'result save')
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
