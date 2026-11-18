"""Configuration and trusted local checkpoint loading, shared by train/retrieval."""

import hashlib
import json
import os
import copy
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_OWNS_PROCESS_GROUP = False


def distributed_rank():
    """The actual initialized rank, or zero outside a process group."""
    import torch.distributed as dist
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def distributed_world_size():
    import torch.distributed as dist
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_primary():
    return distributed_rank() == 0


def cleanup_distributed():
    """Idempotently close only the process group initialized by this runtime."""
    global _OWNS_PROCESS_GROUP
    import torch.distributed as dist
    if _OWNS_PROCESS_GROUP and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _OWNS_PROCESS_GROUP = False


def runtime_environment(environ=None):
    """Read torchrun's ranks; nproc in YAML alone never launches processes."""
    env = os.environ if environ is None else environ
    try:
        world_size = int(env.get('WORLD_SIZE', '1'))
        rank = int(env.get('RANK', '0'))
        local_rank = int(env.get('LOCAL_RANK', '0'))
        local_world_size = int(env.get('LOCAL_WORLD_SIZE', str(world_size)))
    except ValueError as error:
        raise ValueError('torchrun rank environment values must be integers') from error
    if world_size < 1 or local_world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid torchrun world size or rank')
    if not 0 <= local_rank < local_world_size or world_size % local_world_size:
        raise ValueError('Invalid torchrun local rank or local world size')
    if world_size > 1:
        required = ('RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')
        if any(not env.get(key) for key in required):
            raise ValueError('Use torchrun to supply ranks and MASTER_ADDR/MASTER_PORT')
    return {'rank': rank, 'local_rank': local_rank, 'world_size': world_size,
            'local_world_size': local_world_size}


def _configure_device(final_config, requested_device=None):
    """Own distributed initialization so RecBole does not initialize it again."""
    global _OWNS_PROCESS_GROUP
    import torch
    import torch.distributed as dist

    context = runtime_environment()
    distributed = context['world_size'] > 1
    configured_nproc = int(final_config.get('nproc', 1))
    backend = os.environ.get('QWEN_DDP_BACKEND', 'nccl').lower()
    if backend not in ('nccl', 'gloo'):
        raise ValueError('QWEN_DDP_BACKEND must be nccl, or gloo for CPU checks')
    explicit_cpu = str(requested_device) == 'cpu'
    use_cpu = explicit_cpu or not bool(final_config.get('use_gpu', True)) or backend == 'gloo'

    final_config.update({
        'single_spec': not distributed, 'local_rank': context['local_rank'],
        'world_size': context['world_size'], 'qwen_rank': context['rank'],
        'qwen_world_size': context['world_size'], 'qwen_local_world_size': context['local_world_size'],
        'qwen_requested_nproc': configured_nproc,
    })
    if distributed:
        if configured_nproc != context['local_world_size']:
            raise ValueError('Configured nproc must match torchrun LOCAL_WORLD_SIZE')
        if int(final_config['train_batch_size']) % context['world_size']:
            raise ValueError('Global train_batch_size must be divisible by WORLD_SIZE')
        if use_cpu and backend != 'gloo':
            raise ValueError('CPU distributed checks require QWEN_DDP_BACKEND=gloo')
        if backend == 'nccl':
            # An existing CUDA_VISIBLE_DEVICES is an explicit physical-device map.
            # Never replace it with logical local ranks after CUDA initialization.
            if 'CUDA_VISIBLE_DEVICES' not in os.environ and not torch.cuda.is_initialized():
                os.environ['CUDA_VISIBLE_DEVICES'] = str(final_config['gpu_id'])
            if not torch.cuda.is_available() or torch.cuda.device_count() < context['local_world_size']:
                raise RuntimeError('Four-card training requires enough visible CUDA devices')
            torch.cuda.set_device(context['local_rank'])
            selected = torch.device('cuda', context['local_rank'])
        else:
            selected = torch.device('cpu')
        if dist.is_initialized():
            if (dist.get_rank(), dist.get_world_size(), dist.get_backend()) != (
                    context['rank'], context['world_size'], backend):
                raise RuntimeError('Existing process group differs from torchrun configuration')
        else:
            dist.init_process_group(backend=backend, init_method='env://',
                                    rank=context['rank'], world_size=context['world_size'],
                                    timeout=timedelta(seconds=180))
            _OWNS_PROCESS_GROUP = True
    elif use_cpu:
        selected = torch.device('cpu')
    else:
        # An explicit cuda:N is a logical PyTorch device request. Do not hide
        # that device by replacing the visible-device map with YAML's gpu_id.
        if (requested_device is None and 'CUDA_VISIBLE_DEVICES' not in os.environ
                and not torch.cuda.is_initialized()):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(final_config['gpu_id'])
        if not torch.cuda.is_available():
            if requested_device is not None:
                raise RuntimeError('Requested CUDA device is unavailable')
            selected = torch.device('cpu')
        else:
            selected = torch.device('cuda:0' if requested_device in (None, 'cuda') else requested_device)
            torch.cuda.set_device(selected)
    final_config['device'] = selected
    final_config['use_gpu'] = selected.type == 'cuda'
    final_config['qwen_ddp_backend'] = backend if distributed else None
    if context['rank'] != 0:
        final_config.update({'show_progress': False, 'verbose': False, 'state': 'error'})


class _DeviceConfigMixin:
    def _set_default_parameters(self):
        super()._set_default_parameters()
        # RecBole derives this flag from its registered metric names, and
        # would otherwise maximize our manually supplied vector NLL metric.
        if str(self.final_config_dict.get('valid_metric', '')).lower() == 'vector_nll':
            if not self.final_config_dict.get('vector_quality_eval', False):
                raise ValueError('valid_metric=vector_nll requires vector_quality_eval=true')
            self.final_config_dict['valid_metric_bigger'] = False

    def _init_device(self):
        _configure_device(self.final_config_dict,
                          self.final_config_dict.get('qwen_requested_device'))


def _config_class():
    # Lazy loading keeps data-manifest helpers usable without PyTorch/RecBole.
    # Publish the class at module scope so RecBole checkpoints remain picklable.
    if 'QwenConfig' not in globals():
        from recbole.config import Config
        globals()['QwenConfig'] = type('QwenConfig', (_DeviceConfigMixin, Config),
                                     {'__module__': __name__})
    return globals()['QwenConfig']


def __getattr__(name):
    if name == 'QwenConfig':
        return _config_class()
    raise AttributeError(name)


def _prepare_loaders(config, dataset):
    """Build once: shard train only, keep exact complete evaluation datasets."""
    from recbole.data.utils import create_samplers, get_dataloader
    built = dataset.build()
    samplers = create_samplers(config, dataset, built)
    train = get_dataloader(config, 'train')(
        config, built[0], samplers[0], shuffle=config['shuffle'])
    eval_config = copy.copy(config)
    eval_config.final_config_dict = dict(config.final_config_dict)
    eval_config['single_spec'] = True
    valid = get_dataloader(eval_config, 'valid')(
        eval_config, built[1], samplers[1], shuffle=False)
    test = get_dataloader(eval_config, 'test')(
        eval_config, built[2], samplers[2], shuffle=False)
    return train, valid, test


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prepared(prepared, dataset_name, max_history):
    prepared = Path(prepared)
    manifest_path = prepared / 'manifest.json'
    with manifest_path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest['dataset'] != dataset_name or manifest['max_history_len'] != max_history:
        raise ValueError('Prepared dataset/history length differs from configuration')
    expected = manifest.get('prepared_sha256', {})
    required = {dataset_name + suffix for suffix in
                ('.item', '.train.inter', '.valid.inter', '.test.inter',
                 '.emb-qwen-td.npy', '.emb-qwen-td.npy.ids.json')}
    required.update(('catalog.json', 'train_item_ids.json'))
    if not required.issubset(expected):
        raise ValueError('Prepared manifest is missing required file hashes')
    for filename, checksum in expected.items():
        if Path(filename).name != filename or sha256_file(prepared / filename) != checksum:
            raise ValueError('Prepared file differs from manifest: ' + filename)
    return manifest


def load_experiment(config_path='props/qwen_industrial.yaml', device=None):
    import torch
    import yaml
    from recbole.utils import init_seed
    from qwen_dataset import QwenUniSRecDataset
    from qwen_model import QwenUniSRec

    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding='utf-8') as handle:
        overrides = yaml.safe_load(handle)
    dataset_name = overrides.pop('dataset')
    for key in ('data_path', 'embedding_path', 'embedding_ids_path',
                'train_item_ids_path', 'catalog_path', 'checkpoint_dir'):
        if overrides.get(key) is not None:
            value = Path(overrides[key])
            overrides[key] = str(value if value.is_absolute() else ROOT / value)
    overrides['qwen_requested_device'] = str(device) if device is not None else None
    config = _config_class()(model=QwenUniSRec, dataset=dataset_name,
                            config_file_list=[str(ROOT / 'props/UniSRec.yaml')], config_dict=overrides)
    init_seed(config['seed'], config['reproducibility'])
    prepared = Path(config['data_path'])
    manifest_path = prepared / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Run prepare_qwen_data.py first: ' + str(manifest_path))
    validate_prepared(prepared, dataset_name, int(config['MAX_ITEM_LIST_LENGTH']))
    dataset = QwenUniSRecDataset(config)
    loaders = _prepare_loaders(config, dataset)
    model = QwenUniSRec(config, loaders[0].dataset).to(config['device'])
    tokens = [str(v) for v in dataset.field2id_token[dataset.iid_field]]
    provenance = {
        'dataset': dataset_name,
        'embedding_sha256': sha256_file(dataset.embedding_path),
        'embedding_ids_sha256': sha256_file(dataset.embedding_ids_path),
        'catalog_sha256': sha256_file(prepared / 'catalog.json'),
        'manifest_sha256': sha256_file(manifest_path),
        'internal_item_tokens_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        'train_catalog_only': bool(config['train_catalog_only']),
        'max_history_length': int(config['MAX_ITEM_LIST_LENGTH']),
        'architecture': {key: config[key] for key in (
            'train_stage', 'loss_type', 'plm_size', 'adaptor_layers', 'n_exps',
            'n_layers', 'n_heads', 'hidden_size', 'inner_size', 'hidden_act',
            'layer_norm_eps', 'temperature', 'hidden_dropout_prob',
            'attn_dropout_prob', 'adaptor_dropout_prob')},
    }
    model.qwen_provenance = provenance
    return config, dataset, loaders, model


def load_trained_checkpoint(model, checkpoint_path):
    """Only load a trusted checkpoint created by qwen_train.QwenTrainer."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location=next(model.parameters()).device,
                            weights_only=False)
    if not isinstance(checkpoint.get('epoch'), int) or checkpoint['epoch'] < 0:
        raise ValueError('A trained Qwen UniSRec checkpoint with epoch >= 0 is required')
    if checkpoint.get('qwen_provenance') != model.qwen_provenance:
        raise ValueError('Checkpoint feature/catalog/split/ID mapping or settings differ')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.load_other_parameter(checkpoint.get('other_parameter'))
    return checkpoint
