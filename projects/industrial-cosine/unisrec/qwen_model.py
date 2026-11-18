"""Inductive UniSRec using existing frozen 2560-dimensional Qwen features."""

import torch
from torch.nn import functional as F

from qwen_dataset import _config_value
from unisrec import UniSRec


class QwenUniSRec(UniSRec):
    """A shared 2560 -> 300 adaptor for histories and retrieval candidates.

    Use ``eval()`` plus ``torch.no_grad()`` for retrieval. The encoder and MoE
    remain trainable; only the input feature table is frozen. By default, cold
    products are excluded from CE negatives as well as from positive labels.
    """

    def __init__(self, config, dataset):
        if config["train_stage"] != "inductive_ft" or config["loss_type"] != "CE":
            raise ValueError("QwenUniSRec requires train_stage=inductive_ft and loss_type=CE")
        if int(config["plm_size"]) != 2560 or int(config["hidden_size"]) != 300:
            raise ValueError("This Qwen integration requires plm_size=2560 and hidden_size=300")
        if list(config["adaptor_layers"]) != [2560, 300]:
            raise ValueError("adaptor_layers must be [2560, 300]")
        if float(config["temperature"]) <= 0:
            raise ValueError("temperature must be positive")
        super().__init__(config, dataset)
        self.plm_embedding.weight.requires_grad_(False)
        self.train_catalog_only = bool(_config_value(config, "train_catalog_only", True))
        if self.train_catalog_only:
            candidate_ids = torch.as_tensor(dataset.train_item_ids, dtype=torch.long).unique(sorted=True)
        else:
            candidate_ids = torch.arange(1, self.n_items, dtype=torch.long)
        if candidate_ids.numel() == 0 or torch.any(candidate_ids <= 0) or torch.any(candidate_ids >= self.n_items):
            raise ValueError("Training candidates must be nonempty valid internal IDs, excluding PAD")
        # Compact CE classes avoid computing all cold catalog features at each step.
        # These buffers are reconstructed from the prepared dataset on restore.
        class_by_item = torch.full((self.n_items,), -1, dtype=torch.long)
        class_by_item[candidate_ids] = torch.arange(candidate_ids.numel())
        self.register_buffer("train_candidate_ids", candidate_ids, persistent=False)
        self.register_buffer("train_class_by_item", class_by_item, persistent=False)

    def encode_features(self, features):
        """Adapt raw Qwen vectors, including vectors of products absent at training.

        Accepts a tensor or array with last dimension 2560. New feature generation
        must preserve the original Qwen model, pooling, and normalization settings.
        """
        features = torch.as_tensor(
            features, device=self.plm_embedding.weight.device, dtype=self.plm_embedding.weight.dtype
        )
        if features.ndim < 1 or features.shape[-1] != self.plm_embedding.embedding_dim:
            raise ValueError("Expected raw Qwen features with last dimension 2560")
        return F.normalize(self.moe_adaptor(features), p=2, dim=-1)

    def encode_items(self, item_ids=None):
        """Return normalized item vectors in RecBole internal-ID order by default."""
        if item_ids is None:
            item_ids = torch.arange(self.n_items, device=self.plm_embedding.weight.device)
        else:
            item_ids = torch.as_tensor(item_ids, device=self.plm_embedding.weight.device, dtype=torch.long)
        embeddings = self.encode_features(self.plm_embedding(item_ids))
        # PAD is never a candidate, even after the trainable PW bias changes.
        return embeddings.masked_fill(item_ids.eq(0).unsqueeze(-1), 0.0)

    def encode_history(self, item_seq, item_seq_len):
        """Encode nonempty, right-padded histories of internal item IDs."""
        device = self.plm_embedding.weight.device
        item_seq = torch.as_tensor(item_seq, device=device, dtype=torch.long)
        item_seq_len = torch.as_tensor(item_seq_len, device=device, dtype=torch.long)
        if item_seq.ndim != 2 or item_seq_len.shape != (item_seq.shape[0],):
            raise ValueError("Expected item_seq [batch, length] and item_seq_len [batch]")
        if item_seq.shape[1] > self.max_seq_length:
            raise ValueError("History exceeds MAX_ITEM_LIST_LENGTH; truncate to the most recent items")
        if torch.any(item_seq_len <= 0) or torch.any(item_seq_len > item_seq.shape[1]):
            raise ValueError("History lengths must be positive and fit the padded sequence")
        # Keep upstream history behavior: the adaptor output is not normalized
        # until after sequence encoding; candidate vectors are normalized directly.
        item_emb = self.moe_adaptor(self.plm_embedding(item_seq))
        output = self.forward(item_seq, item_emb, item_seq_len)
        return F.normalize(output, p=2, dim=-1)

    def calculate_loss(self, interaction):
        queries = self.encode_history(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        target_ids = interaction[self.POS_ITEM_ID]
        targets = self.train_class_by_item[target_ids]
        if torch.any(targets < 0):
            raise ValueError("A training target is PAD or absent from train_item_ids.json")
        candidates = self.encode_items(self.train_candidate_ids)
        logits = queries @ candidates.transpose(0, 1) / self.temperature
        return self.loss_fct(logits, targets)

    def full_sort_predict(self, interaction):
        queries = self.encode_history(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        scores = queries @ self.encode_items().transpose(0, 1)
        scores[:, 0] = -torch.inf
        return scores

    def predict(self, interaction):
        queries = self.encode_history(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        candidates = self.encode_items(interaction[self.ITEM_ID])
        scores = (queries * candidates).sum(dim=-1)
        return scores.masked_fill(interaction[self.ITEM_ID].eq(0), -torch.inf)
