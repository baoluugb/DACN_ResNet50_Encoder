"""HMER Model with Auxiliary Tasks

Single-stage encoder-decoder architecture for Handwritten Mathematical Expression Recognition.
Uses image encoder + transformer decoder with auxiliary syntax/structure prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .encoder import Encoder
from .latex_decoder import LatexDecoderWithAux
from .losses import compute_main_loss, compute_aux_ce_loss, compute_coverage_loss, combine_losses


class HMERWithAuxiliary(nn.Module):
    """End-to-end HMER model with auxiliary structural predictions"""

    def __init__(self, config: Dict, latex_vocab: Dict):
        super().__init__()

        self.config = config
        self.latex_vocab = latex_vocab

        # Extract vocabulary info
        self.token_to_id = latex_vocab["token_to_id"]
        self.id_to_token = latex_vocab["id_to_token"]
        self.special = latex_vocab["special"]

        self.pad_id = self.token_to_id[self.special["pad"]]
        self.bos_id = self.token_to_id[self.special["bos"]]
        self.eos_id = self.token_to_id[self.special["eos"]]
        self.vocab_size = len(self.token_to_id)

        # Extract config
        self.d_model = config.get("d_model", 512)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 6)
        self.ffn_dim = config.get("ffn_dim", 2048)
        self.dropout = config.get("dropout", 0.1)
        self.max_len = config.get("max_len", 512)
        self.num_soft_prompts = config.get("num_soft_prompts", 0)

        # Loss weights
        self.aux_type_loss_weight = config.get("aux_type_loss_weight", 0.1)
        self.aux_depth_loss_weight = config.get("aux_depth_loss_weight", 0.1)
        self.aux_rel_loss_weight = config.get("aux_rel_loss_weight", 0.1)
        self.coverage_loss_weight = config.get("coverage_loss_weight", 0.01)

        # Auxiliary head enable flags
        enable_type_head = config.get("enable_type_head", True)
        enable_depth_head = config.get("enable_depth_head", True)
        enable_rel_head = config.get("enable_rel_head", True)

        # Encoder config
        encoder_backbone = config.get("encoder_backbone", "resnet34")
        encoder_pos_max_h = config.get("encoder_pos_max_h", 64)
        encoder_pos_max_w = config.get("encoder_pos_max_w", 256)

        # Image encoder
        print(
            f"[INFO] Initializing ResNet encoder: backbone={encoder_backbone}, d_model={self.d_model}")
        self.image_encoder = Encoder(
            d_model=self.d_model,
            backbone=encoder_backbone,
            pos_max_h=encoder_pos_max_h,
            pos_max_w=encoder_pos_max_w
        )

        # LaTeX decoder with auxiliary heads
        print(
            f"[INFO] Initializing decoder: {self.num_layers} layers, vocab={self.vocab_size}")
        self.decoder = LatexDecoderWithAux(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            num_soft_prompts=self.num_soft_prompts,
            enable_type_head=enable_type_head,
            enable_depth_head=enable_depth_head,
            enable_rel_head=enable_rel_head
        )

        self.decoder.pad_id = self.pad_id
        print(
            f"[INFO] Model initialized: {self.count_parameters():,} parameters")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, images: torch.Tensor, target_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory, _ = self.image_encoder(images)
        decoder_input = target_ids[:, :-1]
        outputs = self.decoder(tgt_ids=decoder_input,
                               memory=memory, pad_id=self.pad_id)
        return outputs

    def compute_loss(self, batch: Dict, aux_targets: Dict[str, torch.Tensor],
                     criterion_main: Optional[nn.Module] = None,
                     criterion_aux: Optional[Dict[str, nn.Module]] = None,
                     label_smoothing: float = 0.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        images = batch["images"]
        target_ids = batch["target_ids"]
        outputs = self.forward(images, target_ids)

        logits = outputs["logits"]
        type_logits = outputs.get("type_logits")
        depth_logits = outputs.get("depth_logits")
        rel_logits = outputs.get("rel_logits")
        coverage = outputs.get("coverage")

        main_targets = target_ids[:, 1:]

        if criterion_main is not None:
            main_loss = criterion_main(logits, main_targets)
        else:
            main_loss = compute_main_loss(
                logits, main_targets, pad_id=self.pad_id, label_smoothing=label_smoothing)

        aux_losses = {}

        if type_logits is not None and aux_targets.get("type_targets") is not None:
            type_targets = aux_targets["type_targets"][:, 1:]
            aux_losses["type"] = compute_aux_ce_loss(
                type_logits, type_targets, ignore_index=0)

        if depth_logits is not None and aux_targets.get("depth_targets") is not None:
            depth_targets = aux_targets["depth_targets"][:, 1:]
            aux_losses["depth"] = compute_aux_ce_loss(
                depth_logits, depth_targets, ignore_index=0)

        if rel_logits is not None and aux_targets.get("rel_targets") is not None:
            rel_targets = aux_targets["rel_targets"][:, 1:]
            aux_losses["rel"] = compute_aux_ce_loss(
                rel_logits, rel_targets, ignore_index=0)

        if coverage is not None and self.coverage_loss_weight > 0:
            aux_losses["coverage"] = compute_coverage_loss(coverage)

        weights = {
            "type": self.aux_type_loss_weight,
            "depth": self.aux_depth_loss_weight,
            "rel": self.aux_rel_loss_weight,
            "coverage": self.coverage_loss_weight
        }

        total_loss, loss_dict = combine_losses(main_loss, aux_losses, weights)
        return total_loss, loss_dict

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int = 200, beam_size: int = 1,
                 length_penalty: float = 1.0, min_len: int = 1, temperature: float = 1.0) -> List[List[int]]:
        self.eval()
        B = images.size(0)
        device = images.device
        memory, _ = self.image_encoder(images)

        if beam_size == 1:
            return self._greedy_decode(memory, max_len, min_len, temperature)
        else:
            return self._beam_search(memory, max_len, beam_size, length_penalty, min_len)

    def _greedy_decode(self, memory: torch.Tensor, max_len: int, min_len: int, temperature: float) -> List[List[int]]:
        B = memory.size(0)
        device = memory.device
        current_ids = torch.full(
            (B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            outputs = self.decoder(tgt_ids=current_ids,
                                   memory=memory, pad_id=self.pad_id)
            logits = outputs["logits"][:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            logits[:, self.pad_id] = -float('inf')
            if step < min_len:
                logits[:, self.eos_id] = -float('inf')

            next_tokens = logits.argmax(dim=-1)
            finished = finished | (next_tokens == self.eos_id)
            next_tokens = next_tokens.masked_fill(finished, self.pad_id)
            current_ids = torch.cat(
                [current_ids, next_tokens.unsqueeze(1)], dim=1)

            if finished.all():
                break

        results = []
        for b in range(B):
            seq = current_ids[b, 1:].tolist()
            if self.eos_id in seq:
                seq = seq[:seq.index(self.eos_id)]
            seq = [t for t in seq if t != self.pad_id]
            results.append(seq)

        return results

    def _beam_search(self, memory: torch.Tensor, max_len: int, beam_size: int,
                     length_penalty: float, min_len: int) -> List[List[int]]:
        B = memory.size(0)
        device = memory.device
        results = []

        for b in range(B):
            mem = memory[b:b+1]
            beams = [(0.0, [self.bos_id])]

            for step in range(max_len):
                candidates = []
                for score, seq in beams:
                    if seq[-1] == self.eos_id:
                        candidates.append((score, seq))
                        continue

                    current_ids = torch.tensor(
                        [seq], dtype=torch.long, device=device)
                    outputs = self.decoder(
                        tgt_ids=current_ids, memory=mem, pad_id=self.pad_id)
                    logits = outputs["logits"][0, -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)

                    log_probs[self.pad_id] = -float('inf')
                    if len(seq) < min_len + 1:
                        log_probs[self.eos_id] = -float('inf')

                    topk_probs, topk_ids = log_probs.topk(beam_size)
                    for prob, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
                        new_score = score + prob
                        new_seq = seq + [tid]
                        candidates.append((new_score, new_seq))

                scored_candidates = [
                    (s / (len(seq) ** length_penalty), s, seq) for s, seq in candidates]
                scored_candidates.sort(reverse=True, key=lambda x: x[0])
                beams = [(s, seq)
                         for _, s, seq in scored_candidates[:beam_size]]

                if all(seq[-1] == self.eos_id for _, seq in beams):
                    break

            best_score, best_seq = beams[0]
            result = best_seq[1:]
            if self.eos_id in result:
                result = result[:result.index(self.eos_id)]
            results.append(result)

        return results
