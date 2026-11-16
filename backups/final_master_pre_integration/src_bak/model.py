"""
Complete Seq2Seq Summarization Model (Encoder + Bahdanau Attention + Decoder).
Supports Teacher Forcing during training, Loss calculation compatibility,
Inference summary generation, and automatic device detection (MPS / CPU / CUDA).
"""

import random
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any

from src.encoder import Encoder
from src.decoder import Decoder
from src.preprocess import PAD_IDX, SOS_IDX, EOS_IDX


def get_device() -> torch.device:
    """
    Automatically selects Apple Silicon MPS if available, otherwise CUDA or CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence Architecture for Document Summarization.
    
    Combines:
    - Bidirectional LSTM Encoder
    - Bahdanau Additive Attention
    - Attention-Augmented LSTM Decoder
    """
    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        dropout: float = 0.0,
        pad_idx: int = PAD_IDX,
        sos_idx: int = SOS_IDX,
        eos_idx: int = EOS_IDX
    ):
        super(Seq2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        # Instantiate Encoder
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
            padding_idx=pad_idx
        )

        # Instantiate Decoder (contains Bahdanau Attention)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            dropout=dropout,
            padding_idx=pad_idx
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass during training / validation.
        
        Args:
            src: Source text token indices of shape [batch_size, source_length].
            tgt: Target summary token indices of shape [batch_size, target_length].
            teacher_forcing_ratio: Probability of feeding ground-truth target token
                                   as next decoder input instead of model prediction.
                                   
        Returns:
            outputs: Unnormalized vocabulary logits of shape [batch_size, target_length - 1, vocab_size].
        """
        batch_size = src.shape[0]
        target_len = tgt.shape[1]

        # Create source padding mask (1 for valid content, 0 for <PAD>)
        mask = (src != self.pad_idx)

        # Step 1: Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src)

        # Step 2: Initialize output logits tensor of shape [batch_size, target_length - 1, vocab_size]
        outputs = torch.zeros(batch_size, target_len - 1, self.vocab_size, device=src.device)

        # First input token to the decoder is always <SOS> (column 0 of target batch)
        decoder_input = tgt[:, 0]

        # Step 3: Decode sequence step-by-step for target_len - 1 steps
        for t in range(target_len - 1):
            # Run single step decoder
            logits, hidden, cell, _ = self.decoder(
                input_token=decoder_input,
                hidden_state=hidden,
                cell_state=cell,
                encoder_outputs=encoder_outputs,
                mask=mask
            )

            # Store logits for current step t
            outputs[:, t, :] = logits

            # Decide next input token via Teacher Forcing
            use_teacher_forcing = (random.random() < teacher_forcing_ratio)
            top1_prediction = logits.argmax(dim=1)

            decoder_input = tgt[:, t + 1] if use_teacher_forcing else top1_prediction

        return outputs

    def compute_loss(
        self,
        logits: torch.Tensor,
        tgt: torch.Tensor,
        criterion: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Computes CrossEntropyLoss ignoring <PAD> tokens and skipping initial <SOS> token.
        
        Args:
            logits: Unnormalized logits of shape [batch_size, target_length - 1, vocab_size].
            tgt: Ground truth target token indices of shape [batch_size, target_length].
            criterion: Optional PyTorch loss function (defaults to CrossEntropyLoss(ignore_index=pad_idx)).
            
        Returns:
            loss: Scalar PyTorch loss tensor.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        # Flatten logits: [(target_len - 1) * batch_size, vocab_size]
        output_dim = logits.shape[-1]
        logits_flat = logits.reshape(-1, output_dim)
        
        # Flatten targets: [(target_len - 1) * batch_size] (skipping <SOS> token at index 0)
        tgt_flat = tgt[:, 1:].reshape(-1)

        loss = criterion(logits_flat, tgt_flat)
        return loss

    def _block_repeat_ngrams(
        self,
        tokens: List[int],
        logits: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Suppresses candidate tokens that would complete an already emitted n-gram.
        
        Args:
            tokens: Previously generated token sequence.
            logits: 1D unnormalized logits tensor for vocabulary [vocab_size].
            n: N-gram size to block (e.g. 3 for trigram blocking).
            
        Returns:
            logits: Logits tensor with banned token positions masked to -1e9.
        """
        if n <= 0 or len(tokens) < n - 1:
            return logits

        prefix = tuple(tokens[-(n - 1):])
        banned = set()
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i : i + n - 1]) == prefix:
                banned.add(tokens[i + n - 1])

        if banned:
            mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
            for tok in banned:
                if tok < logits.size(-1):
                    mask[tok] = True

            # Fail-safe: ensure we do not mask all tokens
            if (~mask).any():
                logits[mask] = -1e9

        return logits

    @torch.no_grad()
    def generate_summary(
        self,
        src_tensor: torch.Tensor,
        vocab: Any,
        max_target_len: int = 60,
        device: Optional[torch.device] = None,
        strategy: str = "greedy",
        beam_width: int = 4,
        no_repeat_ngram_size: int = 0,
        length_penalty: float = 1.0,
    ) -> Tuple[str, List[int], torch.Tensor]:
        """
        Summary generation inference method supporting Greedy decoding, Beam Search,
        and no-repeat n-gram blocking.
        
        Args:
            src_tensor: Source document tensor of shape [1, source_length] or [source_length].
            vocab: Vocabulary object instance for decoding token indices to words.
            max_target_len: Maximum summary token length.
            device: Optional torch.device.
            strategy: Decoding strategy ('greedy' or 'beam_search' / 'beam').
            beam_width: Number of active hypotheses for beam search (default 4).
            no_repeat_ngram_size: If > 0, prevents repeated n-grams of this length (e.g. 3 for trigrams).
            length_penalty: Alpha exponent for Wu et al. length normalization in beam search.
            
        Returns:
            generated_summary: Decoded text string of the generated summary.
            predicted_indices: List of generated token indices.
            attention_matrix: Tensor of shape [generated_steps, source_length] containing attention weights.
        """
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        if src_tensor.dim() == 1:
            src_tensor = src_tensor.unsqueeze(0)  # Shape [1, source_length]

        src_tensor = src_tensor.to(device)
        mask = (src_tensor != self.pad_idx)

        # Encode source document
        encoder_outputs, enc_hidden, enc_cell = self.encoder(src_tensor)

        strategy_lower = strategy.lower().strip()

        # ---------------------------------------------------------------------
        # 1. Greedy Decoding (Baseline & Default)
        # ---------------------------------------------------------------------
        if strategy_lower == "greedy":
            decoder_input = torch.tensor([self.sos_idx], dtype=torch.long, device=device)
            hidden = enc_hidden
            cell = enc_cell

            predicted_indices: List[int] = []
            attn_weights_list: List[torch.Tensor] = []

            for _ in range(max_target_len - 1):
                logits, hidden, cell, attn_weights = self.decoder(
                    input_token=decoder_input,
                    hidden_state=hidden,
                    cell_state=cell,
                    encoder_outputs=encoder_outputs,
                    mask=mask,
                )

                attn_weights_list.append(attn_weights.squeeze(0).cpu())

                step_logits = logits[0].clone()
                # Suppress special tokens (PAD and SOS) from candidate emissions
                step_logits[self.pad_idx] = -1e9
                step_logits[self.sos_idx] = -1e9

                if no_repeat_ngram_size > 0:
                    step_logits = self._block_repeat_ngrams(
                        predicted_indices, step_logits, no_repeat_ngram_size
                    )

                pred_token_id = step_logits.argmax().item()

                if pred_token_id == self.eos_idx:
                    break

                predicted_indices.append(pred_token_id)
                decoder_input = torch.tensor([pred_token_id], dtype=torch.long, device=device)

            generated_summary = vocab.decode(predicted_indices, skip_special_tokens=True)
            attention_matrix = torch.stack(attn_weights_list) if attn_weights_list else torch.empty(0)
            return generated_summary, predicted_indices, attention_matrix

        # ---------------------------------------------------------------------
        # 2. Beam Search Decoding
        # ---------------------------------------------------------------------
        elif strategy_lower in ("beam", "beam_search"):
            active_beams = [{
                "tokens": [],
                "score": 0.0,
                "hidden": enc_hidden,
                "cell": enc_cell,
                "attn_list": [],
                "last_token": self.sos_idx,
            }]
            finished_beams: List[Dict[str, Any]] = []

            for step in range(max_target_len - 1):
                candidates = []

                for beam in active_beams:
                    dec_in = torch.tensor([beam["last_token"]], dtype=torch.long, device=device)
                    logits, next_h, next_c, attn_weights = self.decoder(
                        input_token=dec_in,
                        hidden_state=beam["hidden"],
                        cell_state=beam["cell"],
                        encoder_outputs=encoder_outputs,
                        mask=mask,
                    )

                    step_logits = logits[0].clone()
                    # Filter special tokens from candidate emissions
                    step_logits[self.pad_idx] = -1e9
                    step_logits[self.sos_idx] = -1e9

                    # Apply n-gram blocking if requested
                    if no_repeat_ngram_size > 0 and len(beam["tokens"]) >= no_repeat_ngram_size - 1:
                        step_logits = self._block_repeat_ngrams(
                            beam["tokens"], step_logits, no_repeat_ngram_size
                        )

                    log_probs = torch.log_softmax(step_logits, dim=-1)

                    k_candidates = min(beam_width, log_probs.size(-1))
                    topk_log_probs, topk_indices = torch.topk(log_probs, k=k_candidates)

                    attn_step = attn_weights.squeeze(0).cpu()

                    for lp, tok_idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                        if lp <= -1e8:
                            continue  # Skip severely masked tokens
                        candidates.append({
                            "tokens": beam["tokens"] + [tok_idx],
                            "score": beam["score"] + lp,
                            "hidden": next_h,
                            "cell": next_c,
                            "attn_list": beam["attn_list"] + [attn_step],
                            "last_token": tok_idx,
                        })

                if not candidates:
                    # Fail-safe: if all candidates were masked
                    break

                # Sort candidates descending by accumulated log-prob score
                candidates.sort(key=lambda x: x["score"], reverse=True)

                new_active = []
                for cand in candidates:
                    if cand["last_token"] == self.eos_idx:
                        # Length normalization: score / ((5 + L) / 6) ^ length_penalty
                        seq_len = len(cand["tokens"])
                        lp_factor = ((5.0 + seq_len) / 6.0) ** length_penalty if length_penalty > 0 else 1.0
                        cand["norm_score"] = cand["score"] / lp_factor
                        # Exclude EOS from final token sequence
                        cand["tokens"] = cand["tokens"][:-1]
                        finished_beams.append(cand)
                    else:
                        new_active.append(cand)
                        if len(new_active) == beam_width:
                            break

                active_beams = new_active

                # Terminate early if enough beams finished or no active beams remain
                if len(finished_beams) >= beam_width or len(active_beams) == 0:
                    break

            # Select best hypothesis
            if finished_beams:
                finished_beams.sort(key=lambda x: x["norm_score"], reverse=True)
                best_beam = finished_beams[0]
            elif active_beams:
                for b in active_beams:
                    seq_len = max(1, len(b["tokens"]))
                    lp_factor = ((5.0 + seq_len) / 6.0) ** length_penalty if length_penalty > 0 else 1.0
                    b["norm_score"] = b["score"] / lp_factor
                active_beams.sort(key=lambda x: x["norm_score"], reverse=True)
                best_beam = active_beams[0]
            else:
                best_beam = {"tokens": [], "attn_list": []}

            best_tokens = best_beam["tokens"]
            best_attn = best_beam["attn_list"]

            generated_summary = vocab.decode(best_tokens, skip_special_tokens=True)
            attention_matrix = torch.stack(best_attn) if best_attn else torch.empty(0)
            return generated_summary, best_tokens, attention_matrix

        else:
            raise ValueError(
                f"Unsupported decoding strategy: '{strategy}'. Choose 'greedy' or 'beam_search'."
            )

    def count_parameters(self) -> Dict[str, int]:
        """
        Returns parameter counts for Encoder, Attention, Decoder, and Total model.
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "Encoder": encoder_params,
            "Decoder (including Attention)": decoder_params,
            "Total Trainable Parameters": total_params
        }
