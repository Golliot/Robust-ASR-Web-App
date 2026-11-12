"""Modèle RobustASR du notebook (cellule 17) : frontend CNN, encodeur Transformer, têtes CTC et attention.

Le code est celui du notebook, à une retouche près : `autocast` reçoit le type de périphérique au lieu
de lire la variable globale `device` du notebook (bf16 sur CUDA, désactivé ailleurs, comme avant).
Les choix d'architecture (LayerNorm du frontend, QK-norm, calcul incrémental du décodeur) sont
documentés dans la section 8 du notebook.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.asr.features import LogMel, SpecAugment, make_pad_mask


def autocast(device_type):
    """bf16 sur CUDA, désactivé ailleurs. Le notebook lisait la variable globale `device`."""
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_length=6000):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        scales = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales)
        self.register_buffer("encoding", encoding, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.encoding[: x.size(1)])


class ConvFrontend(nn.Module):
    """Deux Conv2d de pas 2 : le temps et la fréquence sont divisés par 4."""

    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, cfg.conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(cfg.conv_channels, cfg.conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        ).to(memory_format=torch.channels_last)
        self.projection = nn.Linear(cfg.conv_channels * ((cfg.n_mels + 3) // 4), cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, feats, feat_lens):
        # channels_last ne change que la disposition en mémoire : les dimensions restent (B, C, T, F).
        x = self.conv(feats.unsqueeze(1).to(memory_format=torch.channels_last))  # (B, C, T/4, F/4)
        x = x.transpose(1, 2).flatten(2)  # (B, T/4, C × F/4)
        return self.norm(self.projection(x)), ((feat_lens + 1) // 2 + 1) // 2


def allowed_keys(pad_mask):
    """Masque de padding (B, T), True sur le padding -> masque d'attention (B, 1, 1, T), True si permis."""
    return ~pad_mask[:, None, None, :]


def rms_norm(x, gain=None):
    """RMSNorm sur la dernière dimension, dans le type de x (calcul interne en float32).

    Hors autocast, qui la ferait passer en float32 : 5 % de débit en plus.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.rms_norm(x, (x.size(-1),), None if gain is None else gain.to(x.dtype), eps=1e-6)


class Attention(nn.Module):
    """Attention multi-têtes. Avec qk_norm, q et k sont normalisés par tête avant leur produit scalaire."""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.dropout = cfg.dropout
        self.qk_norm = cfg.qk_norm
        self.query = nn.Linear(cfg.d_model, cfg.d_model)
        self.key_value = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.d_model)
        # Un logit ne dépend que du produit des gains de q et de k : un seul vecteur, porté par q, suffit.
        self.query_gain = nn.Parameter(torch.ones(cfg.d_model // cfg.n_heads)) if cfg.qk_norm else None

    def _split_heads(self, x):
        return x.unflatten(-1, (self.n_heads, -1))  # (B, T, H, D/H)

    def forward(self, query, memory, allowed=None):
        """allowed : booléen diffusable vers (B, H, Tq, Tk), True là où l'attention est permise."""
        key, value = self.key_value(memory).chunk(2, dim=-1)
        q, k, v = self._split_heads(self.query(query)), self._split_heads(key), self._split_heads(value)
        if self.qk_norm:
            q, k = rms_norm(q, self.query_gain), rms_norm(k)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))  # (B, H, T, D/H)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0)
        return self.output(x.transpose(1, 2).flatten(2))


def feed_forward(cfg):
    return nn.Sequential(
        nn.Linear(cfg.d_model, cfg.d_ffn),
        nn.ReLU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.d_ffn, cfg.d_model),
    )


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.ffn = feed_forward(cfg)
        self.norm_self = nn.LayerNorm(cfg.d_model)
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, allowed):
        query = self.norm_self(x)
        x = x + self.dropout(self.self_attn(query, query, allowed))
        return x + self.dropout(self.ffn(self.norm_ffn(x)))


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.encoder_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, pad_mask):
        allowed = allowed_keys(pad_mask)
        for layer in self.layers:
            x = layer(x, allowed)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.cross_attn = Attention(cfg)
        self.ffn = feed_forward(cfg)
        self.norm_self = nn.LayerNorm(cfg.d_model)
        self.norm_cross = nn.LayerNorm(cfg.d_model)
        self.norm_ffn = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def _attend_memory_and_ffn(self, x, memory, memory_allowed):
        x = x + self.dropout(self.cross_attn(self.norm_cross(x), memory, memory_allowed))
        return x + self.dropout(self.ffn(self.norm_ffn(x)))

    def forward(self, x, memory, memory_allowed, self_allowed):
        query = self.norm_self(x)
        attended = self.self_attn(query, query, self_allowed)
        return self._attend_memory_and_ffn(x + self.dropout(attended), memory, memory_allowed)

    def step(self, x, memory, memory_allowed):
        """Calcule la sortie de la dernière position de x : (N, 1, D)."""
        keys = self.norm_self(x)
        attended = self.self_attn(keys[:, -1:], keys)  # la dernière position voit tout le préfixe
        return self._attend_memory_and_ffn(x[:, -1:] + attended, memory, memory_allowed)


class AttentionDecoder(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.positional = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.decoder_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, tokens, memory, memory_pad_mask, token_pad_mask):
        length = tokens.size(1)
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
        self_allowed = causal & allowed_keys(token_pad_mask)
        memory_allowed = allowed_keys(memory_pad_mask)
        x = self.positional(self.embedding(tokens))
        for layer in self.layers:
            x = layer(x, memory, memory_allowed, self_allowed)
        return self.output(self.norm(x))

    def initial_cache(self, memory):
        return [memory.new_zeros(memory.size(0), 0, memory.size(2)) for _ in self.layers]

    def step(self, tokens, memory, memory_pad_mask, cache):
        """Un pas incrémental. cache[i] contient les sorties de la couche i, dernier token exclu."""
        memory_allowed = allowed_keys(memory_pad_mask)
        x = self.positional(self.embedding(tokens))
        new_cache = []
        for layer, past_outputs in zip(self.layers, cache):
            x = torch.cat([past_outputs, layer.step(x, memory, memory_allowed)], dim=1)
            new_cache.append(x)
        return self.output(self.norm(x[:, -1])), new_cache


class RobustASR(nn.Module):
    def __init__(self, cfg, tok):
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        self.logmel = LogMel(cfg)
        self.spec_augment = SpecAugment(cfg)
        self.frontend = ConvFrontend(cfg)
        self.positional = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.encoder = Encoder(cfg)
        self.ctc_head = nn.Linear(cfg.d_model, tok.vocab_size)
        self.decoder = AttentionDecoder(cfg, tok.vocab_size)

    def features(self, wavs, wav_lens):
        # Hors autocast : la FFT ne prend pas en charge bfloat16.
        with torch.no_grad():
            feats, feat_lens = self.logmel(wavs, wav_lens)
            return self.spec_augment(feats, feat_lens), feat_lens

    def encode(self, feats, feat_lens):
        x, lengths = self.frontend(feats, feat_lens)
        return self.encoder(self.positional(x), make_pad_mask(lengths, x.size(1))), lengths

    def decoder_io(self, tokens, token_lens):
        """Entrée <sos> y1…yn et cible y1…yn <eos>, avec un masque de padding commun."""
        batch_size, length = tokens.shape
        inputs = torch.cat([tokens.new_full((batch_size, 1), self.tok.sos), tokens], dim=1)
        targets = torch.cat([tokens, tokens.new_zeros(batch_size, 1)], dim=1)
        targets[torch.arange(batch_size), token_lens] = self.tok.eos
        pad_mask = make_pad_mask(token_lens + 1, length + 1)
        return inputs, targets.masked_fill(pad_mask, -100), pad_mask

    def forward(self, wavs, wav_lens, tokens, token_lens):
        feats, feat_lens = self.features(wavs, wav_lens)
        decoder_inputs, decoder_targets, token_pad_mask = self.decoder_io(tokens, token_lens)

        with autocast(wavs.device.type):
            memory, memory_lens = self.encode(feats, feat_lens)
            ctc_logits = self.ctc_head(memory)
            memory_pad_mask = make_pad_mask(memory_lens, memory.size(1))
            decoder_logits = self.decoder(decoder_inputs, memory, memory_pad_mask, token_pad_mask)

        # Les deux termes sont exprimés par caractère, pour que 0.3 / 0.7 ait le sens attendu.
        ctc_log_probs = ctc_logits.float().log_softmax(-1)
        ctc_loss = F.ctc_loss(
            ctc_log_probs.transpose(0, 1),
            tokens,
            memory_lens,
            token_lens,
            blank=self.tok.blank,
            reduction="sum",
            zero_infinity=True,
        ) / token_lens.sum()
        ce_loss = F.cross_entropy(
            decoder_logits.float().transpose(1, 2),
            decoder_targets,
            ignore_index=-100,
            label_smoothing=self.cfg.label_smoothing,
        )
        return {
            "loss": self.cfg.ctc_weight * ctc_loss + (1 - self.cfg.ctc_weight) * ce_loss,
            "ctc_loss": ctc_loss,
            "ce_loss": ce_loss,
            "ctc_log_probs": ctc_log_probs,
            "memory_lens": memory_lens,
        }

