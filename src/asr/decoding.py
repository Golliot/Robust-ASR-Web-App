"""Beam search joint CTC / attention et décodage CTC glouton du notebook (cellule 21).

Chaque hypothèse g est prolongée par un token c selon le score
    score(gc) = score(g) + (1 - λ) log p_att(c | g) + λ [ψ(gc) - ψ(g)]
où ψ est le score de préfixe CTC (Watanabe et al., 2017). Sa récurrence sur les trames devient une
somme cumulée (torch.logcumsumexp) : la recherche est vectorisée sur tout un batch. λ doit rester
strictement positif : sans terme CTC, le beam search favorise les hypothèses tronquées. La
dérivation complète est dans la section 9 du notebook.

Deux retouches par rapport au notebook : `autocast` reçoit le type de périphérique des formes
d'onde, et `ctc_greedy` reçoit l'indice du blank au lieu de lire le tokenizer global.
"""

import math

import torch

from src.asr.features import make_pad_mask
from src.asr.model import autocast


def cumulative_sum(x):
    """Somme cumulée sur le temps, avec un 0 initial : (…, T) -> (…, T+1)."""
    return torch.cat([torch.zeros_like(x[..., :1]), x.cumsum(-1)], dim=-1)


def ctc_advance(phi, token_log_probs, blank_cumulative):
    """Calcule r^n et r^b du préfixe prolongé, sous forme fermée : (N, T+1) chacun."""
    token_cumulative = cumulative_sum(token_log_probs)
    log_zero = torch.full_like(blank_cumulative[:, :1], -math.inf)

    r_token = token_cumulative[:, 1:] + torch.logcumsumexp(phi - token_cumulative[:, :-1], dim=1)
    r_token = torch.cat([log_zero, r_token], dim=1)
    r_blank = blank_cumulative[:, 1:] + torch.logcumsumexp(r_token[:, :-1] - blank_cumulative[:, :-1], dim=1)
    return r_token, torch.cat([log_zero, r_blank], dim=1)


@torch.no_grad()
def joint_beam_search(model, wavs, wav_lens, beam_size=None, ctc_weight=None):
    """Décode un batch d'énoncés et renvoie, pour chacun, la liste des tokens de la meilleure hypothèse."""
    tok = model.tok
    beam_size = beam_size or model.cfg.beam_size
    ctc_weight = model.cfg.ctc_weight if ctc_weight is None else ctc_weight
    assert ctc_weight > 0, "Sans terme CTC, le beam search favorise les hypothèses tronquées (voir ci-dessus)"

    feats, feat_lens = model.features(wavs, wav_lens)
    device_type = wavs.device.type
    with autocast(device_type):
        memory, memory_lens = model.encode(feats, feat_lens)
        ctc_log_probs = model.ctc_head(memory).float().log_softmax(-1).double()

    # Une ligne par hypothèse : N = B × K. Les K lignes d'un énoncé se suivent.
    n_utterances, n_frames, _ = memory.shape
    n_hypotheses = n_utterances * beam_size
    pad_mask = make_pad_mask(memory_lens, n_frames).repeat_interleave(beam_size, 0)
    memory = memory.repeat_interleave(beam_size, 0)
    lengths = memory_lens.repeat_interleave(beam_size)
    ctc_log_probs = ctc_log_probs.repeat_interleave(beam_size, 0).masked_fill(pad_mask[:, :, None], 0.0)

    # Tokens candidats : tout le vocabulaire sauf le blank. Le dernier est <eos>.
    candidates = torch.arange(1, tok.vocab_size, device=memory.device)
    n_candidates = candidates.size(0)
    eos_slot = n_candidates - 1
    candidate_log_probs = ctc_log_probs[:, :, 1:].transpose(1, 2)  # (N, P, T)
    blank_cumulative = cumulative_sum(ctc_log_probs[:, :, tok.blank])  # (N, T+1)

    tokens = torch.full((n_hypotheses, 1), tok.sos, device=memory.device)
    scores = torch.full((n_hypotheses,), -math.inf, dtype=torch.float64, device=memory.device)
    scores[::beam_size] = 0.0  # au départ, seule la première hypothèse de chaque énoncé existe
    finished = torch.zeros(n_hypotheses, dtype=torch.bool, device=memory.device)
    psi = torch.zeros_like(scores)
    r_token = torch.full_like(blank_cumulative, -math.inf)
    r_blank = blank_cumulative.clone()
    cache = model.decoder.initial_cache(memory)
    utterance_offsets = torch.arange(n_utterances, device=memory.device)[:, None] * beam_size

    for _ in range(n_frames + 1):
        with autocast(device_type):
            logits, cache = model.decoder.step(tokens, memory, pad_mask, cache)
        attention_scores = logits.float().log_softmax(-1)[:, 1:].double()  # (N, P)

        # Score de préfixe CTC de chaque candidat. Répéter le dernier token exige un blank entre les deux.
        r_total = torch.logaddexp(r_token, r_blank)
        repeats_last = candidates[None] == tokens[:, -1:]  # (N, P)
        phi = torch.where(repeats_last[:, :, None], r_blank[:, None, :-1], r_total[:, None, :-1])
        new_psi = (phi + candidate_log_probs).masked_fill(pad_mask[:, None], -math.inf).logsumexp(-1)
        new_psi[:, eos_slot] = r_total.gather(1, lengths[:, None])[:, 0]  # préfixe lu en entier

        joint = scores[:, None] + (1 - ctc_weight) * attention_scores + ctc_weight * (new_psi - psi[:, None])
        joint = joint.masked_fill(joint.isnan(), -math.inf)  # -inf - (-inf) : préfixe impossible
        joint[finished] = -math.inf
        joint[finished, eos_slot] = scores[finished]  # une hypothèse terminée garde son score

        # Les K meilleurs prolongements de chaque énoncé, triés par score décroissant.
        top_scores, top_indices = joint.view(n_utterances, beam_size * n_candidates).topk(beam_size, dim=-1)
        parents = (top_indices // n_candidates + utterance_offsets).flatten()
        slots = (top_indices % n_candidates).flatten()
        new_tokens = candidates[slots]

        # Seuls les survivants ont besoin de leur état CTC complet.
        chosen_phi = torch.where(repeats_last[parents, slots, None], r_blank[parents, :-1], r_total[parents, :-1])
        r_token, r_blank = ctc_advance(chosen_phi, candidate_log_probs[parents, slots], blank_cumulative)

        psi = new_psi[parents, slots]
        scores = top_scores.flatten()
        finished = finished[parents] | (new_tokens == tok.eos)
        tokens = torch.cat([tokens[parents], new_tokens[:, None]], dim=1)
        cache = [outputs[parents] for outputs in cache]

        if finished[::beam_size].all():
            break

    hypotheses = []
    for sequence in tokens[::beam_size, 1:].tolist():
        hypotheses.append(sequence[: sequence.index(tok.eos)] if tok.eos in sequence else sequence)
    return hypotheses


def ctc_greedy(ctc_log_probs, memory_lens, blank=0):
    """Décodage CTC rapide : meilleur token par trame, répétitions fusionnées, blanks retirés."""
    best = ctc_log_probs.argmax(-1)
    keep = torch.ones_like(best, dtype=torch.bool)
    keep[:, 1:] = best[:, 1:] != best[:, :-1]
    keep &= best != blank
    return [best[i, : length][keep[i, : length]].tolist() for i, length in enumerate(memory_lens.tolist())]
