import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tomllib
import pickle
import regex
import math
import re
import numpy as np

from cs336_basics.create_obj import create_obj


def load_model(cfg_path, checkpoint_path):
    """
    Load model from config and checkpoint. Call once and pass LM to generate_text.
    """
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    LM = create_obj(
        cfg, field="model", extra_kwargs={"theta": cfg["training"]["params"]["theta"]}
    )
    device = LM.device
    LM = LM.to(device)

    LM_state = torch.load(checkpoint_path, map_location=device)
    state_dict = LM_state.get("model_state", LM_state)
    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned_state[k] = v
    LM.load_state_dict(cleaned_state, strict=True)
    LM.eval()

    return LM, cfg


def load_vocab(vocab_path, merges_path):
    """
    Load vocab and merges. Call once and pass to generate_text.
    """
    with open(vocab_path, "rb") as f:
        V = pickle.load(f)
    with open(merges_path, "rb") as f:
        M = pickle.load(f)
    return V, M


def generate_text(x, len_x, temp, top_p, LM, cfg, V, M):
    """
    generate_text uses the trained model and generates text.

    Args :
        x    - Input prompt (string)
        len_x - Maximum length of output expected (int)
        temp  - softmax temperature
        top_p - nucleus sampling threshold
        LM    - pre-loaded model (from load_model)
        cfg   - config dict (from load_model)
        V     - vocab dict (from load_vocab)
        M     - merges list (from load_vocab)

    Return :
        x - Completion to x (string)
    """
    device = LM.device

    # Find EOT token id
    eot_id = None
    for v in range(len(V)):
        if V[v] == b"<|endoftext|>":
            eot_id = v
            break

    special_tokens = ("<|endoftext|>",)

    if temp <= 0:
        raise ValueError("Temperature must be > 0 for sampling.")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1].")

    x_token = encode(x, specials=special_tokens, V=V, merge_lst=M)
    x_token = torch.from_numpy(x_token).unsqueeze(0).to(device=device, dtype=torch.long)
    new_tokens = []
    max_seq_len = cfg["model"]["params"]["context_length"]
    rope_theta = cfg["training"]["params"]["theta"]

    with torch.inference_mode():
        while len(new_tokens) < len_x:
            model_input = x_token[:, -max_seq_len:]
            token_positions = torch.arange(
                model_input.size(1), device=device, dtype=torch.long
            )

            y = LM.tranform_lm_model(
                model_input,
                rope_theta=rope_theta,
                token_positions=token_positions,
                max_seq_len=max_seq_len,
            )

            p = softmax_with_temp(y[:, -1, :], -1, temp).squeeze(0)

            sorted_p, sort_idx = p.sort(descending=True)
            cum_p = torch.cumsum(sorted_p, dim=0)
            cutoff_idx = torch.where(cum_p >= top_p)[0]
            top_count = cutoff_idx[0].item() + 1 if len(cutoff_idx) else len(sorted_p)
            top_p_elem = sort_idx[:top_count]
            top_p_prob = sorted_p[:top_count]

            top_p_mod = top_p_prob / top_p_prob.sum()
            chosen_index = torch.multinomial(top_p_mod, num_samples=1)
            token_chosen = top_p_elem[chosen_index]

            next_token = V[token_chosen.item()]
            new_tokens.append(next_token)
            token_to_str = next_token.decode('utf-8', errors='replace')
            x = x + token_to_str

            token_tensor = token_chosen.view(1, 1).to(device)
            x_token = torch.cat((x_token, token_tensor), dim=1)

            if eot_id is not None and token_chosen.item() == eot_id:
                break

    return x


def softmax_with_temp(x, dim, temp):
    x_d_norm = x - torch.max(x, dim=dim, keepdim=True).values
    x_d_sum = torch.sum(torch.exp(x_d_norm / temp), dim=dim, keepdim=True)
    y = torch.exp(x_d_norm / temp) / x_d_sum
    return y


def encode(text, specials, V, merge_lst):
    final_token = []
    merge_rank = {merge_lst[i]: i for i in range(len(merge_lst))}

    if specials:
        specials_sorted = sorted(specials, key=len, reverse=True)
        special_pat = "(" + "|".join(map(re.escape, specials_sorted)) + ")"
        split_text = regex.split(
            special_pat, text  # type: ignore
        )  # pyright: ignore[reportArgumentType, reportCallIssue]
    else:
        split_text = [text]

    reverse_dict = {value: key for key, value in V.items()}
    for s_t in split_text:
        if specials and s_t in specials:
            s_t_bytes = s_t.encode("utf-8")  # type: ignore
            final_token.append(reverse_dict[s_t_bytes])
        elif s_t == "":
            continue
        else:
            PAT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            pre_token = regex.finditer(PAT, s_t)  # type: ignore
            for matches in pre_token:
                bs = matches.group(0).encode("utf-8")
                temp_str = [bytes([b]) for b in bs]
                combos = list(zip(temp_str[:-1], temp_str[1:]))
                combo_rank = []
                for c in range(len(combos)):
                    temp_rank = merge_rank.get(combos[c])
                    combo_rank.append(
                        (temp_rank, c) if temp_rank is not None else (math.inf, c)
                    )
                while True:
                    if any(c[0] < math.inf for c in combo_rank):
                        min_find = min(combo_rank)
                        temp_str[min_find[1]] = (
                            temp_str[min_find[1]] + temp_str[min_find[1] + 1]
                        )
                        temp_str[min_find[1] + 1:] = temp_str[min_find[1] + 2:]
                        combo_rank.pop(min_find[1])
                        for idx in range(min_find[1], len(combo_rank)):
                            combo_rank[idx] = (combo_rank[idx][0], combo_rank[idx][1] - 1)
                        if min_find[1] - 1 >= 0:
                            lower_rank = merge_rank.get(
                                (temp_str[min_find[1] - 1], temp_str[min_find[1]]),
                                math.inf,
                            )
                            combo_rank[min_find[1] - 1] = (lower_rank, min_find[1] - 1)
                        if min_find[1] < len(combo_rank):
                            higher_rank = merge_rank.get(
                                (temp_str[min_find[1]], temp_str[min_find[1] + 1]),
                                math.inf,
                            )
                            combo_rank[min_find[1]] = (higher_rank, min_find[1])
                    else:
                        for v in temp_str:
                            final_token.append(reverse_dict[v])
                        break
    as_array = np.asarray(final_token, dtype=np.uint16)
    return as_array


if __name__ == "__main__":

    LM, cfg = load_model(
        cfg_path="configs/small.toml",
        checkpoint_path="world_1_small_15k_owt/checkpoint/iter_15000.pt",
    )
    if torch.backends.mps.is_available():
        LM = LM.to("mps")
        LM.device = torch.device("mps")
    V, M = load_vocab(
        vocab_path="owt_bpe/owt_vocab.pkl",
        merges_path="owt_bpe/owt_merges.pkl",
    )

    prompt = "Once there lived a dog called bruno"
    completion = generate_text(prompt, len_x=256, temp=0.4, top_p=0.75, LM=LM, cfg=cfg, V=V, M=M)
    print(completion)
