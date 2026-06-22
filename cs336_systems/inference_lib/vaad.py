import torch
import torch.nn as nn
import math
import regex
import re
import numpy as np
import pickle


class vaad(nn.Module):
    def __init__(self, vocab, merges, specials, model, cfg):
        super().__init__()
        # Tokenizer and its Vocab and merges
        with open(vocab, "rb") as file:
            self.V = pickle.load(file)
        with open(merges, "rb") as file:
            self.M = pickle.load(file)
        # Special tokens
        self.specials = specials
        # LM model
        self.LM = model
        self.LM_config = cfg

    def encode(self, text):
        """
        encode takes in the text and encodes it according to the V and M.
        """
        final_token = []
        # merge reverse dictionary
        merge_rank = {self.M[i]: i for i in range(len(self.M))}
        # Vocab reverse dictionary
        reverse_dict = {value: key for key, value in self.V.items()}

        # special token handling
        if self.specials:
            # split it based on the specials first
            specials_sorted = sorted(self.specials, key=len, reverse=True)
            special_pat = "(" + "|".join(map(re.escape, specials_sorted)) + ")"  # type: ignore
            split_text = regex.split(
                special_pat, text  # type: ignore
            )  # pyright: ignore[reportArgumentType, reportCallIssue]
        else:
            split_text = [text]

        for s_t in split_text:
            if self.specials and s_t in self.specials:
                # encode specials
                s_t_bytes = s_t.encode("utf-8")  # type: ignore
                final_token.append(reverse_dict[s_t_bytes])
            elif s_t == "":
                continue
            else:
                PAT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
                pre_token = regex.finditer(PAT, s_t)  # type: ignore
                # read merge list and create a reverse dict
                for matches in pre_token:
                    # ex : "cat is here" is the string
                    # temp str is in bytes.[b'c',b'a',b't']
                    # have to find the non overlapping combos and merge based on priority
                    bs = matches.group(0).encode("utf-8")
                    temp_str = [bytes([b]) for b in bs]
                    # combos are [(c,a),(a,t)]
                    combos = list(zip(temp_str[:-1], temp_str[1:]))
                    # find in merge_rank to find out if it is present
                    combo_rank = []
                    for c in range(len(combos)):
                        temp_rank = merge_rank.get(combos[c])
                        combo_rank.append(
                            (temp_rank, c) if temp_rank is not None else (math.inf, c)
                        )
                    while True:
                        if any(c[0] < math.inf for c in combo_rank):
                            # atleast one merge is present. picks lexi. smalled tuple
                            min_find = min(combo_rank)
                            # merge the min pair
                            temp_str[min_find[1]] = (
                                temp_str[min_find[1]] + temp_str[min_find[1] + 1]
                            )
                            # modify combos to get token
                            temp_str[min_find[1] + 1 :] = temp_str[min_find[1] + 2 :]
                            # bump up the old rank so that its irrelevant.
                            combo_rank.pop(min_find[1])
                            # reduce the idx on right to end of combo_rank by 1
                            for idx in range(min_find[1], len(combo_rank)):
                                offset_idx = idx
                                combo_rank[offset_idx] = (
                                    combo_rank[offset_idx][0],
                                    combo_rank[offset_idx][1] - 1,
                                )
                            if min_find[1] - 1 >= 0:
                                # recompute the new rank for lower idx
                                lower_rank = merge_rank.get(  # type: ignore
                                    (
                                        temp_str[min_find[1] - 1],
                                        temp_str[min_find[1]],
                                    ),
                                    math.inf,
                                )  # type: ignore
                                combo_rank[min_find[1] - 1] = (
                                    lower_rank,
                                    min_find[1] - 1,
                                )
                            if min_find[1] < len(combo_rank):
                                # recompute the new rank for higher idx
                                higher_rank = merge_rank.get(  # type: ignore
                                    (
                                        temp_str[min_find[1]],
                                        temp_str[min_find[1] + 1],
                                    ),
                                    math.inf,
                                )
                                combo_rank[min_find[1]] = (
                                    higher_rank,
                                    min_find[1],
                                )
                        else:
                            for v in temp_str:
                                final_token.append(reverse_dict[v])
                            break
            # final token contains encoded token
            as_array = np.asarray(final_token, dtype=np.uint16)
            return as_array

    def softmax_with_temp(self, x, dim, temp):
        """
        softmax = e^(z_i/T)/Sum_j (e^(z_j/T))
        """
        x_d_norm = x - torch.max(x, dim=dim, keepdim=True).values
        x_d_sum = torch.sum(torch.exp(x_d_norm / temp), dim=dim, keepdim=True)
        y = torch.exp(x_d_norm / temp) / x_d_sum
        return y

    def get_token(self, input):
        """
        Runs one forward pass — prefill on the first call (full prompt),
        decode on every call after (single new token), since the KV cache
        cursor (self.KV.prefill) determines which path mha_self_attention takes.
        """
        rope_theta = self.LM_config["training"]["params"]["theta"]
        seq_len = input.shape[-1]
        token_positions = torch.arange(0, seq_len, device=input.device)

        with torch.inference_mode():
            y = self.LM.tranform_lm_model(
                input,
                rope_theta,
                token_positions,
            )

        # Flip the prefill flag once the first (prefill) call has completed
        for j in range(self.LM.num_layers):
            if self.LM.Tr[j].KV is not None:
                self.LM.Tr[j].KV.prefill = False

        return y

    def decode(self, ids) -> str:
        """
        Given a list of token ids, returns the decoded string.
        """
        tokens = b""
        for i in ids:
            tokens = tokens + self.V[i]
        return tokens.decode("utf-8", errors="replace")
