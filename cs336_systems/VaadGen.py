import pickle
import torch
import tomllib

# user defined imports
from cs336_systems.inference_lib.vaad import vaad
from cs336_systems.inference_lib.KVCache import KVCache
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.adamW import AdamW
from cs336_systems.utilities.create_obj import create_obj
from cs336_basics.checkpoint_utils import load_checkpoint


def load_vaad(config):
    """
    One-time setup: load config, build the model, load the checkpoint, build
    the vaad object. Call once (e.g. at server startup) and reuse v_obj/cfg
    for every subsequent generation call.
    """
    with open(config, "rb") as f:
        cfg = tomllib.load(f)

    inf_device = cfg["inference"]["device"]
    model_path = cfg["inference"]["path"]

    # create the LM object (override the training-config device with the
    # inference device — configs/small.toml's [model.params].device is "cuda",
    # set for the Lambda training run, not for local inference)
    LM = create_obj(
        cfg=cfg,
        field="model",
        extra_kwargs={
            "theta": cfg["training"]["params"]["theta"],
            "kv_cache": True,
            "device": inf_device,
        },
    )

    # load the model
    load_checkpoint(src=model_path, model=LM, device=inf_device)

    # generate vaad object
    v_obj = vaad(
        vocab=cfg["inference"]["vocab"],
        merges=cfg["inference"]["merges"],
        specials=("<|endoftext|>",),
        model=LM,
        cfg=cfg,
    )

    return v_obj, cfg


def vaad_generate(v_obj, cfg, txt_ip):
    """
    Per-request generation. Resets the KV cache for every layer first, so each
    call starts a fresh sequence instead of continuing the previous one.
    """
    inf_device = cfg["inference"]["device"]
    max_seq_len = cfg["model"]["params"]["context_length"]
    top_p = cfg["inference"]["top_p"]
    temperature = cfg["inference"]["temp"]

    # start a fresh sequence: rewind every layer's cache cursor
    for layer in v_obj.LM.Tr:
        if layer.KV is not None:
            layer.KV.reset()

    # encode
    encoded_txt = v_obj.encode(text=txt_ip)
    encoded_txt = (
        torch.from_numpy(encoded_txt)
        .unsqueeze(0)
        .to(device=inf_device, dtype=torch.long)
    )

    # Generation loop
    tok_gen_cnt = 0
    response = ""
    while tok_gen_cnt <= max_seq_len:

        if tok_gen_cnt != 0:
            gen_ip = t
        else:
            # this is prefill ip
            gen_ip = encoded_txt

        # Run the vaad forward
        new_tok = v_obj.get_token(input=gen_ip)
        tok_gen_cnt += 1

        # always extract the last token
        last_tok = new_tok[:, -1, :]

        # softmax with Temp
        p = v_obj.softmax_with_temp(x=last_tok, dim=-1, temp=temperature).squeeze(0)
        # top p sample
        sorted_p, sort_idx = p.sort(descending=True)
        cum_p = torch.cumsum(sorted_p, dim=0)
        cutoff_idx = torch.where(cum_p >= top_p)[0]
        top_count = cutoff_idx[0].item() + 1 if len(cutoff_idx) else len(sorted_p)
        top_p_elem = sort_idx[:top_count]
        top_p_prob = sorted_p[:top_count]

        top_p_mod = top_p_prob / top_p_prob.sum()
        chosen_index = torch.multinomial(top_p_mod, num_samples=1)
        token_chosen = top_p_elem[chosen_index]

        # get the response
        response = response + v_obj.decode(token_chosen.tolist())
        t = token_chosen.view(1, 1).to(device=inf_device, dtype=torch.long)

    return response
