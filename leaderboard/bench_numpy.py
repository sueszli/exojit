# /// script
# requires-python = "==3.14.*"
# dependencies = ["numpy"]
# ///

import random
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000

ATTN_MASK = np.triu(np.full((BLOCK_SIZE, BLOCK_SIZE), -1e10), 1)
ARANGE_BS = np.arange(BLOCK_SIZE, dtype=np.int32)
INV_N_EMBED = 1.0 / N_EMBED


def rmsnorm_fwd(x: np.ndarray):
    rms = (np.mean(x * x, axis=-1, keepdims=True) + 1e-5) ** -0.5
    return x * rms, rms


def rmsnorm_bwd(dout: np.ndarray, x: np.ndarray, rms: np.ndarray) -> np.ndarray:
    dot = (dout * x).sum(axis=-1, keepdims=True)
    return dout * rms - (INV_N_EMBED * rms ** 3) * x * dot


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    e /= e.sum(axis=axis, keepdims=True)
    return e


def forward_backward(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], input_ids: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray):
    grads["wte"][:] = 0
    grads["wpe"][:] = 0

    emb = params["wte"][input_ids] + params["wpe"][ARANGE_BS]
    x, rms_init = rmsnorm_fwd(emb)

    layer_cache = []
    for li in range(N_LAYER):
        wq = params[f"layer{li}.attn_wq"]
        wk = params[f"layer{li}.attn_wk"]
        wv = params[f"layer{li}.attn_wv"]
        wo = params[f"layer{li}.attn_wo"]
        fc1 = params[f"layer{li}.mlp_fc1"]
        fc2 = params[f"layer{li}.mlp_fc2"]

        x_pre_attn = x
        xn_attn, rms_attn = rmsnorm_fwd(x)

        q = (xn_attn @ wq.T).reshape(16, N_HEAD, 4).transpose(1, 0, 2)
        k = (xn_attn @ wk.T).reshape(16, N_HEAD, 4).transpose(1, 0, 2)
        v = (xn_attn @ wv.T).reshape(16, N_HEAD, 4).transpose(1, 0, 2)
        attn_w = softmax(q @ k.transpose(0, 2, 1) * 0.5 + ATTN_MASK)
        attn_out = attn_w @ v
        attn_out_flat = attn_out.transpose(1, 0, 2).reshape(16, N_EMBED)
        x = attn_out_flat @ wo.T + x_pre_attn

        x_pre_mlp = x
        xn_mlp, rms_mlp = rmsnorm_fwd(x)

        h_pre = xn_mlp @ fc1.T
        h = np.maximum(0.0, h_pre)
        x = h @ fc2.T + x_pre_mlp

        layer_cache.append({"x_pre_attn": x_pre_attn, "xn_attn": xn_attn, "rms_attn": rms_attn, "q": q, "k": k, "v": v, "attn_w": attn_w, "attn_out_flat": attn_out_flat, "x_pre_mlp": x_pre_mlp, "xn_mlp": xn_mlp, "rms_mlp": rms_mlp, "h_pre": h_pre, "h": h})

    logits = x @ params["lm_head"].T
    probs = softmax(logits)

    sum_mask = loss_mask.sum()
    loss = -(np.log(probs[ARANGE_BS, target_ids]) * loss_mask).sum() / sum_mask

    dlogits = probs / sum_mask
    dlogits[ARANGE_BS, target_ids] -= 1.0 / sum_mask
    dlogits *= loss_mask[:, None]

    np.matmul(dlogits.T, x, out=grads["lm_head"])
    dx = dlogits @ params["lm_head"]

    for li in reversed(range(N_LAYER)):
        cache = layer_cache[li]
        wq = params[f"layer{li}.attn_wq"]
        wk = params[f"layer{li}.attn_wk"]
        wv = params[f"layer{li}.attn_wv"]
        wo = params[f"layer{li}.attn_wo"]
        fc1 = params[f"layer{li}.mlp_fc1"]
        fc2 = params[f"layer{li}.mlp_fc2"]

        dx_res_mlp = dx
        np.matmul(dx.T, cache["h"], out=grads[f"layer{li}.mlp_fc2"])
        dh = dx @ fc2
        dh_pre = dh * (cache["h_pre"] > 0)
        np.matmul(dh_pre.T, cache["xn_mlp"], out=grads[f"layer{li}.mlp_fc1"])
        dxn_mlp = dh_pre @ fc1
        dx = rmsnorm_bwd(dxn_mlp, cache["x_pre_mlp"], cache["rms_mlp"]) + dx_res_mlp

        dx_res_attn = dx
        np.matmul(dx.T, cache["attn_out_flat"], out=grads[f"layer{li}.attn_wo"])
        dattn_out_flat = dx @ wo

        dattn_out = dattn_out_flat.reshape(16, N_HEAD, 4).transpose(1, 0, 2)

        aw = cache["attn_w"]
        dv = aw.transpose(0, 2, 1) @ dattn_out
        dattn_w = dattn_out @ cache["v"].transpose(0, 2, 1)

        dlogits_attn = aw * (dattn_w - (dattn_w * aw).sum(-1, keepdims=True)) * 0.5

        dq = dlogits_attn @ cache["k"]
        dk = dlogits_attn.transpose(0, 2, 1) @ cache["q"]

        dq_flat = dq.transpose(1, 0, 2).reshape(16, N_EMBED)
        dk_flat = dk.transpose(1, 0, 2).reshape(16, N_EMBED)
        dv_flat = dv.transpose(1, 0, 2).reshape(16, N_EMBED)

        np.matmul(dq_flat.T, cache["xn_attn"], out=grads[f"layer{li}.attn_wq"])
        np.matmul(dk_flat.T, cache["xn_attn"], out=grads[f"layer{li}.attn_wk"])
        np.matmul(dv_flat.T, cache["xn_attn"], out=grads[f"layer{li}.attn_wv"])
        dxn_attn = dq_flat @ wq + dk_flat @ wk + dv_flat @ wv

        dx = rmsnorm_bwd(dxn_attn, cache["x_pre_attn"], cache["rms_attn"]) + dx_res_attn

    demb = rmsnorm_bwd(dx, emb, rms_init)
    np.add.at(grads["wte"], input_ids, demb)
    grads["wpe"] += demb

    return loss, grads


def step_fn(params: dict[str, np.ndarray], opt_state: dict, grads: dict[str, np.ndarray], input_ids: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray, step: int) -> tuple[float, dict[str, np.ndarray], dict]:
    loss, grads = forward_backward(params, grads, input_ids, target_ids, loss_mask)

    lr_t = 0.01 * (1 - step / NUM_STEPS)
    bc1 = 1.0 - 0.85 ** (step + 1)
    bc2 = 1.0 - 0.99 ** (step + 1)

    flat_m = opt_state["flat_m"]
    flat_v = opt_state["flat_v"]
    flat_params = opt_state["flat_params"]
    buf = opt_state["buf"]
    tmp = opt_state["tmp"]

    offset = 0
    for k in PARAM_KEYS:
        g = grads[k]
        s = g.size
        buf[offset : offset + s] = g.ravel()
        offset += s

    np.multiply(buf, 0.15, out=tmp)
    flat_m *= 0.85
    flat_m += tmp
    np.multiply(buf, buf, out=tmp)
    tmp *= 0.01
    flat_v *= 0.99
    flat_v += tmp

    np.divide(flat_m, bc1, out=tmp)
    np.divide(flat_v, bc2, out=buf)
    np.sqrt(buf, out=buf)
    buf += 1e-8
    np.divide(tmp, buf, out=tmp)
    tmp *= lr_t
    flat_params -= tmp

    return loss, params, opt_state


def tokenize(doc: str, uchars: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = len(uchars)
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)

    input_ids = np.zeros(BLOCK_SIZE, dtype=np.int32)
    target_ids = np.zeros(BLOCK_SIZE, dtype=np.int32)
    loss_mask = np.zeros(BLOCK_SIZE, dtype=np.float32)

    input_ids[:n] = tokens[:n]
    target_ids[:n] = tokens[1 : n + 1]
    loss_mask[:n] = 1.0

    return input_ids, target_ids, loss_mask


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

matrix = lambda nout, nin, std=0.08: np.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], dtype=np.float64)
state_dict = {
    "wte": matrix(len(uchars) + 1, N_EMBED),
    "wpe": matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": matrix(len(uchars) + 1, N_EMBED),
    **{f"layer{i}.attn_wq": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wk": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wv": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wo": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc1": matrix(4 * N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc2": matrix(N_EMBED, 4 * N_EMBED) for i in range(N_LAYER)},
}

PARAM_KEYS = list(state_dict.keys())
flat_params = np.concatenate([state_dict[k].ravel() for k in PARAM_KEYS])
TOTAL_PARAMS = flat_params.size
offset = 0
for k in PARAM_KEYS:
    n = state_dict[k].size
    state_dict[k] = flat_params[offset : offset + n].reshape(state_dict[k].shape)
    offset += n

opt_state = {
    "flat_m": np.zeros(TOTAL_PARAMS),
    "flat_v": np.zeros(TOTAL_PARAMS),
    "flat_params": flat_params,
    "buf": np.empty(TOTAL_PARAMS),
    "tmp": np.empty(TOTAL_PARAMS),
}

tokenized = [tokenize(doc, uchars) for doc in docs]

grads = {k: np.zeros_like(state_dict[k]) for k in PARAM_KEYS}

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, grads, input_ids, target_ids, loss_mask, step)
    step_times.append(time.perf_counter() - t0)

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
