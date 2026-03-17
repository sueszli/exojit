# /// script
# requires-python = "==3.14.*"
# dependencies = ["torch", "tqdm"]
# ///

import functools
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()


@torch.compile(dynamic=True)
def forward(params: dict[str, torch.Tensor], input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    n = input_ids.shape[0]
    x = rmsnorm(params["wte"][input_ids] + params["wpe"][torch.arange(n)])
    for li in range(N_LAYER):
        x_residual = x
        xn = rmsnorm(x)
        q = (xn @ params[f"layer{li}.attn_wq"].T).view(n, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        k = (xn @ params[f"layer{li}.attn_wk"].T).view(n, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        v = (xn @ params[f"layer{li}.attn_wv"].T).view(n, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        mask = torch.triu(torch.full((n, n), float("-inf"), dtype=x.dtype), 1)
        attn_weights = F.softmax(q @ k.transpose(-2, -1) / (N_EMBED // N_HEAD) ** 0.5 + mask, dim=-1)
        x = (attn_weights @ v).transpose(0, 1).reshape(n, N_EMBED) @ params[f"layer{li}.attn_wo"].T + x_residual
        x_residual = x
        xn = rmsnorm(x)
        x = F.relu(xn @ params[f"layer{li}.mlp_fc1"].T) @ params[f"layer{li}.mlp_fc2"].T + x_residual
    return F.cross_entropy(x @ params["lm_head"].T, target_ids)


def step_fn(params: dict[str, torch.Tensor], opt_state: dict[str, dict[str, torch.Tensor]], input_ids: torch.Tensor, target_ids: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    loss = forward(params, input_ids, target_ids)
    loss.backward()

    learning_rate = 0.01
    beta1 = 0.85
    beta2 = 0.99
    eps_adam = 1e-8
    lr_t = learning_rate * (1 - step / NUM_STEPS)

    m = opt_state["m"]
    v = opt_state["v"]

    with torch.no_grad():
        for k, p in params.items():
            m[k] = beta1 * m[k] + (1 - beta1) * p.grad
            v[k] = beta2 * v[k] + (1 - beta2) * p.grad**2
            m_hat = m[k] / (1 - beta1 ** (step + 1))
            v_hat = v[k] / (1 - beta2 ** (step + 1))
            p -= lr_t * m_hat / (v_hat.sqrt() + eps_adam)
            p.grad.zero_()

    return loss, params, {"m": m, "v": v}


@functools.cache
def char_to_id(uchars_tuple: tuple[str, ...]) -> dict[str, int]:
    return {ch: i for i, ch in enumerate(uchars_tuple)}


def tokenize(doc: str, uchars: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    c2i = char_to_id(tuple(uchars))
    bos = len(uchars)
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    return torch.tensor(tokens[:n]), torch.tensor(tokens[1 : n + 1])


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

matrix = lambda nout, nin, std=0.08: torch.tensor([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], dtype=torch.float64).requires_grad_(True)
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

opt_state = {"m": {k: torch.zeros_like(p) for k, p in state_dict.items()}, "v": {k: torch.zeros_like(p) for k, p in state_dict.items()}}

tokenized = [tokenize(doc, uchars) for doc in tqdm(docs, desc="tokenizing")]

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, input_ids, target_ids, step)
    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {NUM_STEPS:4d} | loss {float(loss):.4f}", end="\r")

save_times(step_times)
assert_weights_match(state_dict)
