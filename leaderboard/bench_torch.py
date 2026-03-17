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


def forward(params: dict[str, torch.Tensor], input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    n = input_ids.shape[0]
    x = rmsnorm(params["wte"][input_ids] + params["wpe"][:n])
    for li in range(N_LAYER):
        x_residual = x
        xn = rmsnorm(x)
        q = F.linear(xn, params[f"layer{li}.attn_wq"]).view(n, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        k = F.linear(xn, params[f"layer{li}.attn_wk"]).view(n, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        v = F.linear(xn, params[f"layer{li}.attn_wv"]).view(n, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.linear(attn_out.transpose(0, 1).reshape(n, N_EMBED), params[f"layer{li}.attn_wo"]) + x_residual
        x_residual = x
        xn = rmsnorm(x)
        x = F.linear(F.relu(F.linear(xn, params[f"layer{li}.mlp_fc1"])), params[f"layer{li}.mlp_fc2"]) + x_residual
    return F.cross_entropy(F.linear(x, params["lm_head"]), target_ids)


def step_fn(params: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer, input_ids: torch.Tensor, target_ids: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.optim.Optimizer]:
    optimizer.zero_grad()
    loss = forward(params, input_ids, target_ids)
    loss.backward()

    learning_rate = 0.01
    lr_t = learning_rate * (1 - step / NUM_STEPS)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_t

    optimizer.step()

    return loss, params, optimizer


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

opt_state = torch.optim.Adam(list(state_dict.values()), lr=0.01, betas=(0.85, 0.99), eps=1e-8)

tokenized = [tokenize(doc, uchars) for doc in tqdm(docs, desc="tokenizing")]

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, input_ids, target_ids, step)
    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {NUM_STEPS:4d} | loss {loss.item():.4f}", end="\r")

save_times(step_times)
assert_weights_match(state_dict)
