# /// script
# requires-python = "==3.14.*"
# dependencies = ["tqdm"]
# ///

import functools
import math
import random
import time
from pathlib import Path

from tqdm import tqdm
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000


class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        out = self.data**other
        return Value(out, (self,), (other * out / self.data,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        out = math.exp(self.data)
        return Value(out, (self,), (out,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        completed = set()
        stack = [self]
        while stack:
            v = stack[-1]
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    if child not in visited:
                        stack.append(child)
            else:
                stack.pop()
                if v not in completed:
                    completed.add(v)
                    topo.append(v)
        self.grad = 1
        for v in reversed(topo):
            vgrad = v.grad
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * vgrad


def linear(x, w):
    result = []
    for wo in w:
        data = sum(wi.data * xi.data for wi, xi in zip(wo, x))
        children = (*wo, *x)
        local_grads = (*(xi.data for xi in x), *(wi.data for wi in wo))
        result.append(Value(data, children, local_grads))
    return result


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def forward(params: dict[str, list[list[Value]]], input_ids: list[int], target_ids: list[int]) -> Value:
    n = len(input_ids)
    keys = [[] for _ in range(N_LAYER)]
    values = [[] for _ in range(N_LAYER)]
    losses = []

    for pos_id in range(n):
        token_id = input_ids[pos_id]
        target_id = target_ids[pos_id]

        tok_emb = params["wte"][token_id]
        pos_emb = params["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(N_LAYER):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, params[f"layer{li}.attn_wq"])
            k = linear(x, params[f"layer{li}.attn_wk"])
            v = linear(x, params[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(N_HEAD):
                hs = h * N_EMBED // N_HEAD
                q_h = q[hs : hs + N_EMBED // N_HEAD]
                k_h = [ki[hs : hs + N_EMBED // N_HEAD] for ki in keys[li]]
                v_h = [vi[hs : hs + N_EMBED // N_HEAD] for vi in values[li]]
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(N_EMBED // N_HEAD)) / (N_EMBED // N_HEAD) ** 0.5 for t in range(len(k_h))]
                attn_weights = softmax(attn_logits)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(N_EMBED // N_HEAD)]
                x_attn.extend(head_out)
            x = linear(x_attn, params[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = rmsnorm(x)
            x = linear(x, params[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear(x, params[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, params["lm_head"])
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    return (1 / n) * sum(losses)


def step_fn(params: dict[str, list[list[Value]]], opt_state: dict[str, list[float]], input_ids: list[int], target_ids: list[int], step: int) -> tuple[Value, dict[str, list[list[Value]]], dict[str, list[float]]]:
    loss = forward(params, input_ids, target_ids)
    loss.backward()

    learning_rate = 0.01
    beta1 = 0.85
    beta2 = 0.99
    eps_adam = 1e-8
    lr_t = learning_rate * (1 - step / NUM_STEPS)

    m = opt_state["m"]
    v = opt_state["v"]

    bc1 = 1 - beta1 ** (step + 1)
    bc2 = 1 - beta2 ** (step + 1)

    all_params = [p for mat in params.values() for row in mat for p in row]
    for i, p in enumerate(all_params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat = m[i] / bc1
        v_hat = v[i] / bc2
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0

    return loss, params, {"m": m, "v": v}


@functools.cache
def char_to_id(uchars_tuple: tuple[str, ...]) -> dict[str, int]:
    return {ch: i for i, ch in enumerate(uchars_tuple)}


def tokenize(doc: str, uchars: list[str]) -> tuple[list[int], list[int]]:
    c2i = char_to_id(tuple(uchars))
    bos = len(uchars)
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    return tokens[:n], tokens[1 : n + 1]


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
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

all_params = [p for mat in state_dict.values() for row in mat for p in row]
opt_state = {"m": [0.0] * len(all_params), "v": [0.0] * len(all_params)}

tokenized = [tokenize(doc, uchars) for doc in tqdm(docs, desc="tokenizing")]

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, input_ids, target_ids, step)
    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {NUM_STEPS:4d} | loss {loss.data:.4f}", end="\r")

save_times(step_times)
assert_weights_match(state_dict)
