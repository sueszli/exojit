#
# microgpt — tiny transformer on character-level names data.
# jit kernels for forward pass, numpy for backward/autograd.
# ref: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
#

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from kernels.add import add as jit_add  # noqa: E402
from kernels.cross_entropy import cross_entropy as jit_cross_entropy  # noqa: E402
from kernels.dot import dot as jit_dot  # noqa: E402
from kernels.embedding import embedding as jit_embedding  # noqa: E402
from kernels.matvec import matvec as jit_matvec  # noqa: E402
from kernels.relu import relu as jit_relu  # noqa: E402
from kernels.rmsnorm import rmsnorm as jit_rmsnorm  # noqa: E402
from kernels.softmax import _jit_core, _jit_max  # noqa: E402
from kernels.weighted_sum import weighted_sum as jit_weighted_sum  # noqa: E402

random.seed(42)

#
# dataset & tokenizer
#

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
if not os.path.exists(data_path):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, data_path)

docs = [line.strip() for line in open(data_path) if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

#
# tensor autograd engine
#


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev")

    def __init__(self, data, _prev=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._prev = _prev
        self._backward = _backward

    def backward(self):
        # topo sort
        topo = []
        visited = set()

        def build_topo(t):
            tid = id(t)
            if tid not in visited:
                visited.add(tid)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        # reverse-order chain rule
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()


#
# tensor ops — jit forward, numpy backward
#


def t_embedding(table, idx):
    # table(vocab, d)[idx] -> (d,)
    d = table.data.shape[1]
    out_data = np.empty(d, dtype=np.float32)
    jit_embedding(d)(out_data, np.ascontiguousarray(table.data[idx]))
    out = Tensor(out_data, _prev=(table,))

    def _backward():
        table.grad[idx] += out.grad

    out._backward = _backward
    return out


def t_add(a, b):
    # z = a + b
    n = a.data.shape[0]
    out_data = np.empty(n, dtype=np.float32)
    jit_add(n)(out_data, a.data, b.data)
    out = Tensor(out_data, _prev=(a, b))

    def _backward():
        a.grad += out.grad
        b.grad += out.grad

    out._backward = _backward
    return out


def t_rmsnorm(x):
    # y = x / sqrt(mean(x^2) + eps)
    n = x.data.shape[0]
    eps = 1e-5
    sumsq_fn, scale_fn = jit_rmsnorm(n)

    # sumsq -> scale factor -> scale kernel
    sumsq_buf = np.empty(1, dtype=np.float32)
    sumsq_fn(sumsq_buf, x.data)
    scale_val = 1.0 / math.sqrt(float(sumsq_buf[0]) / n + eps)
    scale_buf = np.array([scale_val], dtype=np.float32)
    out_data = np.empty(n, dtype=np.float32)
    scale_fn(out_data, x.data, scale_buf)
    out = Tensor(out_data, _prev=(x,))

    saved_x = x.data.copy()
    saved_ss = float(sumsq_buf[0])
    saved_scale = scale_val

    # d/dx(x * s) where s = (ss/n + eps)^-0.5
    def _backward():
        ss_eps = saved_ss + eps * n
        x.grad += saved_scale * (out.grad - saved_x * np.dot(saved_x, out.grad) / ss_eps)

    out._backward = _backward
    return out


def t_linear(x, W):
    # y = W @ x
    m, k = W.data.shape
    out_data = np.empty(m, dtype=np.float32)
    jit_matvec(m, k)(out_data, W.data, x.data)
    out = Tensor(out_data, _prev=(x, W))

    saved_x = x.data.copy()
    saved_W = W.data.copy()

    def _backward():
        x.grad += saved_W.T @ out.grad
        W.grad += np.outer(out.grad, saved_x)

    out._backward = _backward
    return out


def t_relu(x):
    # y = max(0, x)
    n = x.data.shape[0]
    out_data = np.empty(n, dtype=np.float32)
    jit_relu(n)(out_data, x.data)
    out = Tensor(out_data, _prev=(x,))

    saved_mask = x.data > 0

    def _backward():
        x.grad += out.grad * saved_mask

    out._backward = _backward
    return out


def t_cross_entropy(logits, target):
    # loss = -log(softmax(logits)[target])
    n = logits.data.shape[0]
    max_fn, sum_exp_fn = jit_cross_entropy(n)

    # max -> sum_exp -> loss
    mx = np.empty(1, dtype=np.float32)
    max_fn(mx, logits.data)
    sum_exp = np.empty(1, dtype=np.float32)
    sum_exp_fn(sum_exp, logits.data, mx)
    loss_val = float(-logits.data[target] + mx[0]) + math.log(float(sum_exp[0]))
    out = Tensor(np.array([loss_val], dtype=np.float32), _prev=(logits,))

    # softmax probs for backward (numpy exact exp)
    shifted = logits.data - mx[0]
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum()

    # d_logits = softmax - one_hot
    def _backward():
        grad = probs.copy()
        grad[target] -= 1.0
        logits.grad += grad * out.grad[0]

    out._backward = _backward
    return out


def t_attention(q, keys_cache, values_cache, n_head, head_dim):
    # multi-head attention: one query against all cached k/v
    T = len(keys_cache)
    d = head_dim
    scale = 1.0 / math.sqrt(d)

    head_outputs = []
    saved_weights_per_head = []
    dot_fn = jit_dot(d)

    for h in range(n_head):
        hs = h * d
        he = hs + d
        q_h = np.ascontiguousarray(q.data[hs:he])

        # scores[t] = dot(q_h, k_h[t]) / sqrt(d)
        scores = np.empty(T, dtype=np.float32)
        score_buf = np.empty(1, dtype=np.float32)
        for t in range(T):
            k_h_t = np.ascontiguousarray(keys_cache[t].data[hs:he])
            dot_fn(score_buf, q_h, k_h_t)
            scores[t] = score_buf[0] * scale

        # weights = softmax(scores)
        weights = np.empty(T, dtype=np.float32)
        mx = np.empty(1, dtype=np.float32)
        _jit_max(T)(mx, scores)
        _jit_core(T)(weights, scores, mx)

        # head_out[j] = sum_t(weights[t] * V[t,j])
        V_mat = np.empty((T, d), dtype=np.float32)
        for t in range(T):
            V_mat[t] = values_cache[t].data[hs:he]
        head_out = np.empty(d, dtype=np.float32)
        jit_weighted_sum(T, d)(head_out, weights, V_mat)

        head_outputs.append(head_out)
        saved_weights_per_head.append(weights.copy())

    out_data = np.concatenate(head_outputs)

    all_prev = [q]
    for t in range(T):
        all_prev.append(keys_cache[t])
        all_prev.append(values_cache[t])
    out = Tensor(out_data, _prev=tuple(all_prev))

    saved_q = q.data.copy()

    def _backward():
        dout = out.grad
        for h in range(n_head):
            hs = h * d
            he = hs + d
            d_head = dout[hs:he]
            weights = saved_weights_per_head[h]
            q_h = saved_q[hs:he]

            # d_w[t] = dot(d_head, V[t]), d_V[t] += w[t] * d_head
            d_weights = np.zeros(T, dtype=np.float32)
            for t in range(T):
                v_h_t = values_cache[t].data[hs:he]
                d_weights[t] = np.dot(d_head, v_h_t)
                values_cache[t].grad[hs:he] += weights[t] * d_head

            # d_scores = w * (d_w - dot(w, d_w))
            d_scores = weights * (d_weights - np.dot(weights, d_weights))

            # d_q += d_s * k / sqrt(d), d_k += d_s * q / sqrt(d)
            for t in range(T):
                k_h_t = keys_cache[t].data[hs:he]
                q.grad[hs:he] += d_scores[t] * scale * k_h_t
                keys_cache[t].grad[hs:he] += d_scores[t] * scale * q_h

    out._backward = _backward
    return out


def t_mean_loss(losses):
    # average scalar tensors
    n = len(losses)
    avg = sum(float(loss.data[0]) for loss in losses) / n
    out = Tensor(np.array([avg], dtype=np.float32), _prev=tuple(losses))

    def _backward():
        for loss in losses:
            loss.grad += out.grad / n

    out._backward = _backward
    return out


#
# model init
#

n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head  # 4


def make_matrix(nout, nin, std=0.08):
    data = np.array(
        [[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)],
        dtype=np.float32,
    )
    return Tensor(data)


# same init order as reference for reproducibility
state_dict = {
    "wte": make_matrix(vocab_size, n_embd),
    "wpe": make_matrix(block_size, n_embd),
    "lm_head": make_matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = make_matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = make_matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = make_matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = make_matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = make_matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = make_matrix(n_embd, 4 * n_embd)

params = list(state_dict.values())
print(f"num params: {sum(p.data.size for p in params)}")

#
# pre-compile jit kernels
#

print("compiling kernels...", end=" ", flush=True)
jit_embedding(n_embd)
jit_add(n_embd)
jit_rmsnorm(n_embd)
jit_matvec(n_embd, n_embd)
jit_matvec(4 * n_embd, n_embd)
jit_matvec(n_embd, 4 * n_embd)
jit_matvec(vocab_size, n_embd)
jit_relu(4 * n_embd)
jit_dot(head_dim)
jit_cross_entropy(vocab_size)
for t in range(1, block_size + 1):
    _jit_max(t)
    _jit_core(t)
    jit_weighted_sum(t, head_dim)
print("done")

#
# forward pass
#


def gpt(token_id, pos_id, keys, values):
    # token + position embedding
    tok_emb = t_embedding(state_dict["wte"], token_id)
    pos_emb = t_embedding(state_dict["wpe"], pos_id)
    x = t_add(tok_emb, pos_emb)
    x = t_rmsnorm(x)

    for li in range(n_layer):
        # attention block
        x_res = x
        x = t_rmsnorm(x)
        q = t_linear(x, state_dict[f"layer{li}.attn_wq"])
        k = t_linear(x, state_dict[f"layer{li}.attn_wk"])
        v = t_linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = t_attention(q, keys[li], values[li], n_head, head_dim)
        x = t_add(t_linear(x_attn, state_dict[f"layer{li}.attn_wo"]), x_res)

        # mlp block
        x_res = x
        x = t_linear(t_rmsnorm(x), state_dict[f"layer{li}.mlp_fc1"])
        x = t_relu(x)
        x = t_add(t_linear(x, state_dict[f"layer{li}.mlp_fc2"]), x_res)

    logits = t_linear(x, state_dict["lm_head"])
    return logits


#
# training loop
#

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
adam_m = [np.zeros_like(p.data) for p in params]
adam_v = [np.zeros_like(p.data) for p in params]

num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # forward: build computation graph through all positions
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        loss_t = t_cross_entropy(logits, target_id)
        losses.append(loss_t)
    loss = t_mean_loss(losses)

    # backward
    loss.backward()

    # adam update with linear lr decay
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        adam_m[i] = beta1 * adam_m[i] + (1 - beta1) * p.grad
        adam_v[i] = beta2 * adam_v[i] + (1 - beta2) * p.grad**2
        m_hat = adam_m[i] / (1 - beta1 ** (step + 1))
        v_hat = adam_v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)
        p.grad = np.zeros_like(p.grad)

    print(f"step {step + 1:4d} / {num_steps:4d} | loss {float(loss.data[0]):.4f}", end="\r")

#
# inference
#

temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        # softmax with temperature
        logits_scaled = logits.data / temperature
        shifted = logits_scaled - logits_scaled.max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")
