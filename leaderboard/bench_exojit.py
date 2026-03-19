# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
#   "numpy",
#   "tqdm",
# ]
# ///

import math
import random
import sys
import time
from collections import namedtuple
from functools import cache
from pathlib import Path

import numpy as np
from exo import *
from exo.libs.externs import select, sqrt
from exo.stdlib.scheduling import rename, simplify
from tqdm import tqdm
from utils import assert_weights_match, save_times

from exojit.main import compile_jit
from exojit.patches_exo import Stack

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD
NUM_STEPS = 1000

BLOCK_RANGE = tuple(range(BLOCK_SIZE))
INV_SCALE = 1.0 / HEAD_DIM**0.5
CAUSAL_MASK_VALUE = -1e10
SOFTMAX_LIMIT = 1e30
LOSS_FLOOR = sys.float_info.min
ADAM_PARAMS = {
    "LR_T": [0.01 * (1.0 - step / NUM_STEPS) for step in range(NUM_STEPS)],
    "BC1": [1.0 - 0.85 ** (step + 1) for step in range(NUM_STEPS)],
    "BC2": [1.0 - 0.99 ** (step + 1) for step in range(NUM_STEPS)],
}


AttnCache = namedtuple("AttnCache", ["x_pre", "xn", "rms", "q", "k", "v", "attn_w", "out_flat"])
MlpCache = namedtuple("MlpCache", ["x_pre", "xn", "rms", "h_pre", "h"])
FwdCache = namedtuple("FwdCache", ["input_ids", "target_ids", "loss_mask", "sum_mask", "emb", "rms_init", "x", "probs", "layer_caches"])


def scalar_array(value: float) -> np.ndarray:
    out = np.empty(1, dtype=np.float64)
    out[0] = value
    return out


def empty_array(shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


def array_numel(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def zeros_array(shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    if len(shape) == 1:
        for i in range(shape[0]):
            out[i] = 0
    else:
        for idx in iter_indices(shape):
            out[idx] = 0
    return out


def empty_like_array(x: np.ndarray) -> np.ndarray:
    return empty_array(x.shape, x.dtype)


def flat_view(x: np.ndarray, offset: int, shape: tuple[int, ...]) -> np.ndarray:
    size = array_numel(shape)
    return x[offset : offset + size].reshape(shape)


def random_matrix(nout: int, nin: int, std: float = 0.08) -> np.ndarray:
    out = empty_array((nout, nin), dtype=np.float64)
    for i in range(nout):
        for j in range(nin):
            out[i, j] = random.gauss(0.0, std)
    return out


RMS_INV_N = scalar_array(1.0 / N_EMBED)
RMS_EPS = scalar_array(1e-5)
ADAM_B1 = scalar_array(0.85)
ADAM_B2 = scalar_array(0.99)
ADAM_EPS = scalar_array(1e-8)
INV_SCALE_ARRAY = scalar_array(INV_SCALE)
CAUSAL_MASK_ARRAY = scalar_array(CAUSAL_MASK_VALUE)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    if axis != -1:
        raise ValueError("softmax only supports axis=-1")
    shape = x.shape
    softmax_row = _jit_softmax_row(shape[-1])
    if len(shape) == 2:
        for i in range(shape[0]):
            softmax_row(x[i], x[i])
    elif len(shape) == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                softmax_row(x[i, j], x[i, j])
    else:
        raise ValueError(f"unsupported softmax rank: {len(shape)}")
    return x


def _sanitize_scalar(value: float) -> float:
    if math.isnan(value):
        return -SOFTMAX_LIMIT
    if math.isinf(value):
        return SOFTMAX_LIMIT if value > 0 else -SOFTMAX_LIMIT
    return value


@proc
def _matmul_nt(M: size, K: size, N: size, out: f64[M, N] @ DRAM, a: f64[M, K] @ DRAM, b: f64[N, K] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = 0.0
            for k in seq(0, K):
                acc += a[i, k] * b[j, k]
            out[i, j] = acc


@proc
def _matmul_nn(M: size, K: size, N: size, out: f64[M, N] @ DRAM, a: f64[M, K] @ DRAM, b: f64[K, N] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = 0.0
            for k in seq(0, K):
                acc += a[i, k] * b[k, j]
            out[i, j] = acc


@proc
def _matmul_tn(M: size, K: size, N: size, out: f64[K, N] @ DRAM, a: f64[M, K] @ DRAM, b: f64[M, N] @ DRAM):
    for i in seq(0, K):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = 0.0
            for k in seq(0, M):
                acc += a[k, i] * b[k, j]
            out[i, j] = acc


@cache
def _jit_matmul_nt(m: int, k: int, n: int):
    p = simplify(_matmul_nt.partial_eval(M=m, K=k, N=n))
    name = f"_matmul_nt_{m}_{k}_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_matmul_nn(m: int, k: int, n: int):
    p = simplify(_matmul_nn.partial_eval(M=m, K=k, N=n))
    name = f"_matmul_nn_{m}_{k}_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_matmul_tn(m: int, k: int, n: int):
    p = simplify(_matmul_tn.partial_eval(M=m, K=k, N=n))
    name = f"_matmul_tn_{m}_{k}_{n}"
    return compile_jit(rename(p, name))[name]


def matmul_nt(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    rows, inner = a.shape
    out_cols, b_inner = b.shape
    if inner != b_inner or out.shape != (rows, out_cols):
        raise ValueError("shape mismatch in matmul_nt")
    _jit_matmul_nt(rows, inner, out_cols)(out, a, b)
    return out


def matmul_nn(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    rows, inner = a.shape
    b_inner, cols = b.shape
    if inner != b_inner or out.shape != (rows, cols):
        raise ValueError("shape mismatch in matmul_nn")
    _jit_matmul_nn(rows, inner, cols)(out, a, b)
    return out


def matmul_tn(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
    rows, inner = a.shape
    b_rows, cols = b.shape
    if rows != b_rows or out.shape != (inner, cols):
        raise ValueError("shape mismatch in matmul_tn")
    _jit_matmul_tn(rows, inner, cols)(out, a, b)
    return out


def iter_indices(shape: tuple[int, ...]):
    if len(shape) == 1:
        for i in range(shape[0]):
            yield (i,)
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                yield (i, j)
    elif len(shape) == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    yield (i, j, k)
    else:
        raise ValueError(f"unsupported shape rank: {len(shape)}")


def add_inplace(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    for idx in iter_indices(dst.shape):
        dst[idx] += float(src[idx])
    return dst


def relu_inplace(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    for idx in iter_indices(src.shape):
        val = float(src[idx])
        dst[idx] = val if val > 0.0 else 0.0
    return dst


def relu_backward_masked_mul(out: np.ndarray, grad: np.ndarray, preact: np.ndarray) -> np.ndarray:
    for idx in iter_indices(grad.shape):
        out[idx] = float(grad[idx]) if float(preact[idx]) > 0.0 else 0.0
    return out


def sum_array(x: np.ndarray) -> float:
    total = 0.0
    for idx in iter_indices(x.shape):
        total += float(x[idx])
    return total


def copy_array(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    for idx in iter_indices(src.shape):
        dst[idx] = src[idx]
    return dst


def zero_array(x: np.ndarray) -> np.ndarray:
    for idx in iter_indices(x.shape):
        x[idx] = 0.0
    return x


def add_rows_at(dst: np.ndarray, row_ids: np.ndarray, src: np.ndarray) -> np.ndarray:
    for row in range(src.shape[0]):
        dst_row = int(row_ids[row])
        for col in range(src.shape[1]):
            dst[dst_row, col] += float(src[row, col])
    return dst


@proc
def _attn_logits(logits: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, inv_scale: f64[1] @ DRAM, causal_mask: f64[1] @ DRAM):
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for s in seq(0, BLOCK_SIZE):
                if s > t:
                    logits[h, t, s] = causal_mask[0]
                else:
                    acc: f64 @ Stack
                    acc = 0.0
                    for d in seq(0, HEAD_DIM):
                        acc += q[h, t, d] * k[h, s, d]
                    logits[h, t, s] = acc * inv_scale[0]


@proc
def _softmax_row(N: size, out: f64[N] @ DRAM, inp: f64[N] @ DRAM):
    mx: f64 @ Stack
    sum_val: f64 @ Stack
    t: f64 @ Stack
    y: f64 @ Stack
    e5: f64 @ Stack
    e4: f64 @ Stack
    e3: f64 @ Stack
    e2: f64 @ Stack
    e1: f64 @ Stack
    s1: f64 @ Stack
    s2: f64 @ Stack
    s3: f64 @ Stack
    s4: f64 @ Stack
    s5: f64 @ Stack

    mx = inp[0]
    for i in seq(1, N):
        mx = select(mx, inp[i], inp[i], mx)

    sum_val = 0.0
    for j in seq(0, N):
        t = inp[j] - mx
        y = t * 0.03125
        e5 = y * 0.008333333333333333 + 0.041666666666666664
        e4 = e5 * y + 0.16666666666666666
        e3 = e4 * y + 0.5
        e2 = e3 * y + 1.0
        e1 = e2 * y + 1.0
        s1 = e1 * e1
        s2 = s1 * s1
        s3 = s2 * s2
        s4 = s3 * s3
        s5 = s4 * s4
        out[j] = s5
        sum_val += s5

    for k in seq(0, N):
        out[k] = out[k] / sum_val


@proc
def _attn_softmax_bwd(dlogits: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, dattn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, inv_scale: f64[1] @ DRAM):
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            dot: f64 @ Stack
            dot = 0.0
            for s in seq(0, BLOCK_SIZE):
                dot += dattn_w[h, t, s] * attn_w[h, t, s]
            for s in seq(0, BLOCK_SIZE):
                dlogits[h, t, s] = attn_w[h, t, s] * (dattn_w[h, t, s] - dot) * inv_scale[0]


@cache
def _jit_attn_logits():
    p = simplify(_attn_logits)
    name = "_attn_logits"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_softmax_row(n: int):
    p = simplify(_softmax_row.partial_eval(N=n))
    name = f"_softmax_row_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_attn_softmax_bwd():
    p = simplify(_attn_softmax_bwd)
    name = "_attn_softmax_bwd"
    return compile_jit(rename(p, name))[name]


def build_attention_logits(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    logits = empty_array((N_HEAD, BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
    _jit_attn_logits()(logits, q, k, INV_SCALE_ARRAY, CAUSAL_MASK_ARRAY)
    return logits


def attention_softmax_backward(attn_w: np.ndarray, dattn_w: np.ndarray) -> np.ndarray:
    dlogits = empty_like_array(attn_w)
    _jit_attn_softmax_bwd()(dlogits, attn_w, dattn_w, INV_SCALE_ARRAY)
    return dlogits


def cross_entropy_loss(probs: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray, sum_mask: float) -> float:
    total = 0.0
    for i in range(BLOCK_SIZE):
        weight = float(loss_mask[i])
        if weight == 0.0:
            continue
        prob = float(probs[i, int(target_ids[i])])
        clipped = min(1.0, max(prob, LOSS_FLOOR))
        total -= math.log(clipped) * weight
    return total / sum_mask


@proc
def _rmsnorm_fwd(M: size, N: size, out: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, inp: f64[M, N] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    for i in seq(0, M):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N):
            sumsq += inp[i, j] * inp[i, j]
        scale = 1.0 / sqrt(sumsq * inv_n[0] + eps[0])
        rms[i, 0] = scale
        for j in seq(0, N):
            out[i, j] = inp[i, j] * scale


@proc
def _rmsnorm_bwd(M: size, N: size, dx: f64[M, N] @ DRAM, dout: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, inv_n: f64[1] @ DRAM):
    for i in seq(0, M):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = 0.0
        scale = rms[i, 0]
        for j in seq(0, N):
            dot += dout[i, j] * x[i, j]
        corr = scale * scale * scale * inv_n[0] * dot
        for j in seq(0, N):
            dx[i, j] = dout[i, j] * scale - x[i, j] * corr


@proc
def _attn_qkv_fwd(q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM):
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc_q: f64 @ Stack
                acc_k: f64 @ Stack
                acc_v: f64 @ Stack
                acc_q = 0.0
                acc_k = 0.0
                acc_v = 0.0
                for e in seq(0, N_EMBED):
                    acc_q += xn[t, e] * wq[h * HEAD_DIM + d, e]
                    acc_k += xn[t, e] * wk[h * HEAD_DIM + d, e]
                    acc_v += xn[t, e] * wv[h * HEAD_DIM + d, e]
                q[h, t, d] = acc_q
                k[h, t, d] = acc_k
                v[h, t, d] = acc_v


@proc
def _attn_av_fwd(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM):
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc: f64 @ Stack
                acc = 0.0
                for s in seq(0, BLOCK_SIZE):
                    acc += attn_w[h, t, s] * v[h, s, d]
                out[t, h * HEAD_DIM + d] = acc


@proc
def _attn_dv_dattnw(dv: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dattn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM):
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for s in seq(0, BLOCK_SIZE):
                acc_w: f64 @ Stack
                acc_w = 0.0
                for d in seq(0, HEAD_DIM):
                    acc_w += dattn_out[h, t, d] * v[h, s, d]
                dattn_w[h, t, s] = acc_w
        for s in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc_v: f64 @ Stack
                acc_v = 0.0
                for t in seq(0, BLOCK_SIZE):
                    acc_v += attn_w[h, t, s] * dattn_out[h, t, d]
                dv[h, s, d] = acc_v


@proc
def _attn_dq_dk(dq: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dk: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dlogits: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM):
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc_q: f64 @ Stack
                acc_q = 0.0
                for s in seq(0, BLOCK_SIZE):
                    acc_q += dlogits[h, t, s] * k[h, s, d]
                dq[h, t, d] = acc_q
        for s in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc_k: f64 @ Stack
                acc_k = 0.0
                for t in seq(0, BLOCK_SIZE):
                    acc_k += dlogits[h, t, s] * q[h, t, d]
                dk[h, s, d] = acc_k


@proc
def _attn_qkv_bwd(dwq: f64[N_EMBED, N_EMBED] @ DRAM, dwk: f64[N_EMBED, N_EMBED] @ DRAM, dwv: f64[N_EMBED, N_EMBED] @ DRAM, dxn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dq: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dk: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dv: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            acc_x: f64 @ Stack
            acc_x = 0.0
            for h in seq(0, N_HEAD):
                for d in seq(0, HEAD_DIM):
                    acc_x += dq[h, t, d] * wq[h * HEAD_DIM + d, e]
                    acc_x += dk[h, t, d] * wk[h * HEAD_DIM + d, e]
                    acc_x += dv[h, t, d] * wv[h * HEAD_DIM + d, e]
            dxn[t, e] = acc_x

    for h in seq(0, N_HEAD):
        for d in seq(0, HEAD_DIM):
            for e in seq(0, N_EMBED):
                acc_q: f64 @ Stack
                acc_k: f64 @ Stack
                acc_v: f64 @ Stack
                acc_q = 0.0
                acc_k = 0.0
                acc_v = 0.0
                for t in seq(0, BLOCK_SIZE):
                    acc_q += dq[h, t, d] * xn[t, e]
                    acc_k += dk[h, t, d] * xn[t, e]
                    acc_v += dv[h, t, d] * xn[t, e]
                dwq[h * HEAD_DIM + d, e] = acc_q
                dwk[h * HEAD_DIM + d, e] = acc_k
                dwv[h * HEAD_DIM + d, e] = acc_v


@proc
def _adam(N: size, param: f64[N] @ DRAM, grad: f64[N] @ DRAM, m: f64[N] @ DRAM, v: f64[N] @ DRAM, b1: f64[1] @ DRAM, b2: f64[1] @ DRAM, eps: f64[1] @ DRAM, lr: f64[1] @ DRAM, beta1_t: f64[1] @ DRAM, beta2_t: f64[1] @ DRAM):
    inv_b1: f64 @ Stack
    inv_b2: f64 @ Stack
    inv_beta1_t: f64 @ Stack
    inv_beta2_t: f64 @ Stack
    inv_b1 = 1.0 - b1[0]
    inv_b2 = 1.0 - b2[0]
    inv_beta1_t = 1.0 / beta1_t[0]
    inv_beta2_t = 1.0 / beta2_t[0]

    for i in par(0, N):
        g: f64 @ Stack
        m_val: f64 @ Stack
        v_val: f64 @ Stack
        m_hat: f64 @ Stack
        v_hat: f64 @ Stack
        g = grad[i]
        m_val = b1[0] * m[i] + inv_b1 * g
        v_val = b2[0] * v[i] + inv_b2 * g * g
        m_hat = m_val * inv_beta1_t
        v_hat = v_val * inv_beta2_t
        param[i] = param[i] - lr[0] * m_hat / (sqrt(v_hat) + eps[0])
        m[i] = m_val
        v[i] = v_val


@cache
def _jit_rmsnorm_fwd(m: int, n: int):
    p = simplify(_rmsnorm_fwd.partial_eval(M=m, N=n))
    name = f"_rmsnorm_fwd_{m}_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_rmsnorm_bwd(m: int, n: int):
    p = simplify(_rmsnorm_bwd.partial_eval(M=m, N=n))
    name = f"_rmsnorm_bwd_{m}_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_attn_qkv_fwd():
    p = simplify(_attn_qkv_fwd)
    name = "_attn_qkv_fwd"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_attn_av_fwd():
    p = simplify(_attn_av_fwd)
    name = "_attn_av_fwd"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_attn_dv_dattnw():
    p = simplify(_attn_dv_dattnw)
    name = "_attn_dv_dattnw"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_attn_dq_dk():
    p = simplify(_attn_dq_dk)
    name = "_attn_dq_dk"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_attn_qkv_bwd():
    p = simplify(_attn_qkv_bwd)
    name = "_attn_qkv_bwd"
    return compile_jit(rename(p, name))[name]


@cache
def _jit_adam(n: int):
    p = simplify(_adam.partial_eval(N=n))
    name = f"_adam_{n}"
    return compile_jit(rename(p, name))[name]


def rmsnorm_fwd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fn = _jit_rmsnorm_fwd(*x.shape)
    out = empty_like_array(x)
    rms = empty_array((x.shape[0], 1), dtype=np.float64)
    fn(out, rms, x, RMS_INV_N, RMS_EPS)
    return out, rms


def rmsnorm_bwd(dout: np.ndarray, x: np.ndarray, rms: np.ndarray) -> np.ndarray:
    fn = _jit_rmsnorm_bwd(*x.shape)
    dx = empty_like_array(x)
    fn(dx, dout, x, rms, RMS_INV_N)
    return dx


def attn_fwd(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray, wo: np.ndarray) -> tuple[np.ndarray, AttnCache]:
    xn, rms = rmsnorm_fwd(x)
    q = empty_array((N_HEAD, BLOCK_SIZE, HEAD_DIM), dtype=np.float64)
    k = empty_like_array(q)
    v = empty_like_array(q)
    _jit_attn_qkv_fwd()(q, k, v, xn, wq, wk, wv)

    attn_logits = build_attention_logits(q, k)
    attn_w = softmax(attn_logits)

    out_flat = empty_array((BLOCK_SIZE, N_EMBED), dtype=np.float64)
    _jit_attn_av_fwd()(out_flat, attn_w, v)
    out = empty_array((BLOCK_SIZE, N_EMBED), dtype=np.float64)
    matmul_nt(out_flat, wo, out)
    add_inplace(out, x)
    return out, AttnCache(x, xn, rms, q, k, v, attn_w, out_flat)


def attn_bwd(dx: np.ndarray, grads: dict, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray, wo: np.ndarray, c: AttnCache, li: int) -> np.ndarray:
    matmul_tn(dx, c.out_flat, grads[f"layer{li}.attn_wo"])
    dattn_out_flat = empty_array((BLOCK_SIZE, N_EMBED), dtype=np.float64)
    matmul_nn(dx, wo, dattn_out_flat)
    dattn_out = empty_array((N_HEAD, BLOCK_SIZE, HEAD_DIM), dtype=np.float64)
    for t in range(BLOCK_SIZE):
        for h in range(N_HEAD):
            for d in range(HEAD_DIM):
                dattn_out[h, t, d] = dattn_out_flat[t, h * HEAD_DIM + d]

    dv = empty_like_array(c.v)
    dattn_w = empty_like_array(c.attn_w)
    _jit_attn_dv_dattnw()(dv, dattn_w, dattn_out, c.attn_w, c.v)

    dlogits_attn = attention_softmax_backward(c.attn_w, dattn_w)
    dq = empty_like_array(c.q)
    dk = empty_like_array(c.k)
    _jit_attn_dq_dk()(dq, dk, dlogits_attn, c.q, c.k)

    dxn = empty_like_array(c.xn)
    _jit_attn_qkv_bwd()(
        grads[f"layer{li}.attn_wq"],
        grads[f"layer{li}.attn_wk"],
        grads[f"layer{li}.attn_wv"],
        dxn,
        dq,
        dk,
        dv,
        c.xn,
        wq,
        wk,
        wv,
    )
    out = rmsnorm_bwd(dxn, c.x_pre, c.rms)
    add_inplace(out, dx)
    return out


def mlp_fwd(x: np.ndarray, fc1: np.ndarray, fc2: np.ndarray) -> tuple[np.ndarray, MlpCache]:
    xn, rms = rmsnorm_fwd(x)
    h_pre = empty_array((BLOCK_SIZE, 4 * N_EMBED), dtype=np.float64)
    matmul_nt(xn, fc1, h_pre)
    h = empty_like_array(h_pre)
    relu_inplace(h, h_pre)
    out = empty_array((BLOCK_SIZE, N_EMBED), dtype=np.float64)
    matmul_nt(h, fc2, out)
    add_inplace(out, x)
    return out, MlpCache(x, xn, rms, h_pre, h)


def mlp_bwd(dx: np.ndarray, grads: dict, fc1: np.ndarray, fc2: np.ndarray, c: MlpCache, li: int) -> np.ndarray:
    matmul_tn(dx, c.h, grads[f"layer{li}.mlp_fc2"])
    dh = empty_array((BLOCK_SIZE, 4 * N_EMBED), dtype=np.float64)
    matmul_nn(dx, fc2, dh)
    dh_pre = empty_like_array(dh)
    relu_backward_masked_mul(dh_pre, dh, c.h_pre)
    matmul_tn(dh_pre, c.xn, grads[f"layer{li}.mlp_fc1"])
    dx_resid = empty_array((BLOCK_SIZE, N_EMBED), dtype=np.float64)
    matmul_nn(dh_pre, fc1, dx_resid)
    out = rmsnorm_bwd(dx_resid, c.x_pre, c.rms)
    add_inplace(out, dx)
    return out


def forward(params: dict, input_ids: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray) -> tuple[float, FwdCache]:
    emb = empty_array((BLOCK_SIZE, N_EMBED), dtype=np.float64)
    for i in range(BLOCK_SIZE):
        tok = int(input_ids[i])
        for j in range(N_EMBED):
            emb[i, j] = float(params["wte"][tok, j]) + float(params["wpe"][i, j])
    x, rms_init = rmsnorm_fwd(emb)

    layer_caches = []
    for li in range(N_LAYER):
        x, ac = attn_fwd(x, params[f"layer{li}.attn_wq"], params[f"layer{li}.attn_wk"], params[f"layer{li}.attn_wv"], params[f"layer{li}.attn_wo"])
        x, mc = mlp_fwd(x, params[f"layer{li}.mlp_fc1"], params[f"layer{li}.mlp_fc2"])
        layer_caches.append((ac, mc))

    logits = empty_array((BLOCK_SIZE, params["lm_head"].shape[0]), dtype=np.float64)
    matmul_nt(x, params["lm_head"], logits)
    probs = softmax(logits)
    sum_mask = sum_array(loss_mask)
    loss = cross_entropy_loss(probs, target_ids, loss_mask, sum_mask)
    return float(loss), FwdCache(input_ids, target_ids, loss_mask, sum_mask, emb, rms_init, x, probs, layer_caches)


def backward(params: dict, grads: dict, cache: FwdCache) -> None:
    zero_array(grads["wte"])
    zero_array(grads["wpe"])

    dlogits = empty_like_array(cache.probs)
    copy_array(dlogits, cache.probs)
    inv_sum_mask = 1.0 / cache.sum_mask
    for i in range(BLOCK_SIZE):
        for j in range(dlogits.shape[1]):
            dlogits[i, j] *= inv_sum_mask
        dlogits[i, int(cache.target_ids[i])] -= inv_sum_mask
        weight = float(cache.loss_mask[i])
        for j in range(dlogits.shape[1]):
            dlogits[i, j] *= weight

    matmul_tn(dlogits, cache.x, grads["lm_head"])
    dx = empty_array((BLOCK_SIZE, N_EMBED), dtype=np.float64)
    matmul_nn(dlogits, params["lm_head"], dx)

    for li in reversed(range(N_LAYER)):
        ac, mc = cache.layer_caches[li]
        dx = mlp_bwd(dx, grads, params[f"layer{li}.mlp_fc1"], params[f"layer{li}.mlp_fc2"], mc, li)
        dx = attn_bwd(dx, grads, params[f"layer{li}.attn_wq"], params[f"layer{li}.attn_wk"], params[f"layer{li}.attn_wv"], params[f"layer{li}.attn_wo"], ac, li)

    demb = rmsnorm_bwd(dx, cache.emb, cache.rms_init)
    add_rows_at(grads["wte"], cache.input_ids, demb)
    add_inplace(grads["wpe"], demb)


def step_fn(params: dict, opt_state: dict, grads: dict, input_ids: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray, step: int) -> tuple[float, dict, dict]:
    loss, cache = forward(params, input_ids, target_ids, loss_mask)
    backward(params, grads, cache)

    opt_state["lr"][0] = ADAM_PARAMS["LR_T"][step]
    opt_state["bc1"][0] = ADAM_PARAMS["BC1"][step]
    opt_state["bc2"][0] = ADAM_PARAMS["BC2"][step]
    _jit_adam(opt_state["flat_params"].shape[0])(
        opt_state["flat_params"],
        opt_state["flat_grads"],
        opt_state["flat_m"],
        opt_state["flat_v"],
        ADAM_B1,
        ADAM_B2,
        ADAM_EPS,
        opt_state["lr"],
        opt_state["bc1"],
        opt_state["bc2"],
    )
    return loss, params, opt_state


def tokenize(doc: str, uchars: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = len(uchars)
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)

    input_ids = zeros_array((BLOCK_SIZE,), dtype=np.int32)
    target_ids = zeros_array((BLOCK_SIZE,), dtype=np.int32)
    loss_mask = zeros_array((BLOCK_SIZE,), dtype=np.float64)
    for i in range(n):
        input_ids[i] = tokens[i]
        target_ids[i] = tokens[i + 1]
        loss_mask[i] = 1.0
    return input_ids, target_ids, loss_mask


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

state_dict = {
    "wte": random_matrix(len(uchars) + 1, N_EMBED),
    "wpe": random_matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": random_matrix(len(uchars) + 1, N_EMBED),
    **{f"layer{i}.attn_wq": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wk": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wv": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wo": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc1": random_matrix(4 * N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc2": random_matrix(N_EMBED, 4 * N_EMBED) for i in range(N_LAYER)},
}

total_params = sum(array_numel(state_dict[k].shape) for k in state_dict)
flat_params = empty_array((total_params,), dtype=np.float64)
offset = 0
for k in state_dict:
    arr = state_dict[k]
    for idx in iter_indices(arr.shape):
        flat_params[offset] = arr[idx]
        offset += 1
offset = 0
for k in state_dict:
    shape = state_dict[k].shape
    state_dict[k] = flat_view(flat_params, offset, shape)
    offset += array_numel(shape)

flat_grads = zeros_array((total_params,), dtype=np.float64)
grads = {}
offset = 0
for k in state_dict:
    shape = state_dict[k].shape
    grads[k] = flat_view(flat_grads, offset, shape)
    offset += array_numel(shape)

opt_state = {
    "flat_m": zeros_array((total_params,), dtype=np.float64),
    "flat_v": zeros_array((total_params,), dtype=np.float64),
    "flat_params": flat_params,
    "flat_grads": flat_grads,
    "lr": empty_array((1,), dtype=np.float64),
    "bc1": empty_array((1,), dtype=np.float64),
    "bc2": empty_array((1,), dtype=np.float64),
}

tokenized = [tokenize(doc, uchars) for doc in docs]

_jit_rmsnorm_fwd(BLOCK_SIZE, N_EMBED)
_jit_rmsnorm_bwd(BLOCK_SIZE, N_EMBED)
_jit_attn_qkv_fwd()
_jit_attn_logits()
_jit_softmax_row(BLOCK_SIZE)
_jit_attn_av_fwd()
_jit_attn_dv_dattnw()
_jit_attn_softmax_bwd()
_jit_attn_dq_dk()
_jit_attn_qkv_bwd()
_jit_matmul_nt(BLOCK_SIZE, N_EMBED, N_EMBED)
_jit_matmul_nn(BLOCK_SIZE, N_EMBED, N_EMBED)
_jit_matmul_tn(BLOCK_SIZE, N_EMBED, N_EMBED)
_jit_matmul_nt(BLOCK_SIZE, N_EMBED, 4 * N_EMBED)
_jit_matmul_nn(BLOCK_SIZE, N_EMBED, 4 * N_EMBED)
_jit_matmul_tn(BLOCK_SIZE, 4 * N_EMBED, N_EMBED)
_jit_matmul_nt(BLOCK_SIZE, N_EMBED, len(uchars) + 1)
_jit_matmul_nn(BLOCK_SIZE, len(uchars) + 1, N_EMBED)
_jit_matmul_tn(BLOCK_SIZE, len(uchars) + 1, N_EMBED)
_jit_adam(total_params)

step_times = []
for step in tqdm(range(NUM_STEPS)):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, grads, input_ids, target_ids, loss_mask, step)
    step_times.append(time.perf_counter() - t0)

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
