from __future__ import annotations

import timeit
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

import xnumpy as xnp

REPEATS = 200


bench = lambda fn: min(timeit.repeat(fn, number=1, repeat=REPEATS))

np.random.seed(42)
rows: list[dict[str, object]] = []

matmul_sizes = [32, 64, 128, 256]


for n in tqdm(matmul_sizes, desc="matmul"):
    A_np = np.random.randn(n, n).astype(np.float32)
    B_np = np.random.randn(n, n).astype(np.float32)
    A_xnp = xnp.array(A_np)
    B_xnp = xnp.array(B_np)

    assert xnp.allclose(A_xnp @ B_xnp, A_np @ B_np, atol=1e-3)

    t_np = bench(lambda A=A_np, B=B_np: A @ B)
    t_xnp = bench(lambda A=A_xnp, B=B_xnp: A @ B)
    rows.append({"op": "matmul", "n": n, "numpy_us": t_np * 1e6, "xnumpy_us": t_xnp * 1e6, "speedup": t_xnp / t_np})


if __name__ == "__main__":
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("numpy_us").round(4), pl.col("xnumpy_us").round(4), pl.col("speedup").round(4))
    print(df)
    df.write_csv(Path(__file__).parent / "results.csv")
