from __future__ import annotations

from exo import *
from exo.libs.externs import select, sqrt
from exo.platforms.neon import Neon

__all__ = [
    "neon_add_acc_f32x4",
    "neon_add_f32x4",
    "neon_broadcast_f32x4",
    "neon_broadcast_f64x2",
    "neon_div_acc_f32x4",
    "neon_div_f32x4",
    "neon_fmadd_f32x4",
    "neon_fmadd_f64x2",
    "neon_fmax_acc_f32x4",
    "neon_loadu_f32x4",
    "neon_loadu_f64x2",
    "neon_mul_acc_f32x4",
    "neon_mul_f32x4",
    "neon_sqrt_f32x4",
    "neon_square_f32x4",
    "neon_storeu_f32x4",
    "neon_storeu_f64x2",
    "neon_sub_acc_f32x4",
    "neon_sub_f32x4",
    "vec_add_f32x4",
    "vec_add_f64x2",
    "vec_add_red_f32x4",
    "vec_copy_f32x4",
    "vec_mul_f32x4",
    "vec_mul_f64x2",
    "vec_neg_f32x4",
]


@instr("neon_add_acc_f32x4({acc_data}, {src_data});")
def neon_add_acc_f32x4(acc: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] += src[i]


@instr("neon_add_f32x4({dst_data}, {a_data}, {b_data});")
def neon_add_f32x4(dst: [f32][4] @ Neon, a: [f32][4] @ Neon, b: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ Neon, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


@instr("neon_broadcast_f64x2({dst_data}, {src_data});")
def neon_broadcast_f64x2(dst: [f64][2] @ Neon, src: [f64][1] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 2):
        dst[i] = src[0]


@instr("neon_div_acc_f32x4({acc_data}, {src_data});")
def neon_div_acc_f32x4(acc: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = acc[i] / src[i]


@instr("neon_div_f32x4({dst_data}, {a_data}, {b_data});")
def neon_div_f32x4(dst: [f32][4] @ Neon, a: [f32][4] @ Neon, b: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] / b[i]


@instr("neon_fmadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f32x4(dst: [f32][4] @ Neon, a: [f32][4] @ Neon, b: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] += a[i] * b[i]


@instr("neon_fmadd_f64x2({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f64x2(dst: [f64][2] @ Neon, a: [f64][2] @ Neon, b: [f64][2] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 2):
        dst[i] += a[i] * b[i]


@instr("neon_fmax_acc_f32x4({acc_data}, {src_data});")
def neon_fmax_acc_f32x4(acc: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = select(acc[i], src[i], src[i], acc[i])


@instr("neon_loadu_f32x4({dst_data}, {src_data});")
def neon_loadu_f32x4(dst: [f32][4] @ Neon, src: [f32][4] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_loadu_f64x2({dst_data}, {src_data});")
def neon_loadu_f64x2(dst: [f64][2] @ Neon, src: [f64][2] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 2):
        dst[i] = src[i]


@instr("neon_mul_acc_f32x4({acc_data}, {src_data});")
def neon_mul_acc_f32x4(acc: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = acc[i] * src[i]


@instr("neon_mul_f32x4({dst_data}, {a_data}, {b_data});")
def neon_mul_f32x4(dst: [f32][4] @ Neon, a: [f32][4] @ Neon, b: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] * b[i]


@instr("neon_sqrt_f32x4({dst_data}, {src_data});")
def neon_sqrt_f32x4(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = sqrt(src[i])


@instr("neon_square_f32x4({dst_data}, {src_data});")
def neon_square_f32x4(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i] * src[i]


@instr("neon_storeu_f32x4({dst_data}, {src_data});")
def neon_storeu_f32x4(dst: [f32][4] @ DRAM, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_storeu_f64x2({dst_data}, {src_data});")
def neon_storeu_f64x2(dst: [f64][2] @ DRAM, src: [f64][2] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 2):
        dst[i] = src[i]


@instr("neon_sub_acc_f32x4({acc_data}, {src_data});")
def neon_sub_acc_f32x4(acc: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = acc[i] - src[i]


@instr("neon_sub_f32x4({dst_data}, {a_data}, {b_data});")
def neon_sub_f32x4(dst: [f32][4] @ Neon, a: [f32][4] @ Neon, b: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] - b[i]


@instr("vec_add_f32x4({dst_data}, {a_data}, {b_data});")
def vec_add_f32x4(dst: [f32][4] @ Neon, a: [f32][4] @ Neon, b: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


@instr("vec_add_f64x2({dst_data}, {a_data}, {b_data});")
def vec_add_f64x2(dst: [f64][2] @ Neon, a: [f64][2] @ Neon, b: [f64][2] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 2):
        dst[i] = a[i] + b[i]


@instr("vec_add_red_f32x4({dst_data}, {src_data});")
def vec_add_red_f32x4(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] += src[i]


@instr("vec_copy_f32x4({dst_data}, {src_data});")
def vec_copy_f32x4(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("vec_mul_f32x4({dst_data}, {a_data}, {b_data});")
def vec_mul_f32x4(dst: [f32][4] @ Neon, a: [f32][4] @ Neon, b: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] * b[i]


@instr("vec_mul_f64x2({dst_data}, {a_data}, {b_data});")
def vec_mul_f64x2(dst: [f64][2] @ Neon, a: [f64][2] @ Neon, b: [f64][2] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 2):
        dst[i] = a[i] * b[i]


@instr("vec_neg_f32x4({dst_data}, {src_data});")
def vec_neg_f32x4(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = -src[i]
