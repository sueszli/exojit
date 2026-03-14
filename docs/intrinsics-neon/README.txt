ARM NEON Intrinsics Reference
=============================

All files downloaded from public sources on 2026-03-14.

LLVM/Clang source files (from https://github.com/llvm/llvm-project, branch: main):

  Intrinsics.td
    Master TableGen file defining all LLVM intrinsics (base classes, properties, type system).
    Source: llvm/include/llvm/IR/Intrinsics.td

  IntrinsicsAArch64.td
    AArch64-specific intrinsic definitions: NEON, SVE, SME, crypto, memory ops.
    Source: llvm/include/llvm/IR/IntrinsicsAArch64.td

  arm_neon.td
    TableGen source that defines every C-level NEON intrinsic (vaddq_f32, vfmaq_f32, etc.).
    NeonEmitter.cpp reads this to generate the arm_neon.h header.
    Source: clang/include/clang/Basic/arm_neon.td

  arm_fp16.td
    TableGen source for FP16 NEON intrinsics.
    Source: clang/include/clang/Basic/arm_fp16.td

  arm_sve.td
    TableGen source for SVE (Scalable Vector Extension) intrinsics.
    Source: clang/include/clang/Basic/arm_sve.td

  NeonEmitter.cpp
    The TableGen backend that reads arm_neon.td and emits the arm_neon.h header.
    Source: clang/utils/TableGen/NeonEmitter.cpp

  BigEndianNEON.rst
    LLVM documentation on using NEON in big-endian mode.
    Source: llvm/docs/BigEndianNEON.rst

ARM official documentation:

  arm_neon_intrinsics_acle.html
    Complete ARM ACLE (Arm C Language Extensions) NEON intrinsics reference.
    Every intrinsic with signatures, descriptions, and instruction mappings.
    Source: https://arm-software.github.io/acle/neon_intrinsics/advsimd.html

  arm_neon_intrinsics_acle.md
    Same as above, converted to GitHub-flavored markdown via pandoc.

Generated:

  neon_intrinsics_list.txt
    Exhaustive list of all NEON intrinsics extracted from arm_neon.td + arm_fp16.td.
    425 base names organized by category, plus 3660 fully expanded type variants.
    Generated via Python script parsing the TableGen definitions.
