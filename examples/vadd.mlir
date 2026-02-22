builtin.module {
  func.func @vadd(%0 : i64, %1 : !llvm.ptr, %2 : !llvm.ptr, %3 : !llvm.ptr) {
    %4 = arith.constant 0 : i64
    %5 = arith.constant 1 : i64
    cf.br ^0(%4 : i64)
  ^0(%6 : i64):
    %7 = arith.cmpi slt, %6, %0 : i64
    cf.cond_br %7, ^1, ^2
  ^1:
    %8 = "llvm.getelementptr"(%1, %6) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %9 = "llvm.load"(%8) : (!llvm.ptr) -> f32
    %10 = "llvm.getelementptr"(%2, %6) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32}> : (!llvm.ptr, i64) -> !llvm.ptr
    %11 = "llvm.load"(%10) : (!llvm.ptr) -> f32
    %12 = arith.addf %9, %11 : f32
    %13 = "llvm.getelementptr"(%3, %6) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32}> : (!llvm.ptr, i64) -> !llvm.ptr
    "llvm.store"(%12, %13) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
    %14 = arith.addi %6, %5 : i64
    cf.br ^0(%14 : i64)
  ^2:
    func.return
  }
  "llvm.func"() <{sym_name = "malloc", function_type = !llvm.func<!llvm.ptr (i64)>, CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<"external">, visibility_ = 0 : i64}> ({
  }) : () -> ()
  "llvm.func"() <{sym_name = "free", function_type = !llvm.func<void (!llvm.ptr)>, CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<"external">, visibility_ = 0 : i64}> ({
  }) : () -> ()
}