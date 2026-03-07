import llvmlite.binding as llvm_binding
import llvmlite.ir as ir
from xdsl.backend.llvm.convert_op import convert_op as _xdsl_convert_op
from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.llvm import FNegOp
from xdsl.ir import Block, Operation, SSAValue

from xdsl_exo.patches_llvm import BrOp, CondBrOp, FCmpOp, SelectOp

ValMap = dict[SSAValue, ir.Value]
BlockMap = dict[Block, ir.Block]
PhiMap = dict[SSAValue, ir.PhiInstr]


def _convert_op(op: Operation, builder: ir.IRBuilder, block_map: BlockMap, phi_map: PhiMap, val_map: ValMap) -> None:
    match op:
        case llvm.ConstantOp():
            val_map[op.result] = ir.Constant(convert_type(op.result.type), op.value.value.data)
        case FNegOp():
            val_map[op.res] = builder.fneg(val_map[op.arg])
        case FCmpOp():
            pred, is_ordered = {"oeq": ("==", True), "ogt": (">", True), "oge": (">=", True), "olt": ("<", True), "ole": ("<=", True), "one": ("!=", True), "ord": ("ord", True), "ueq": ("==", False), "ugt": (">", False), "uge": (">=", False), "ult": ("<", False), "ule": ("<=", False), "une": ("!=", False), "uno": ("uno", False)}[op.predicate.data]
            val_map[op.res] = (builder.fcmp_ordered if is_ordered else builder.fcmp_unordered)(pred, val_map[op.lhs], val_map[op.rhs])
        case SelectOp():
            val_map[op.res] = builder.select(val_map[op.cond], val_map[op.lhs], val_map[op.rhs])
        case BrOp():
            cur = builder.block
            for a, v in zip(op.successor.args, op.operands):
                if a in phi_map:
                    phi_map[a].add_incoming(val_map[v], cur)
            builder.branch(block_map[op.successor])
        case CondBrOp():
            cur = builder.block
            for a, v in zip(op.successors[0].args, op.then_arguments):
                if a in phi_map:
                    phi_map[a].add_incoming(val_map[v], cur)
            for a, v in zip(op.successors[1].args, op.else_arguments):
                if a in phi_map:
                    phi_map[a].add_incoming(val_map[v], cur)
            builder.cbranch(val_map[op.cond], block_map[op.successors[0]], block_map[op.successors[1]])
        case _:
            _xdsl_convert_op(op, builder, val_map)


def _emit_func(func_op: llvm.FuncOp, llvm_module: ir.Module) -> None:
    ir_func = llvm_module.get_global(func_op.sym_name.data)
    mlir_blocks = list(func_op.body.blocks)

    block_map: BlockMap = {block: ir_func.append_basic_block() for block in mlir_blocks}
    phi_map: PhiMap = {arg: ir.IRBuilder(block_map[blk]).phi(convert_type(arg.type)) for blk in mlir_blocks[1:] for arg in blk.args}
    val_map: ValMap = dict(zip(mlir_blocks[0].args, ir_func.args)) | phi_map

    for mlir_block in mlir_blocks:
        builder = ir.IRBuilder(block_map[mlir_block])
        for op in mlir_block.ops:
            _convert_op(op, builder, block_map, phi_map, val_map)


def jit_compile(module: ModuleOp) -> llvm_binding.ExecutionEngine:
    llvm_module = ir.Module()
    func_ops = list(module.ops)

    # forward-declare all functions so call sites can resolve them regardless of order
    for op in func_ops:
        assert isinstance(op, llvm.FuncOp)
        ftype = ir.FunctionType(convert_type(op.function_type.output), [convert_type(t) for t in op.function_type.inputs])
        ir.Function(llvm_module, ftype, name=op.sym_name.data)

    # emit bodies
    for op in func_ops:
        if op.body.blocks:
            _emit_func(op, llvm_module)

    llvm_binding.initialize_native_target()
    llvm_binding.initialize_native_asmprinter()
    llvm_mod = llvm_binding.parse_assembly(str(llvm_module))
    llvm_mod.verify()
    target_machine = llvm_binding.Target.from_default_triple().create_target_machine()
    engine = llvm_binding.create_mcjit_compiler(llvm_mod, target_machine)
    engine.finalize_object()
    engine.run_static_constructors()
    return engine
