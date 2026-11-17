# /// script
# requires-python = "==3.14.*"
# dependencies = [
#     "xdsl==0.56.0",
#     "lark==1.3.1",
# ]
# ///


import argparse
from functools import lru_cache
from pathlib import Path

from lark import Lark, Tree
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm, scf
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, FunctionType, IntegerAttr, ModuleOp, StringAttr
from xdsl.dialects.builtin import SymbolRefAttr as SymbolAttr
from xdsl.dialects.builtin import f64, i8, i32
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import dce

#
# parsing + ir gen
#


GRAMMAR = r"""
start: top_level*
?top_level: defun | expr
defun: "(" "defun" IDENTIFIER "(" args? ")" expr* ")"
args: IDENTIFIER+
?expr: print_expr | if_expr | binary_expr | call_expr | atom
print_expr: "(" "print" expr ")"
if_expr: "(" "if" expr expr expr ")"
binary_expr: "(" BINARY_OP expr expr ")"
call_expr: "(" IDENTIFIER expr* ")"
atom: NUMBER -> number | STRING -> string | IDENTIFIER -> variable
BINARY_OP: "+" | "-" | "*" | "/" | "%" | "<=" | ">=" | "==" | "!=" | "<" | ">"
COMMENT: /;[^\n]*/
%import common.SIGNED_NUMBER -> NUMBER
%import common.ESCAPED_STRING -> STRING
%import common.WS
%ignore WS
%ignore COMMENT
IDENTIFIER.-1: /[a-zA-Z0-9_+\-*\/%<>=!?]+/
"""


class IRGen:
    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))  # string builder
        self.symbol_table = {}  # arg name -> SSAValue in current scope (manually reset per function)
        self.str_cache = {}  # reference string consts on dup values
        self.str_cnt = 0  # unique id gen for const strings

        # declare printf signature from libc, so it can be called
        self.builder.insert(llvm.FuncOp("printf", llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True), linkage=llvm.LinkageAttr("external")))

    def gen(self, tree: Lark) -> ModuleOp:
        # generate functions
        function_defs = [node for node in tree.children if node.data == "defun"]
        inferred_arg_types = self._get_func_signatures()

        for func_def in function_defs:
            func_name = str(func_def.children[0])
            remaining_nodes = func_def.children[1:]

            args_node = next((node for node in remaining_nodes if node.data == "args"), None)
            arg_names = [str(token) for token in args_node.children] if args_node else []
            body_nodes = [node for node in remaining_nodes if node.data != "args"]

            # determine argument types (default to i32 if not inferred)
            arg_types = inferred_arg_types.get(func_name, [i32] * len(arg_names))
            func_type = FunctionType.from_lists(arg_types, [i32])  # assume i32 return

            # create mlir function operation
            entry_block = Block(arg_types=arg_types)
            func_op = func.FuncOp(func_name, func_type, Region(entry_block), visibility=StringAttr("private"))
            self.module.body.blocks[0].add_op(func_op)

            # enter function scope
            prev_builder = self.builder
            self.builder = Builder(InsertPoint.at_end(entry_block))
            self.symbol_table = {name: arg for name, arg in zip(arg_names, entry_block.args)}

            # generate function body + return instruction
            last_value = self._gen_expr(body_nodes[-1]) if body_nodes else None
            zero_constant = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
            return_value = last_value if last_value else zero_constant
            self.builder.insert(func.ReturnOp(return_value))
            func_op.function_type = FunctionType.from_lists(arg_types, [return_value.type])

            # exit function scope
            self.builder = prev_builder

        # create main function
        main_expressions = [node for node in tree.children if node.data != "defun"]
        if main_expressions:
            entry_block = Block()
            main_func = func.FuncOp("main", FunctionType.from_lists([], [i32]), Region(entry_block))
            self.module.body.blocks[0].add_op(main_func)

            prev_builder = self.builder
            self.builder = Builder(InsertPoint.at_end(entry_block))
            self.symbol_table = {}

            for expr in main_expressions:
                self._gen_expr(expr)

            zero_constant = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
            self.builder.insert(func.ReturnOp(zero_constant))
            self.builder = prev_builder

        return self.module

    def _get_func_signatures(self):
        # function name -> list of argument types from all call sites
        call_signatures = {}
        get_number_type = lambda node: f64 if "." in node.children[0] else i32
        get_type = lambda node: llvm.LLVMPointerType() if node.data == "string" else get_number_type(node) if node.data == "number" else None

        def visit_tree(node: Tree) -> None:
            if node.data == "call_expr":
                func_name = str(node.children[0])
                arg_types = [get_type(arg) for arg in node.children[1:]]
                if all(arg_types):
                    call_signatures.setdefault(func_name, []).append(arg_types)

            for child in node.children:
                if hasattr(child, "children"):
                    visit_tree(child)

        visit_tree(tree)

        # improve `call_signatures` by promoting types on mismatch
        resolved_signatures = {}
        for func_name, signatures in call_signatures.items():
            if not signatures:
                continue
            final_types = list(signatures[0])
            for signature in signatures[1:]:
                for i, (type1, type2) in enumerate(zip(final_types, signature)):
                    if type1 != type2:
                        final_types[i] = f64  # promote int to float on mismatch
            resolved_signatures[func_name] = final_types

        return resolved_signatures

    def _gen_expr(self, node):
        if node.data == "number":
            val = node.children[0]
            is_float = "." in val
            return self.builder.insert(arith.ConstantOp(FloatAttr(float(val), f64) if is_float else IntegerAttr(int(val), i32))).results[0]

        if node.data == "string":
            return self._get_str_global(str(node.children[0])[1:-1])

        if node.data == "variable":
            return self.symbol_table[str(node.children[0])]

        if node.data == "binary_expr":
            op = str(node.children[0])
            lhs_node, rhs_node = node.children[1], node.children[2]
            lhs, rhs = self._gen_expr(lhs_node), self._gen_expr(rhs_node)
            is_float = isinstance(lhs.type, builtin.Float64Type)
            match op:
                case "+":
                    cls = arith.AddfOp if is_float else arith.AddiOp
                case "-":
                    cls = arith.SubfOp if is_float else arith.SubiOp
                case "*":
                    cls = arith.MulfOp if is_float else arith.MuliOp
                case "<=":
                    if is_float:
                        return self.builder.insert(arith.CmpfOp(lhs, rhs, "ole")).results[0]
                    # lhs <= rhs equivalent !(rhs < lhs)
                    lt = self.builder.insert(arith.CmpiOp(rhs, lhs, "slt")).results[0]
                    z = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32) if isinstance(lt.type, builtin.IntegerType) and lt.type.width.data > 1 else IntegerAttr(0, builtin.i1))).results[0]
                    return self.builder.insert(arith.CmpiOp(lt, z, "eq")).results[0]
            return self.builder.insert(cls(lhs, rhs)).results[0]

        if node.data == "call_expr":
            name = str(node.children[0])
            call_args = [self._gen_expr(c) for c in node.children[1:]]
            func_op = next((o for o in self.module.body.blocks[0].ops if isinstance(o, func.FuncOp) and o.sym_name.data == name), None)
            # cast args on type mismatch with signature
            final_args = []
            for call_arg, actual_type in zip(call_args, func_op.function_type.inputs.data):
                if call_arg.type != actual_type:
                    final_args.append(self.builder.insert(arith.SIToFPOp(call_arg, f64)).results[0])
                else:
                    final_args.append(call_arg)
            return self.builder.insert(func.CallOp(name, final_args, func_op.function_type.outputs.data)).results[0]

        if node.data == "if_expr":
            cond = self._gen_expr(node.children[0])
            is_bool = cond.type == builtin.i1
            if not is_bool:
                z = self.builder.insert(arith.ConstantOp(IntegerAttr(0, cond.type))).results[0]
                cond = self.builder.insert(arith.CmpiOp(cond, z, "ne")).results[0]

            orig_builder = self.builder

            then_block = Block()
            self.builder = Builder(InsertPoint.at_end(then_block))
            then_val = self._gen_expr(node.children[1])
            self.builder.insert(scf.YieldOp(then_val))

            else_block = Block()
            self.builder = Builder(InsertPoint.at_end(else_block))
            else_val = self._gen_expr(node.children[2])
            self.builder.insert(scf.YieldOp(else_val))

            self.builder = orig_builder
            return self.builder.insert(scf.IfOp(cond, [then_val.type], Region(then_block), Region(else_block))).results[0]

        if node.data == "print_expr":
            val = self._gen_expr(node.children[0])
            fmt = "%s\n" if isinstance(val.type, llvm.LLVMPointerType) else "%f\n" if isinstance(val.type, builtin.Float64Type) else "%d\n"

            # casts for printf
            if fmt == "%f\n" and val.type != f64:
                val = self.builder.insert(arith.ExtFOp(val, f64)).results[0]
            if fmt == "%d\n":  # doesn't handle i64, defaults to i32
                if val.type.width.data < 32:
                    val = self.builder.insert(arith.ExtSIOp(val, i32)).results[0]
                elif val.type.width.data > 32:
                    val = self.builder.insert(arith.TruncIOp(val, i32)).results[0]

            fmt_ptr = self._get_str_global(fmt)
            call = llvm.CallOp(SymbolAttr("printf"), fmt_ptr, val, return_type=i32)
            call.attributes["var_callee_type"] = llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True)
            self.builder.insert(call)
            return None

    def _get_str_global(self, val: str) -> builtin.SSAValue:
        # create global string constant if not cached
        if val not in self.str_cache:
            global_name = f".str.{self.str_cnt}"
            self.str_cnt += 1
            self.str_cache[val] = global_name
            string_data = val.encode("utf-8") + b"\0"
            array_type = llvm.LLVMArrayType.from_size_and_type(len(string_data), i8)
            array_value = ArrayAttr([IntegerAttr(byte, i8) for byte in string_data])

            global_op = llvm.GlobalOp(array_type, StringAttr(global_name), linkage=llvm.LinkageAttr("internal"), constant=True, value=array_value)
            self.module.body.blocks[0].insert_op_before(global_op, self.module.body.blocks[0].first_op)

        # return address of cached global
        global_name = self.str_cache[val]
        return self.builder.insert(llvm.AddressOfOp(global_name, llvm.LLVMPointerType())).results[0]


#
# rewrites
#


class InlineFunctions(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriteWalker):
        callee = self._callee(op)
        if not callee:
            return
        callee_name = callee.sym_name.data
        is_recursive = any(isinstance(child, func.CallOp) and child.callee.string_value() == callee_name for child in callee.walk())
        if is_recursive:
            return
        is_single_line = len(list(callee.body.blocks[0].ops)) == 2  # one op + return
        if not is_single_line:
            return

        # replace func args with SSAValues from caller
        cloned_block = callee.clone().body.blocks[0]
        for operand, arg in zip(op.operands, cloned_block.args):
            arg.replace_by(operand)

        # inline all ops
        operations = list(cloned_block.ops)
        return_op = operations[-1]
        for operation in operations[:-1]:
            operation.detach()
            rewriter.insert_op(operation, InsertPoint.before(op))
        rewriter.replace_op(op, [], return_op.operands)

    @lru_cache(None)
    def _callee(self, op: func.CallOp) -> func.FuncOp | None:
        module = op
        while not isinstance(module, ModuleOp):
            module = module.parent_op()
        callee_name = op.callee.string_value()
        return next((func_op for func_op in module.body.blocks[0].ops if isinstance(func_op, func.FuncOp) and func_op.sym_name.data == callee_name), None)


class RemoveUnusedPrivateFunctions(RewritePattern):
    _used = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriteWalker):
        is_main = op.sym_name.data == "main"
        is_public = op.sym_visibility != StringAttr("private")
        if is_main or is_public:
            return

        # update set of functions that ever get called
        if self._used is None:
            self._used = {call.callee.string_value() for call in op.parent_op().walk() if isinstance(call, func.CallOp)}

        # remove unused functions
        if op.sym_name.data not in self._used:
            rewriter.erase_op(op)


class OptimizePass(ModulePass):
    name = "optimize"

    def apply(self, _, op: ModuleOp):
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        CanonicalizePass().apply(_, op)
        PatternRewriteWalker(RemoveUnusedPrivateFunctions()).rewrite_module(op)
        dce(op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tree = Lark(GRAMMAR, start="start").parse(Path(args.file).read_text())
    module_op = IRGen().gen(tree)
    orig_module_op = module_op.clone()

    ctx = Context()
    for d in [arith.Arith, builtin.Builtin, func.Func, scf.Scf, llvm.LLVM]:
        ctx.load_dialect(d)
    OptimizePass().apply(ctx, module_op)
    module_op.verify()

    if args.debug:
        print(f"\n{'-'*80} source\n{Path(args.file).read_text()}")
        print(f"\n{'-'*80} mlir before optimization\n{orig_module_op}")
        print(f"\n{'-'*80} mlir after optimization\n")
    print(module_op)
