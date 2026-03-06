from typing import ClassVar

from xdsl.dialects.builtin import I1, AnyFloatConstr, IntegerAttr, VectorType, i32
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import IRDLOperation, ParsePropInAttrDict, VarConstraint, irdl_op_definition, operand_def, prop_def, result_def

from xdsl_exo.patches_llvm import FCmpOp, SelectOp


@irdl_op_definition
class FAbsOp(IRDLOperation):
    # https://github.com/xdslproject/xdsl/commit/5f30cfdd78d8dbaddb70b15358c406ab63524b5b
    T: ClassVar = VarConstraint("T", AnyFloatConstr | VectorType.constr(AnyFloatConstr))

    name = "llvm.intr.fabs"

    input = operand_def(T)
    result = result_def(T)

    assembly_format = "`(` operands `)` attr-dict `:` functional-type(operands, results)"

    irdl_options = (ParsePropInAttrDict(),)

    def __init__(self, input: Operation | SSAValue, result_type: Attribute):
        super().__init__(operands=[input], result_types=[result_type])


@irdl_op_definition
class MaskedStoreOp(IRDLOperation):
    # https://github.com/xdslproject/xdsl/commit/726e2c40df108e700fc9eab071555adc4fff8b75
    name = "llvm.intr.masked.store"

    value = operand_def(AnyFloatConstr | VectorType.constr(AnyFloatConstr))
    data = operand_def(LLVMPointerType)
    mask = operand_def(I1 | VectorType[I1])
    alignment = prop_def(IntegerAttr[i32])

    assembly_format = "$value `,` $data `,` $mask attr-dict `:` type($value) `,` type($mask) `into` type($data)"

    irdl_options = (ParsePropInAttrDict(),)

    def __init__(self, value: Operation | SSAValue, data: Operation | SSAValue, mask: Operation | SSAValue, alignment: int = 32):
        super().__init__(operands=[value, data, mask], result_types=[], properties={"alignment": IntegerAttr(alignment, 32)})


LLVMIntrinsics = Dialect(
    "llvm.intr",
    [
        FAbsOp,
        FCmpOp,
        SelectOp,
        MaskedStoreOp,
    ],
    [],
)
