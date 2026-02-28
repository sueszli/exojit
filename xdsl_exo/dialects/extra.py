# custom dialect definitions for ops that xdsl doesn't provide upstream

from typing import ClassVar

from xdsl.dialects.builtin import I1, AnyFloatConstr, AnySignlessIntegerOrIndexType, IndexType, IntegerAttr, IntegerType, VectorType, i32
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import Attribute, IRDLOperation, ParsePropInAttrDict, VarConstraint, irdl_op_definition, operand_def, prop_def, result_def


@irdl_op_definition
class CastsOp(IRDLOperation):
    name = "index.casts"

    input = operand_def(AnySignlessIntegerOrIndexType)
    result = result_def(AnySignlessIntegerOrIndexType)

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    def __init__(self, input: SSAValue | Operation, result_type: Attribute) -> None:
        super().__init__(
            operands=[input],
            result_types=[result_type],
        )

    def verify_(self):
        if isinstance(self.input.type, IndexType):
            assert isinstance(self.result.type, IntegerType) and not isinstance(self.result.type, IndexType), "result type must be integer for index type input"

        elif isinstance(self.input.type, IntegerType):
            assert isinstance(self.result.type, IndexType), "result type must be index type for integer type input"


Index = Dialect(
    "index",
    [CastsOp],
    [],
)


@irdl_op_definition
class FAbsOp(IRDLOperation):
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
    name = "llvm.intr.masked.store"

    value = operand_def(AnyFloatConstr | VectorType.constr(AnyFloatConstr))
    data = operand_def(LLVMPointerType)
    mask = operand_def(I1 | VectorType[I1])
    alignment = prop_def(IntegerAttr[i32])

    assembly_format = "$value `,` $data `,` $mask attr-dict `:` type($value) `,` type($mask) `into` type($data)"

    irdl_options = (ParsePropInAttrDict(),)

    def __init__(
        self,
        value: Operation | SSAValue,
        data: Operation | SSAValue,
        mask: Operation | SSAValue,
        alignment: int = 32,
    ):
        super().__init__(
            operands=[value, data, mask],
            result_types=[],
            properties={
                "alignment": IntegerAttr.from_int_and_width(alignment, 32),
            },
        )


LLVMIntrinsics = Dialect(
    "llvm.intr",
    [FAbsOp, MaskedStoreOp],
    [],
)
