from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects import memref
from xdsl.dialects.builtin import DenseArrayBase, MemRefType, i64
from xdsl.dialects.utils import split_dynamic_index_list
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import AnyAttr, Attribute, AttrSizedOperandSegments, IRDLOperation, ParsePropInAttrDict, VarConstraint, irdl_op_definition, operand_def, prop_def, result_def, var_operand_def


@irdl_op_definition
class WindowOp(IRDLOperation):
    T: ClassVar = VarConstraint("T", AnyAttr())

    name = "exo.window"

    input = operand_def(
        MemRefType.constr(element_type=AnyAttr()),
    )
    indices = var_operand_def()
    input_sizes = var_operand_def()
    static_input_sizes = prop_def(DenseArrayBase)
    output_sizes = var_operand_def()
    static_output_sizes = prop_def(DenseArrayBase)

    result = result_def(
        MemRefType.constr(element_type=T),
    )

    irdl_options = (AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict())

    def __init__(
        self,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        input_sizes: Sequence[SSAValue | Operation | int],
        output_sizes: Sequence[SSAValue[Attribute] | int],
        result_type: MemRefType,
    ) -> None:
        static_input_sizes, dyn_input_sizes = split_dynamic_index_list(input_sizes, memref.DYNAMIC_INDEX)
        static_output_sizes, dyn_output_sizes = split_dynamic_index_list(output_sizes, memref.DYNAMIC_INDEX)

        super().__init__(
            operands=[SSAValue.get(input), indices, dyn_input_sizes, dyn_output_sizes],
            result_types=[result_type],
            properties={
                "static_input_sizes": DenseArrayBase.create_dense_int(i64, static_input_sizes),
                "static_output_sizes": DenseArrayBase.create_dense_int(i64, static_output_sizes),
            },
        )


Exo = Dialect(
    "exo",
    [
        WindowOp,
    ],
    [],
)
