import unittest

from llvm_to_air import to_air


class TestOpaquePointers(unittest.TestCase):
    def test_opaque_pointers_replacement(self):
        """
        Test that opaque pointers 'ptr' are replaced by 'float addrspace(N)*'
        based on inferred address space or default.
        """
        llvm_ir = """
define void @opaque_ptr_kernel(ptr %a, ptr %b) {
  %val = load float, ptr %a
  store float %val, ptr %b
  ret void
}
"""
        air = to_air(llvm_ir)

        self.assertIn("float addrspace(1)* %a", air)
        self.assertIn("float addrspace(1)* %b", air)

        self.assertIn("load float, float addrspace(1)* %a", air)

        self.assertIn("store float %val, float addrspace(1)* %b", air)

    def test_opaque_pointers_addrspace_propagation(self):
        """
        Test that address spaces propagated from getelementptr are used for rewriting 'ptr'.
        """
        llvm_ir = """
define void @propagate(ptr %a) {
  %p = getelementptr float, ptr addrspace(3) %a, i32 0
  %v = load float, ptr %p
  ret void
}
"""
        air = to_air(llvm_ir)
        self.assertIn("float addrspace(3)* %p", air)


if __name__ == "__main__":
    unittest.main()
