import unittest

import backend_metal


class TestLlvmToAirUnit(unittest.TestCase):
    def test_get_type_info(self):
        # scalars
        self.assertEqual(backend_metal.get_type_info("i32"), ("int", 4, 4))
        self.assertEqual(backend_metal.get_type_info("float"), ("float", 4, 4))
        self.assertEqual(backend_metal.get_type_info("double"), ("double", 8, 8))
        self.assertEqual(backend_metal.get_type_info("i64"), ("long", 8, 8))
        self.assertEqual(backend_metal.get_type_info("i16"), ("short", 2, 2))
        self.assertEqual(backend_metal.get_type_info("i8"), ("char", 1, 1))
        self.assertEqual(backend_metal.get_type_info("ptr"), ("float", 4, 4))

        # vectors
        self.assertEqual(backend_metal.get_type_info("<4 x float>"), ("float", 16, 16))
        self.assertEqual(backend_metal.get_type_info("<2 x i32>"), ("int", 8, 8))

        # spaced vectors
        self.assertEqual(backend_metal.get_type_info("< 4 x float >"), ("float", 16, 16))

        self.assertIsNone(backend_metal.get_type_info("unknown_type"))

    def test_intrinsic_handler_handle_type_casts(self):
        used_intrinsics = set()

        # uitofp
        line = "%res = uitofp i32 %val to float"
        out = backend_metal.IntrinsicHandler.handle_type_casts(line, used_intrinsics)
        self.assertIn("@air.convert.f.f32.u.i32", out)
        self.assertIn("@air.convert.f.f32.u.i32", used_intrinsics)

        # sitofp
        line = "%res = sitofp i32 %val to float"
        out = backend_metal.IntrinsicHandler.handle_type_casts(line, used_intrinsics)
        self.assertIn("@air.convert.f.f32.s.i32", out)

        # fptoui
        line = "%res = fptoui float %val to i32"
        out = backend_metal.IntrinsicHandler.handle_type_casts(line, used_intrinsics)
        self.assertIn("@air.convert.u.i32.f.f32", out)

        # no match
        line = "%res = add i32 %a, %b"
        out = backend_metal.IntrinsicHandler.handle_type_casts(line, used_intrinsics)
        self.assertEqual(out, line)

    def test_intrinsic_handler_replace_intrinsics(self):
        used_intrinsics = set()

        line = "call float @llvm.sin.f32(float %v)"
        out = backend_metal.IntrinsicHandler.replace_intrinsics(line, used_intrinsics)
        self.assertIn("@air.sin.f32", out)
        self.assertIn("@air.sin.f32", used_intrinsics)

        line = "call float @llvm.minnum.f32(float %a, float %b)"
        out = backend_metal.IntrinsicHandler.replace_intrinsics(line, used_intrinsics)
        self.assertIn("@air.fmin.f32", out)
        self.assertIn("@air.fmin.f32", used_intrinsics)

        line = "call void @other_func()"
        out = backend_metal.IntrinsicHandler.replace_intrinsics(line, used_intrinsics)
        self.assertEqual(out, line)

    def test_signature_parser_parse(self):
        lines = ["define void @my_kernel(float* %A, i32 %id) {", "  ret void", "}"]

        func_name, args_list, air_sig, _next_idx, var_addrspaces, _scalar_loads = backend_metal.SignatureParser.parse(lines, 0)

        self.assertEqual(func_name, "my_kernel")

        self.assertEqual(len(args_list), 2)

        self.assertIn("float addrspace(1)*", args_list[0][0])
        self.assertEqual(args_list[0][1], "A")
        self.assertFalse(args_list[0][2])

        self.assertEqual(args_list[1][0], "i32")
        self.assertEqual(args_list[1][1], "id")

        self.assertIn("define void @my_kernel", air_sig)
        self.assertIn("air-buffer-no-alias", air_sig)

        self.assertEqual(var_addrspaces["%A"], 1)
        self.assertEqual(var_addrspaces["%id"], 0)

    def test_signature_parser_overrides(self):
        lines = ["define void @my_kernel(float* %0, i32 %1) {"]
        overrides = {"my_kernel": {"0": "out", "1": "gid"}}
        _func_name, args_list, _air_sig, _, var_addrspaces, _ = backend_metal.SignatureParser.parse(lines, 0, overrides)
        self.assertEqual(args_list[0][1], "out")
        self.assertTrue(args_list[0][2])
        self.assertEqual(args_list[1][1], "gid")
        self.assertEqual(var_addrspaces["%1"], 0)

    def test_signature_parser_multiline(self):
        lines = ["define void @long_sig(float* %a,", "                      i32 %b) {"]
        name, args_list, _air_sig, idx, _var_as, _scalar_loads = backend_metal.SignatureParser.parse(lines, 0)
        self.assertEqual(name, "long_sig")
        self.assertEqual(len(args_list), 2)
        self.assertEqual(idx, 2)

    def test_signature_parser_special_names_and_loads(self):
        lines = ["define void @spec(i32 %global_id, float* %result) {"]
        _name, args_list, _air_sig, _idx, var_as, _scalar_loads = backend_metal.SignatureParser.parse(lines, 0)

        self.assertEqual(args_list[0][1], "global_id")
        self.assertFalse(args_list[0][2])
        self.assertEqual(var_as["%global_id"], 0)

        self.assertEqual(args_list[1][1], "result")
        self.assertTrue(args_list[1][2])
        self.assertEqual(var_as["%result"], 1)

    def test_metadata_generator_emit(self):
        kernels = [("my_kernel", [("float addrspace(1)*", "%A", False), ("i32", "gid", False)])]
        used_intrinsics = {"@air.sin.f32"}

        lines = backend_metal.MetadataGenerator.emit(kernels, used_intrinsics)

        full_text = "\n".join(lines)

        self.assertIn("declare float @air.sin.f32(float) #2", full_text)

        self.assertIn("!air.kernel", full_text)
        self.assertIn("my_kernel", full_text)

        self.assertIn("air.buffer", full_text)
        self.assertIn("air.thread_position_in_grid", full_text)


if __name__ == "__main__":
    unittest.main()
