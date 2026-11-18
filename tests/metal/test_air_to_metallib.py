import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import backend_metal

TOOLCHAIN_ENV = {"DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer", "MACOSX_DEPLOYMENT_TARGET": "15.0"}


class TestAirToMetallib(unittest.TestCase):
    def setUp(self):
        # mocked so tests run off macos
        for target, value in (("backend_metal.shutil.which", "/usr/bin/xcrun"), ("backend_metal.metal_toolchain_env", TOOLCHAIN_ENV)):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @patch("backend_metal.subprocess.run")
    @patch("backend_metal.Path")
    @patch("backend_metal.tempfile.NamedTemporaryFile")
    def test_compile_to_metallib_success(self, mock_tempfile, mock_path, mock_subprocess):
        mock_f_ll = MagicMock()
        mock_f_ll.name = "/tmp/test.ll"
        mock_f_air = MagicMock()
        mock_f_air.name = "/tmp/test.air"
        mock_f_lib = MagicMock()
        mock_f_lib.name = "/tmp/test.metallib"

        mock_f_ll.__enter__.return_value = mock_f_ll
        mock_f_air.__enter__.return_value = mock_f_air
        mock_f_lib.__enter__.return_value = mock_f_lib

        mock_tempfile.side_effect = [mock_f_ll, mock_f_air, mock_f_lib]

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        mock_path_obj = MagicMock()
        mock_path_obj.read_bytes.return_value = b"fake_metallib_content"
        mock_path.return_value = mock_path_obj

        result = backend_metal.compile_to_metallib("fake_ir")

        self.assertEqual(result, b"fake_metallib_content")
        mock_f_ll.write.assert_called_with(b"fake_ir")
        # 2 tool checks, compile, link
        self.assertEqual(mock_subprocess.call_count, 4)
        cmds = [call.args[0] for call in mock_subprocess.call_args_list]
        self.assertEqual(cmds[0], ["xcrun", "-sdk", "macosx", "metal", "--version"])
        self.assertEqual(cmds[1], ["xcrun", "-sdk", "macosx", "metallib", "--version"])
        self.assertEqual(cmds[2], ["xcrun", "-sdk", "macosx", "metal", "-x", "ir", "-c", "/tmp/test.ll", "-o", "/tmp/test.air"])
        self.assertEqual(cmds[3], ["xcrun", "-sdk", "macosx", "metallib", "/tmp/test.air", "-o", "/tmp/test.metallib"])
        mock_path.assert_called_with("/tmp/test.metallib")
        # toolchain env, no shell
        self.assertTrue(all(call.kwargs["env"] == TOOLCHAIN_ENV for call in mock_subprocess.call_args_list))
        self.assertTrue(all(not call.kwargs.get("shell", False) for call in mock_subprocess.call_args_list))
        self.assertTrue(all(call.kwargs.get("check") is False for call in mock_subprocess.call_args_list))

    @patch("backend_metal.subprocess.run")
    @patch("backend_metal.tempfile.NamedTemporaryFile")
    def test_compile_to_metallib_failure(self, mock_tempfile, mock_subprocess):
        mock_f_ll = MagicMock()
        mock_f_ll.name = "/tmp/test.ll"
        mock_f_air = MagicMock()
        mock_f_air.name = "/tmp/test.air"
        mock_f_lib = MagicMock()
        mock_f_lib.name = "/tmp/test.metallib"

        mock_f_ll.__enter__.return_value = mock_f_ll
        mock_f_air.__enter__.return_value = mock_f_air
        mock_f_lib.__enter__.return_value = mock_f_lib

        mock_tempfile.side_effect = [mock_f_ll, mock_f_air, mock_f_lib]

        # metal check fails
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "some error"
        mock_result.stderr = "some stderr"
        mock_subprocess.return_value = mock_result

        with self.assertRaises(AssertionError) as cm:
            backend_metal.compile_to_metallib("fake_ir")

        self.assertIn("metal not found", str(cm.exception))
        # stops before build
        self.assertEqual(mock_subprocess.call_count, 1)
        self.assertEqual(mock_subprocess.call_args.args[0], ["xcrun", "-sdk", "macosx", "metal", "--version"])

    @patch("backend_metal.subprocess.run")
    @patch("backend_metal.Path")
    @patch("backend_metal.tempfile.NamedTemporaryFile")
    def test_compile_to_metallib_empty_output(self, mock_tempfile, mock_path, mock_subprocess):
        mock_f_ll = MagicMock()
        mock_f_ll.name = "/tmp/test.ll"
        mock_f_air = MagicMock()
        mock_f_air.name = "/tmp/test.air"
        mock_f_lib = MagicMock()
        mock_f_lib.name = "/tmp/test.metallib"

        mock_f_ll.__enter__.return_value = mock_f_ll
        mock_f_air.__enter__.return_value = mock_f_air
        mock_f_lib.__enter__.return_value = mock_f_lib

        mock_tempfile.side_effect = [mock_f_ll, mock_f_air, mock_f_lib]

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        mock_path_obj = MagicMock()
        mock_path_obj.read_bytes.return_value = b""
        mock_path.return_value = mock_path_obj

        with self.assertRaises(AssertionError) as cm:
            backend_metal.compile_to_metallib("fake_ir")

        self.assertIn("generated metallib is empty", str(cm.exception))
        # both steps ran first
        self.assertEqual(mock_subprocess.call_count, 4)
        self.assertEqual(mock_subprocess.call_args.args[0], ["xcrun", "-sdk", "macosx", "metallib", "/tmp/test.air", "-o", "/tmp/test.metallib"])

    def _build_in_dir(self, tmpdir, calls, fail_compile=False):
        # real temp files, fake xcrun
        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[-1] == "--version":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if cmd[3] == "metal":
                self.assertEqual(Path(cmd[7]).read_text(), "fake_ir")
                if fail_compile:
                    return SimpleNamespace(returncode=1, stdout="compile out", stderr="compile err")
                Path(cmd[9]).write_bytes(b"air")
            else:
                self.assertEqual(Path(cmd[4]).read_bytes(), b"air")
                Path(cmd[6]).write_bytes(b"lib")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(backend_metal.tempfile, "tempdir", tmpdir), patch("backend_metal.subprocess.run", side_effect=fake_run):
            return backend_metal.compile_to_metallib("fake_ir")

    def test_compile_to_metallib_paths_with_spaces_stay_single_argv_items(self):
        with tempfile.TemporaryDirectory(prefix="exo jit; dir ") as tmpdir:
            calls = []
            result = self._build_in_dir(tmpdir, calls)
            self.assertEqual(result, b"lib")
            builds = [cmd for cmd, _ in calls if cmd[-1] != "--version"]
            self.assertEqual(len(builds), 2)
            ll, air, lib = builds[0][7], builds[0][9], builds[1][6]
            self.assertEqual(builds[0], ["xcrun", "-sdk", "macosx", "metal", "-x", "ir", "-c", ll, "-o", air])
            self.assertEqual(builds[1], ["xcrun", "-sdk", "macosx", "metallib", air, "-o", lib])
            for path, suffix in ((ll, ".ll"), (air, ".air"), (lib, ".metallib")):
                self.assertTrue(path.startswith(tmpdir + os.sep) and path.endswith(suffix), path)
                self.assertIn(" ", path)
                self.assertIn(";", path)
            for cmd, kwargs in calls:
                self.assertIsInstance(cmd, list)
                self.assertFalse(kwargs.get("shell", False))
                self.assertEqual(kwargs["env"], TOOLCHAIN_ENV)
            self.assertEqual(os.listdir(tmpdir), [])

    def test_compile_to_metallib_failed_compile_never_links(self):
        with tempfile.TemporaryDirectory(prefix="exo jit; dir ") as tmpdir:
            calls = []
            with self.assertRaises(AssertionError) as cm:
                self._build_in_dir(tmpdir, calls, fail_compile=True)
            error = cm.exception
            self.assertIn("metal compilation failed", str(error))
            self.assertIn("compile out", str(error))
            self.assertIn("compile err", str(error))
            builds = [cmd for cmd, _ in calls if cmd[-1] != "--version"]
            self.assertEqual(len(builds), 1)
            self.assertEqual(builds[0][3], "metal")
            self.assertEqual(os.listdir(tmpdir), [])

    @patch("backend_metal.ctypes")
    def test_create_compute_pipeline_success(self, mock_ctypes):
        mock_metal = MagicMock()
        mock_objc = MagicMock()

        mock_device = MagicMock()
        mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_device

        mock_libdispatch = MagicMock()
        mock_ctypes.CDLL.return_value = mock_libdispatch

        mock_dispatch_data = MagicMock()
        mock_libdispatch.dispatch_data_create.return_value = mock_dispatch_data

        mock_ns_dispatch = MagicMock()
        mock_objc.objc_object.return_value = mock_ns_dispatch

        mock_library = MagicMock()
        mock_device.newLibraryWithData_error_.return_value = (mock_library, None)

        mock_fn = MagicMock()
        mock_library.newFunctionWithName_.return_value = mock_fn

        mock_pso = MagicMock()
        mock_device.newComputePipelineStateWithFunction_error_.return_value = (mock_pso, None)

        with patch.dict(sys.modules, {"Metal": mock_metal, "objc": mock_objc}):
            device, pso = backend_metal.create_compute_pipeline(b"binary", "kernel_name")

        self.assertEqual(device, mock_device)
        self.assertEqual(pso, mock_pso)

        mock_ctypes.create_string_buffer.assert_called()
        mock_libdispatch.dispatch_data_create.assert_called()
        mock_objc.objc_object.assert_called_with(c_void_p=mock_dispatch_data)

        mock_device.newLibraryWithData_error_.assert_called_with(mock_ns_dispatch, None)
        mock_library.newFunctionWithName_.assert_called_with("kernel_name")
        mock_device.newComputePipelineStateWithFunction_error_.assert_called_with(mock_fn, None)

    def test_create_compute_pipeline_no_device(self):
        mock_metal = MagicMock()
        mock_metal.MTLCreateSystemDefaultDevice.return_value = None
        with patch.dict(sys.modules, {"Metal": mock_metal}), self.assertRaises(AssertionError) as cm:
            backend_metal.create_compute_pipeline(b"bin", "name")
        self.assertIn("metal not supported", str(cm.exception))

    @patch("backend_metal.ctypes")
    def test_create_compute_pipeline_library_error(self, mock_ctypes):
        mock_metal = MagicMock()
        mock_objc = MagicMock()
        mock_device = MagicMock()
        mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_device

        mock_libdispatch = MagicMock()
        mock_ctypes.CDLL.return_value = mock_libdispatch
        mock_libdispatch.dispatch_data_create.return_value = MagicMock()
        mock_objc.objc_object.return_value = MagicMock()

        mock_device.newLibraryWithData_error_.return_value = (None, "Some Error")

        with patch.dict(sys.modules, {"Metal": mock_metal, "objc": mock_objc}), self.assertRaises(AssertionError) as cm:
            backend_metal.create_compute_pipeline(b"bin", "name")
        self.assertIn("error loading library", str(cm.exception))

    @patch("backend_metal.ctypes")
    def test_create_compute_pipeline_function_not_found(self, mock_ctypes):
        mock_metal = MagicMock()
        mock_objc = MagicMock()
        mock_device = MagicMock()
        mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_device

        mock_libdispatch = MagicMock()
        mock_ctypes.CDLL.return_value = mock_libdispatch
        mock_libdispatch.dispatch_data_create.return_value = MagicMock()
        mock_objc.objc_object.return_value = MagicMock()

        mock_library = MagicMock()
        mock_device.newLibraryWithData_error_.return_value = (mock_library, None)

        mock_library.newFunctionWithName_.return_value = None

        with patch.dict(sys.modules, {"Metal": mock_metal, "objc": mock_objc}), self.assertRaises(AssertionError) as cm:
            backend_metal.create_compute_pipeline(b"bin", "kernel_name")
        self.assertIn("function 'kernel_name' not found", str(cm.exception))

    def test_execute_kernel(self):
        mock_device = MagicMock()
        mock_pso = MagicMock()
        mock_grid_size = MagicMock()
        mock_threadgroup_size = MagicMock()
        mock_encode_args_fn = MagicMock()

        mock_queue = MagicMock()
        mock_device.newCommandQueue.return_value = mock_queue

        mock_buffer = MagicMock()
        mock_queue.commandBuffer.return_value = mock_buffer

        mock_encoder = MagicMock()
        mock_buffer.computeCommandEncoder.return_value = mock_encoder

        mock_buffer.status.return_value = 0

        mock_metal = MagicMock()
        with patch.dict(sys.modules, {"Metal": mock_metal}):
            mock_metal.MTLCommandBufferStatusCompleted = 1
            mock_buffer.status.return_value = 1

            backend_metal.execute_kernel(mock_device, mock_pso, mock_grid_size, mock_threadgroup_size, mock_encode_args_fn)

            mock_device.newCommandQueue.assert_called()
            mock_queue.commandBuffer.assert_called()
            mock_buffer.computeCommandEncoder.assert_called()
            mock_encoder.setComputePipelineState_.assert_called_with(mock_pso)
            mock_encode_args_fn.assert_called_with(mock_encoder)
            mock_encoder.dispatchThreads_threadsPerThreadgroup_.assert_called_with(mock_grid_size, mock_threadgroup_size)
            mock_encoder.endEncoding.assert_called()
            mock_buffer.commit.assert_called()
            mock_buffer.waitUntilCompleted.assert_called()

    def test_execute_kernel_failure(self):
        mock_device = MagicMock()
        mock_pso = MagicMock()
        mock_grid_size = MagicMock()
        mock_threadgroup_size = MagicMock()
        mock_encode_args_fn = MagicMock()

        mock_queue = MagicMock()
        mock_device.newCommandQueue.return_value = mock_queue

        mock_buffer = MagicMock()
        mock_queue.commandBuffer.return_value = mock_buffer

        mock_encoder = MagicMock()
        mock_buffer.computeCommandEncoder.return_value = mock_encoder

        mock_metal = MagicMock()
        with patch.dict(sys.modules, {"Metal": mock_metal}):
            mock_metal.MTLCommandBufferStatusCompleted = 1
            mock_buffer.status.return_value = 2
            mock_buffer.error.return_value = "GPU Fault"

            with self.assertRaises(AssertionError) as cm:
                backend_metal.execute_kernel(mock_device, mock_pso, mock_grid_size, mock_threadgroup_size, mock_encode_args_fn)

            self.assertIn("command buffer failed with status 2", str(cm.exception))
            self.assertIn("GPU Fault", str(cm.exception))
