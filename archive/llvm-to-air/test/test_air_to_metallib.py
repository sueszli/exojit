import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import llvm_to_air as air_to_metallib


class TestAirToMetallib(unittest.TestCase):

    @patch("llvm_to_air.subprocess.run")
    @patch("llvm_to_air.Path")
    @patch("llvm_to_air.tempfile.NamedTemporaryFile")
    def test_compile_to_metallib_success(self, mock_tempfile, mock_path, mock_subprocess):
        # setup mocks
        mock_f_ll = MagicMock()
        mock_f_ll.name = "/tmp/test.ll"
        mock_f_air = MagicMock()
        mock_f_air.name = "/tmp/test.air"
        mock_f_lib = MagicMock()
        mock_f_lib.name = "/tmp/test.metallib"

        # context managers for tempfiles
        mock_f_ll.__enter__.return_value = mock_f_ll
        mock_f_air.__enter__.return_value = mock_f_air
        mock_f_lib.__enter__.return_value = mock_f_lib

        mock_tempfile.side_effect = [mock_f_ll, mock_f_air, mock_f_lib]

        # subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        # path read_bytes
        mock_path_obj = MagicMock()
        mock_path_obj.read_bytes.return_value = b"fake_metallib_content"
        mock_path.return_value = mock_path_obj

        # call
        result = air_to_metallib.compile_to_metallib("fake_ir")

        # assertions
        self.assertEqual(result, b"fake_metallib_content")
        mock_f_ll.write.assert_called_with(b"fake_ir")
        # Now called 3 times: metal check, metallib check, compilation
        self.assertEqual(mock_subprocess.call_count, 3)
        # Check the compilation call (last one)
        cmd = mock_subprocess.call_args[0][0]
        self.assertIn("xcrun -sdk macosx metal", cmd)
        self.assertIn("xcrun -sdk macosx metallib", cmd)

    @patch("llvm_to_air.subprocess.run")
    @patch("llvm_to_air.tempfile.NamedTemporaryFile")
    def test_compile_to_metallib_failure(self, mock_tempfile, mock_subprocess):
        # setup mocks
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

        # subprocess result - first call (metal check) fails
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "some error"
        mock_result.stderr = "some stderr"
        mock_subprocess.return_value = mock_result

        # call and assert raises - should fail on metal check
        with self.assertRaises(AssertionError) as cm:
            air_to_metallib.compile_to_metallib("fake_ir")

        self.assertIn("metal not found", str(cm.exception))

    @patch("llvm_to_air.subprocess.run")
    @patch("llvm_to_air.Path")
    @patch("llvm_to_air.tempfile.NamedTemporaryFile")
    def test_compile_to_metallib_empty_output(self, mock_tempfile, mock_path, mock_subprocess):
        # setup mocks
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
        mock_path_obj.read_bytes.return_value = b""  # Empty
        mock_path.return_value = mock_path_obj

        # call and assert raises
        with self.assertRaises(AssertionError) as cm:
            air_to_metallib.compile_to_metallib("fake_ir")

        self.assertIn("generated metallib is empty", str(cm.exception))

    @patch("llvm_to_air.ctypes")
    def test_create_compute_pipeline_success(self, mock_ctypes):
        mock_metal = MagicMock()
        mock_objc = MagicMock()

        # mock metal device
        mock_device = MagicMock()
        mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_device

        # mock ctypes
        mock_libdispatch = MagicMock()
        mock_ctypes.CDLL.return_value = mock_libdispatch

        # mock dispatch_data_create
        mock_dispatch_data = MagicMock()
        mock_libdispatch.dispatch_data_create.return_value = mock_dispatch_data

        # mock objc bridge
        mock_ns_dispatch = MagicMock()
        mock_objc.objc_object.return_value = mock_ns_dispatch

        # mock library
        mock_library = MagicMock()
        mock_device.newLibraryWithData_error_.return_value = (mock_library, None)

        # mock function
        mock_fn = MagicMock()
        mock_library.newFunctionWithName_.return_value = mock_fn

        # mock PSO
        mock_pso = MagicMock()
        mock_device.newComputePipelineStateWithFunction_error_.return_value = (mock_pso, None)

        # call
        with patch.dict(sys.modules, {"Metal": mock_metal, "objc": mock_objc}):
            device, pso = air_to_metallib.create_compute_pipeline(b"binary", "kernel_name")

        # assertions
        self.assertEqual(device, mock_device)
        self.assertEqual(pso, mock_pso)

        # Verify ctypes calls
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
            air_to_metallib.create_compute_pipeline(b"bin", "name")
        self.assertIn("metal not supported", str(cm.exception))

    @patch("llvm_to_air.ctypes")
    def test_create_compute_pipeline_library_error(self, mock_ctypes):
        mock_metal = MagicMock()
        mock_objc = MagicMock()
        mock_device = MagicMock()
        mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_device

        # mock ctypes setup
        mock_libdispatch = MagicMock()
        mock_ctypes.CDLL.return_value = mock_libdispatch
        mock_libdispatch.dispatch_data_create.return_value = MagicMock()
        mock_objc.objc_object.return_value = MagicMock()

        mock_device.newLibraryWithData_error_.return_value = (None, "Some Error")

        with patch.dict(sys.modules, {"Metal": mock_metal, "objc": mock_objc}), self.assertRaises(AssertionError) as cm:
            air_to_metallib.create_compute_pipeline(b"bin", "name")
        self.assertIn("error loading library", str(cm.exception))

    @patch("llvm_to_air.ctypes")
    def test_create_compute_pipeline_function_not_found(self, mock_ctypes):
        mock_metal = MagicMock()
        mock_objc = MagicMock()
        mock_device = MagicMock()
        mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_device

        # mock ctypes setup
        mock_libdispatch = MagicMock()
        mock_ctypes.CDLL.return_value = mock_libdispatch
        mock_libdispatch.dispatch_data_create.return_value = MagicMock()
        mock_objc.objc_object.return_value = MagicMock()

        mock_library = MagicMock()
        mock_device.newLibraryWithData_error_.return_value = (mock_library, None)

        mock_library.newFunctionWithName_.return_value = None

        with patch.dict(sys.modules, {"Metal": mock_metal, "objc": mock_objc}), self.assertRaises(AssertionError) as cm:
            air_to_metallib.create_compute_pipeline(b"bin", "kernel_name")
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

            air_to_metallib.execute_kernel(mock_device, mock_pso, mock_grid_size, mock_threadgroup_size, mock_encode_args_fn)

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

        # simulate failure
        mock_metal = MagicMock()
        with patch.dict(sys.modules, {"Metal": mock_metal}):
            mock_metal.MTLCommandBufferStatusCompleted = 1
            mock_buffer.status.return_value = 2
            mock_buffer.error.return_value = "GPU Fault"

            with self.assertRaises(AssertionError) as cm:
                air_to_metallib.execute_kernel(mock_device, mock_pso, mock_grid_size, mock_threadgroup_size, mock_encode_args_fn)

            self.assertIn("command buffer failed with status 2", str(cm.exception))
            self.assertIn("GPU Fault", str(cm.exception))
