"""Causal ONNX controller validation/execution. This module contains no plasma physics."""
from __future__ import annotations

import time
import math
import os
import select
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf.message import DecodeError


class InvalidController(ValueError):
    pass


OPERATORS = frozenset({
    'Add', 'Sub', 'Mul', 'Div', 'Neg', 'Abs', 'Exp', 'Log', 'Sqrt', 'Pow',
    'Max', 'Min', 'Clip', 'Relu', 'Sigmoid', 'Tanh', 'Softplus', 'Softmax',
    'MatMul', 'Gemm', 'Identity', 'Concat', 'Split', 'Slice', 'Gather',
    'Reshape', 'Transpose', 'Unsqueeze', 'Squeeze', 'Cast', 'Constant',
    'ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin', 'Equal', 'Less',
    'Greater', 'LessOrEqual', 'GreaterOrEqual', 'Where', 'And', 'Or', 'Not',
})


def no_external_tensors(graph):
    if graph.sparse_initializer:
        raise InvalidController('Sparse initializer tensors are not supported')
    for tensor in list(graph.initializer) + list(graph.sparse_initializer):
        if hasattr(tensor, 'data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL:
            raise InvalidController('External ONNX tensors are forbidden')
        if getattr(tensor, 'external_data', None):
            raise InvalidController('External ONNX data are forbidden')
    for node in graph.node:
        if node.domain not in ('', 'ai.onnx') or node.op_type not in OPERATORS:
            raise InvalidController(f'Unsupported controller operator {node.domain}:{node.op_type}')
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise InvalidController('Nested/conditional/loop graphs are forbidden')
            for tensor in ([attr.t] if attr.type == onnx.AttributeProto.TENSOR else attr.tensors):
                if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
                    raise InvalidController('External constant tensors are forbidden')


def read_deadline(stream, size, deadline):
    result = bytearray()
    while len(result) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            raise InvalidController('Controller process exceeded its online time budget')
        block = os.read(stream.fileno(), size - len(result))
        if not block:
            raise InvalidController('Controller process terminated before a complete response')
        result.extend(block)
    return bytes(result)


class Controller:
    def __init__(self, path: Path, *, worker=False):
        self.process = None
        if not path.is_file() or path.is_symlink():
            raise InvalidController('Submission must be a regular controller.onnx file')
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise InvalidController('Controller exceeds 4 MiB')
        try:
            model = onnx.load_model_from_string(raw)
        except DecodeError as error:
            raise InvalidController('Malformed ONNX protobuf') from error
        if model.functions or model.training_info:
            raise InvalidController('ONNX local functions/training graphs are forbidden')
        if len(model.opset_import) != 1 or model.opset_import[0].domain not in ('', 'ai.onnx') or model.opset_import[0].version != 18:
            raise InvalidController('Use only the standard ONNX opset 18')
        if model.ir_version > 10:
            raise InvalidController('Use ONNX IR version 10 or earlier')
        if len(model.graph.node) > 512:
            raise InvalidController('Controller exceeds 512 graph nodes')
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise InvalidController('Controller requires one input and one output')
        no_external_tensors(model.graph)
        try:
            onnx.checker.check_model(model, full_check=True)
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as error:
            raise InvalidController('Invalid ONNX graph: ' + str(error)) from error
        tensors = list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info)
        known = {v.name for v in tensors} | {v.name for v in inferred.graph.initializer}
        if any(name not in known for node in inferred.graph.node for name in node.output):
            raise InvalidController('ONNX shape inference must cover every intermediate tensor')
        for value in tensors:
            t = value.type.tensor_type
            if t.elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE,
                                   onnx.TensorProto.INT32, onnx.TensorProto.INT64,
                                   onnx.TensorProto.BOOL):
                raise InvalidController('Controller intermediates must be bounded numeric/bool tensors')
            dimensions = [d.dim_value for d in t.shape.dim]
            if not dimensions or any(d <= 0 or d > 4096 for d in dimensions):
                raise InvalidController('Every ONNX tensor must have bounded static dimensions')
            if math.prod(dimensions) > 65536:
                raise InvalidController('ONNX intermediate tensor exceeds 65536 elements')
        for tensor in inferred.graph.initializer:
            if tensor.dims and (any(d <= 0 or d > 4096 for d in tensor.dims)
                                or math.prod(tensor.dims) > 65536):
                raise InvalidController('ONNX initializer exceeds bounded tensor size')
        input = inferred.graph.input[0]
        output = inferred.graph.output[0]
        for value, expected in [(input, [1, 48]), (output, [1, 34])]:
            if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                raise InvalidController('Input/output must be float32')
            if [d.dim_value for d in value.type.tensor_type.shape.dim] != expected:
                raise InvalidController(f'Expected controller tensor shape {expected}')
        self.worker = worker
        self.input_name = input.name
        if not worker:
            self.process = subprocess.Popen([sys.executable, '-B', '-I', str(Path(__file__).resolve()),
                                             '--worker', str(path.resolve())],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, bufsize=0)
            try:
                if read_deadline(self.process.stdout, 4, time.monotonic()+20.) != b'T0OK':
                    raise InvalidController('Controller worker could not initialize its ONNX graph')
            except BaseException:
                self.close()
                raise
            return
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        try:
            self.session = ort.InferenceSession(raw, sess_options=options, providers=['CPUExecutionProvider'])
        except (ort.capi.onnxruntime_pybind11_state.InvalidGraph,
                ort.capi.onnxruntime_pybind11_state.InvalidArgument) as error:
            raise InvalidController('ONNX Runtime rejected the graph: ' + str(error)) from error

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.)
            process.stdin.close()
            process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, 'process', None) is not None:
            self.close()

    def step(self, features: np.ndarray):
        if features.shape != (48,) or not np.isfinite(features).all():
            raise ArithmeticError('Trusted observation has invalid shape/nonfinite values')
        started = time.monotonic()
        if self.worker:
            answer = self.session.run(None, {self.input_name: features[None].astype(np.float32)})[0]
        else:
            try:
                self.process.stdin.write(features.astype('<f4').tobytes())
                raw = read_deadline(self.process.stdout, 34 * 4, started + 1.)
                answer = np.frombuffer(raw, dtype='<f4').reshape(1, 34)
            except (BrokenPipeError, InvalidController) as error:
                self.close()
                raise InvalidController('Controller failed or exceeded the hard 1 second call deadline') from error
        if answer.shape != (1, 34) or not np.isfinite(answer).all():
            raise InvalidController('Controller produced invalid action/memory')
        if np.max(np.abs(answer[0, 2:]), initial=0.0) > 100.0:
            raise InvalidController('Controller memory must remain in [-100,100]')
        return answer[0, :2].astype(float), answer[0, 2:].astype(float)


def worker_main(path):
    # Only trusted Python and a previously validated standard graph run here.
    # No plant, scenario, observation history or future disturbance is imported.
    try:
        controller = Controller(Path(path), worker=True)
    except Exception:
        os.write(sys.stdout.fileno(), b'T0ER')
        return 1
    os.write(sys.stdout.fileno(), b'T0OK')
    while True:
        block = sys.stdin.buffer.read(48 * 4)
        if not block:
            return 0
        if len(block) != 48 * 4:
            return 1
        try:
            action, memory = controller.step(np.frombuffer(block, dtype='<f4'))
        except Exception:
            return 1
        os.write(sys.stdout.fileno(), np.concatenate((action, memory)).astype('<f4').tobytes())


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('Controller worker accepts only its trusted fixed protocol')
    raise SystemExit(worker_main(sys.argv[2]))
