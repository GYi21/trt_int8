# trt_int8
retinanet: https://github.com/yhenon/pytorch-retinanet

trt Version: 10.8.0.43

torch Vision: 2.4.1


问题描述：

1. ONNX无法转化到INT8：它可以转化成FP16并且使用trtexec在没有cathe的情况也可以转换成INT8。但是通过Python API 配上 npy的校准集却不能转化。（怀疑校准器接收不到校准集）
错误代码 ：
[ERROR] Exception caught in get_batch(): Unable to cast Python instance to C++ type (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)

Code 2: Internal Error (Assertion attr.type == cudaMemoryTypeHost failed. )
[03/20/2025-15:47:22] [TRT] [E] [calibrator.cpp::calibrateEngine::1236] Error Code 2: Internal Error (Assertion context->executeV2(bindings.data()) failed. )

[03/20/2025-15:42:47] [TRT] [E] [calibrator.cpp::add::810] Error Code 1: Cuda Runtime (an illegal memory access was encountered)


2. 同样的转化代码(Python API)，同样的校准集(npy)在rtdetr能够实现
