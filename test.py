import tensorrt as trt
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

onnx_file = "./RetinaNet_v3.onnx"
engine_file = "./model_8.trt"
calibration_data_path = "./cali_datei"

# ✅ TensorRT Logger
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# ✅ Load ONNX Model
with open(onnx_file, "rb") as model:
    if not parser.parse(model.read()):
        for i in range(parser.num_errors):
            print(f"ONNX Parsing Error {i}: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model")
    else:
        print("✅ ONNX model successfully parsed!")

# ✅ Create TensorRT Config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
config.set_flag(trt.BuilderFlag.INT8)  # 🔥 Enable INT8
config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)  # 🔥 Avoid Timing Cache Errors

import tensorrt as trt
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data_folder, batch_size=1, cache_file="calibration.cache"):
        super(Int8Calibrator, self).__init__()

        self.batch_size = batch_size
        self.cache_file = cache_file

        # 🔥 获取所有 .npy 文件
        self.image_paths = sorted([
            os.path.join(calibration_data_folder, f)
            for f in os.listdir(calibration_data_folder) if f.endswith(".npy")
        ])
        self.current_index = 0

        # ✅ 预分配 GPU 设备内存 (Device Memory)
        self.device_input = cuda.mem_alloc(batch_size * 3 * 640 * 640 * np.float32().nbytes)

        # ✅ 预分配 CPU (Host) 内存
        self.pinned_memory = np.zeros((batch_size, 3, 640, 640), dtype=np.float32)

        # ✅ 创建一个生成器 (batches) 提供数据
        self.batches = self._batch_generator()

        print(f"✅ Initialized Int8Calibrator with {len(self.image_paths)} calibration images")

    def _batch_generator(self):
        """ 生成校准数据 """
        for i in range(0, len(self.image_paths), self.batch_size):
            batch = []
            for j in range(self.batch_size):
                if i + j < len(self.image_paths):
                    npy_file = self.image_paths[i + j]
                    img = np.load(npy_file).astype(np.float32)

                if img.shape == (1, 3, 640, 640):
                    img = img.unsqueeze()
                
                batch.append(img)

            if len(batch) > 0:
                yield np.array(batch)
            else:
                raise RuntimeError("❌ Batch data is empty!")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            # ✅ 从生成器获取数据
            data = next(self.batches)
            print(f"✅ Loaded batch, shape: {data.shape}")

            # ✅ 复制数据到 GPU
            cuda.memcpy_htod(self.device_input, data)
            print(f"✅ Copy batch success.")

            # ✅ 返回 GPU 设备内存指针
            return [int(self.device_input)]

        except StopIteration:
            print("❌ No more calibration data available")
            return None

    def read_calibration_cache(self):
        """✅ 读取 INT8 校准缓存"""
        try:
            with open(self.cache_file, "rb") as f:
                cache = f.read()
            print(f"✅ Using existing INT8 calibration cache: {self.cache_file}")
            return cache
        except FileNotFoundError:
            print(f"❌ Calibration cache not found, running fresh calibration")
            return None

    def write_calibration_cache(self, cache):
        """✅ 写入 INT8 校准缓存"""
        if cache is None or len(cache) == 0:
            print("❌ Calibration cache is empty, possible issue!")
        else:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            print(f"✅ Calibration cache saved: {self.cache_file}")

# ✅ Apply INT8 Calibrator
config.int8_calibrator = Int8Calibrator(calibration_data_path)

# ✅ Create Optimization Profile
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))  # 🔥 Static batch size = 1
config.add_optimization_profile(profile)

# ✅ Build TensorRT INT8 Engine
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("❌ Failed to build TensorRT INT8 engine!")

# ✅ Save TensorRT INT8 Engine
with open(engine_file, "wb") as f:
    f.write(serialized_engine)

print(f"✅ INT8 TensorRT engine saved at: {engine_file}")
