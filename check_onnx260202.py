#!/usr/bin/env python
import os
os.environ['ORT_LOGGING_LEVEL'] = '2'  # 显示警告和错误

import onnxruntime as ort
import numpy as np

print("=" * 60)
print("ONNX Runtime 验证")
print("=" * 60)

# 1. 版本信息
print(f"\n1. ONNX Runtime 版本: {ort.__version__}")

# 2. 可用的 Execution Providers
print(f"\n2. 可用的 Execution Providers:")
available_providers = ort.get_available_providers()
for provider in available_providers:
    print(f"   - {provider}")

# 3. 检查 GPU 支持
has_cuda = 'CUDAExecutionProvider' in available_providers
has_tensorrt = 'TensorrtExecutionProvider' in available_providers

print(f"\n3. GPU 支持:")
print(f"   - CUDA 支持: {'✓ YES' if has_cuda else '✗ NO'}")
print(f"   - TensorRT 支持: {'✓ YES' if has_tensorrt else '✗ NO'}")

# 4. 设备信息
print(f"\n4. CUDA 设备信息:")
if has_cuda:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   - PyTorch CUDA: {torch.cuda.is_available()}")
            print(f"   - CUDA 版本: {torch.version.cuda}")
            print(f"   - GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   - PyTorch CUDA 不可用")
    except ImportError:
        print("   - PyTorch 未安装，无法获取详细 GPU 信息")
else:
    print("   - CUDA Execution Provider 不��用")

print("\n" + "=" * 60)