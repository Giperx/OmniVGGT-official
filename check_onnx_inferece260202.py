#!/usr/bin/env python
import os
os.environ['ORT_LOGGING_LEVEL'] = '2'

import onnxruntime as ort
import numpy as np
import time

print("=" * 60)
print("ONNX Runtime 推理测试")
print("=" * 60)

# 创建一个简单的测试会话
def create_simple_model():
    """创建一个简单的 ONNX 模型进行测试"""
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    # 导出为 ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input, 
        "test_model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("✓ 测试模型已创建: test_model.onnx")

# 测试推理
def test_inference(provider_name):
    """测试指定 provider 的推理性能"""
    print(f"\n--- 测试 {provider_name} ---")
    
    try:
        # 创建会话
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 2
        
        sess = ort.InferenceSession(
            "test_model.onnx",
            sess_options=session_options,
            providers=[provider_name]
        )
        
        # 检查实际使用的 provider
        actual_provider = sess.get_providers()[0]
        print(f"实际使用的 Provider: {actual_provider}")
        
        # 获取输入输出信息
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        print(f"输入: {input_name}, shape: {input_shape}")
        print(f"输出: {output_name}")
        
        # 准备测试数据
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # 预热
        for _ in range(5):
            _ = sess.run([output_name], {input_name: test_input})
        
        # 性能测试
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            outputs = sess.run([output_name], {input_name: test_input})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
        print(f"✓ 推理成功")
        print(f"平均推理时间: {avg_time:.2f} ms")
        print(f"输出 shape: {outputs[0].shape}")
        
        return True, avg_time
    
    except Exception as e:
        print(f"✗ 推理失败: {str(e)}")
        return False, None

# 主测试流程
def main():
    # 1. 创建测试模型
    try:
        create_simple_model()
    except Exception as e:
        print(f"✗ 创建模型失败: {str(e)}")
        return
    
    # 2. 测试不同的 providers
    available_providers = ort.get_available_providers()
    results = {}
    
    # CPU 测试
    if 'CPUExecutionProvider' in available_providers:
        success, time_cpu = test_inference('CPUExecutionProvider')
        if success:
            results['CPU'] = time_cpu
    
    # CUDA 测试
    if 'CUDAExecutionProvider' in available_providers:
        success, time_cuda = test_inference('CUDAExecutionProvider')
        if success:
            results['CUDA'] = time_cuda
    
    # TensorRT 测试
    if 'TensorrtExecutionProvider' in available_providers:
        success, time_trt = test_inference('TensorrtExecutionProvider')
        if success:
            results['TensorRT'] = time_trt
    
    # 3. 性能对比
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("性能对比:")
        print("=" * 60)
        for provider, exec_time in results.items():
            print(f"{provider:15s}: {exec_time:6.2f} ms")
        
        if 'CPU' in results and 'CUDA' in results:
            speedup = results['CPU'] / results['CUDA']
            print(f"\nCUDA 加速比: {speedup:.2f}x")
    
    # 4. 清理
    import os
    if os.path.exists("test_model.onnx"):
        os.remove("test_model.onnx")
        print("\n✓ 测试模型已清理")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()