# Copyright (c) Tencent Inc. All rights reserved.
import abc
import copy
import os
import random
import traceback
import cv2
import argparse
import os.path as osp
from functools import partial
from io import BytesIO
from copy import deepcopy

# import mtk_quantization.pytorch
# import mtk_quantization.pytorch
import onnx
from onnx import TensorProto
from onnx.numpy_helper import to_array, from_array

import onnxsim
import torch
# import torch.quantization as quantization
import torch.nn as nn
import torch.optim as optim
# from torch.ao.quantization import QConfig, float_qparams_weight_only_qconfig
# from torch.ao.quantization.fake_quantize import default_fake_quant, default_weight_fake_quant
# import torch.ao.quantization.fx as fx
# from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import get_default_qat_qconfig_mapping, get_default_qat_qconfig
# from torch.ao.quantization import fuse_fx
import torch.ao.quantization as quantization
import gradio as gr
import numpy as np
import supervision as sv
from PIL import Image
from torchvision.ops import nms
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.visualization import DetLocalVisualizer
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

from yolo_world.easydeploy.model import DeployModel, MMYOLOBackend
import tensorflow as tf

import os
# from onnxruntime.quantization import quantize_static, CalibrationDataReader
# from onnxruntime.quantization import QuantType, CalibrationMethod

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)

def get_calibrattion_data(runner, text, use_static_data=False, num_samples=100):
    import os
    import random
    import numpy as np

    if use_static_data:
        # 靜態生成數據
        for _ in range(num_samples):
            yield np.random.randn(1, 640, 640, 3).astype(np.float32)
        return

    # 動態生成數據
    root = "/storage/SSD-3/yptsai/data/coco2017/val2017"
    image_list = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(image_list)
    total_images = image_list[:num_samples]
    dirs = ['Build_fire_detection_solutions.v1i.yolov8/train/images/', 'ASD.v8i.yolov8/train/images/', 'fire_detection.v2-fire-detection.yolov8.zip/train/images/','Fire_detection.v2i.yolov8/train/images/', 'kidBoy.v8i.yolov8/train/images/', 'Packages_on_Conveyor.v1i.yolov8/train/images/','piscina.v1i.yolov8/train/images/','pool/train/images/']
    root = "/storage/SSD-3/yptsai/data/" 
    spec_images=[] 
    for d in dirs:
        cur_path = root + d
        image_list = [os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.lower().endswith(('.jpg', '.png'))]
        if len(image_list)>0:
            spec_images.extend(image_list)
    image_list = total_images
    random.shuffle(spec_images)
    num_samples = min(len(spec_images), num_samples)
    image_list = image_list+spec_images[:num_samples*10]
    num_samples = len(image_list)
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    
    for idx, file in enumerate(image_list[:num_samples]):
        data_info = dict(img_id=0, img_path=file, texts=texts)
        data_info = runner.pipeline(data_info)

        # data_batch = dict(
        #     inputs=data_info['inputs'].unsqueeze(0),
        #     data_samples=[data_info['data_samples']]
        # )
        # fake_input_2 = runner.model.data_preprocessor(data_batch, False)

        # yield [fake_input_2['inputs'].permute(0, 2, 3, 1).cpu().numpy()]
        yield [data_info['inputs'].to(dtype=torch.float32).unsqueeze(0)[:1,[2,1,0],...].permute(0, 2, 3, 1).cpu().numpy()]
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            model(inputs)
 
class CalibrationDataReader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "get_next") and callable(subclass.get_next)
            or NotImplemented
        )

    @abc.abstractmethod
    def get_next(self) -> dict:
        """Generate the input data dict for ONNX inferenceSession run."""
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        result = self.get_next()
        if result is None:
            raise StopIteration
        return result

    def __len__(self):
        raise NotImplementedError

    def set_range(self, start_index: int, end_index: int):
        raise NotImplementedError


class DataReader(CalibrationDataReader):
    def __init__(self, generator, data_length=100):
        """
        Args:
            generator: A generator that produces the calibration data.
            data_length: The total number of data samples available.
        """
        self.generator = generator
        self.data_length = data_length
        self.reset_iterator()

    def reset_iterator(self):
        """Reset the generator iterator."""
        try:
            self.iterator = iter(self.generator)
        except TypeError as e:
            raise ValueError("Provided generator is not iterable.") from e

    def get_next(self):
        """
        Retrieve the next data sample from the generator.
        If the generator is exhausted, reset and start over.
        """
        try:
            data = next(self.iterator)
            if data is None:
                raise ValueError("Generator returned None.")
            return {"images": data}
        except StopIteration:
            print("Data exhausted, resetting iterator.")
            self.reset_iterator()
            return {"images": next(self.iterator)}
        except Exception as e:
            raise RuntimeError("Error retrieving data from generator.") from e

    def __len__(self):
        return self.data_length

# 通用 QConfig
# custom_qconfig = QConfig(
#     activation=default_fake_quant,
#     weight=default_weight_fake_quant
# )

def set_qat_qconfig_all_layers(model, parent_name=''):
    """
    遍歷模型所有層，遞歸設置 QConfig。
    - nn.Embedding 使用 float_qparams_weight_only_qconfig
    - Conv/Linear 使用 custom_qconfig
    """
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, nn.Embedding):
            # 特殊處理 Embedding
            module.qconfig = torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig
            print(f"[INFO] QConfig set for Embedding: {full_name}")
        elif 'text' in name:
            # 通用 QConfig
            module.qconfig = None
            print(f"[INFO] QConfig set None for text in: {full_name}")
        elif 'data_preprocessor' in name:
            module.qconfig = None
            print(f"[INFO] QConfig set None for data_preprocessor in: {full_name}")
        else:
            # 遞歸處理子模組
            set_qat_qconfig_all_layers(module, full_name)

import torch.nn.quantized as nnq

class QuantizedEmbedding(nnq.Embedding):
    @classmethod
    def from_float(cls, mod):
        return nnq.Embedding(mod.num_embeddings, mod.embedding_dim)
class SiLUApprox(torch.nn.Module):
    def forward(self, x):
        return x * nnq.Sigmoid(x)  # Swish 激活的基本形式
# 替換模型中的 nn.Embedding
def replace_embedding_with_quantized(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Embedding):
            setattr(module, name, QuantizedEmbedding.from_float(child))
        else:
            replace_embedding_with_quantized(child)


        
def replace_activation(module, target_activation, replacement_activation):
    """
    遞歸替換模型中的目標激活函數。

    Args:
        module (nn.Module): 模型或子模組。
        target_activation (nn.Module): 需要替換的激活函數類型。
        replacement_activation (nn.Module): 替換為的新激活函數類型。
    """
    for name, child in module.named_children():
        if isinstance(child, target_activation):
            print(f"Replacing {name}: {target_activation} -> {replacement_activation}")
            setattr(module, name, replacement_activation())
        else:
            # 遞歸處理子模組
            replace_activation(child, target_activation, replacement_activation)
            
# 替換 SiLU
def replace_silu_recursive(module):
    if not isinstance(module, nn.Module):
        return  # 如果模塊不是 torch.nn.Module，跳過處理
    for name, child in list(module._modules.items()):  # 使用 list 防止修改錯誤
        if isinstance(child, nn.SiLU):
            module._modules[name] = nn.ReLU()  # 替換 SiLU
        else:
            replace_silu_recursive(child)  # 遞歸處理子模塊

# 自定義 ReplaceSiLU
class ReplaceSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    # 支持量化的必要函數
    def fuse_modules(self):
        pass  # 如果需要融合模塊，可在此處實現

    def prepare_for_quantization(self):
        self.relu = torch.nn.quantized.ReLU()  # 替換為量化版本的 ReLU

def custom_convert(model):
    for name, module in model.named_modules():
        if isinstance(module, ReplaceSiLU):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else None
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, name.split('.')[-1], nn.ReLU())  # 替換為標準 ReLU
            else:
                setattr(model, name, nn.ReLU())

from torch.ao.quantization import fuse_modules

def fuse_all_layers(module):
    for name, submodule in module.named_children():
        # 跳過不需要融合的模塊
        if isinstance(submodule, torch.nn.Identity) or isinstance(submodule, torch.nn.Conv2d):
            continue
        
        # 如果子模塊有 conv, bn, activate，嘗試融合
        if hasattr(submodule, "conv") and hasattr(submodule, "bn") and hasattr(submodule, "activate"):
            try:
                fuse_modules(submodule, ["conv", "bn", "activate"], inplace=True)
            except AssertionError as e:
                print(f"Skipping fusion for {name}: {e}")
        else:
            # 遞歸處理子模塊
            fuse_all_layers(submodule)
                              
def mtk_qat(runner, model, text, test_input):
    
    import mtk_converter
    original_model = model
    original_model.eval()

    # 使用 deepcopy 複製模型
    quantized_model = copy.deepcopy(original_model)
    # quantized_model.train()
    data_preprocessor = quantized_model.baseModel.data_preprocessor
    quantized_model.baseModel.data_preprocessor = None
    
    # 3. 設定量化配置 (QAT)
    # qconfig_mapping = get_default_qat_qconfig("fbgemm")  # CPU-friendly量化配置
    # quantized_model = prepare_qat_fx(quantized_model, qconfig_mapping, test_input)    
    # modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
    # quantized_model = torch.ao.quantization.fuse_modules(quantized_model, modules_to_fuse)
    # quantized_model.fuse_model(is_qat=True)
    

    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    quantized_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    set_qat_qconfig_all_layers(quantized_model)
    # 驗證所有層的 QConfig
    print("\n[INFO] Model QConfig settings:")
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'qconfig') and module.qconfig:
            print(f"{name}: {module.qconfig}")
    replace_silu_recursive(quantized_model)
    # quantized_model.eval()
    fuse_all_layers(quantized_model)
    # replace_embedding_with_quantized(quantized_model)
    quantized_model.train()
    torch.ao.quantization.prepare_qat(quantized_model, inplace=True)
    
    # 模擬訓練
    optimizer = optim.Adam(quantized_model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    batch=4
    num_data=4
    total_loss=0
    def calibration_data_function():
        return iter(get_calibrattion_data(runner, text, num_samples=num_data))
    # 訓練迴圈
    for i, input_tensor in enumerate(calibration_data_function()):
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).to(test_input.device)
        with torch.no_grad():
            original_output = original_model(input_tensor)  # 原始模型輸出

        quantized_output = quantized_model(input_tensor)  # 量化模型輸出

        # 逐元素計算損失
        for j in range(len(original_output)):
            element_loss = mse_loss(quantized_output[j].float().to('cpu'), original_output[j].float().to('cpu'))  # 計算每個元素的 MSE
            total_loss += element_loss  # 累加損失
        if i%batch==(batch-1):
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"Step {i + 1}, Loss: {total_loss.item()}")
            total_loss=0
        if i > 3 * num_data:
            # Freeze quantizer parameters
            quantized_model.apply(torch.ao.quantization.disable_observer)
        if i > 2 * num_data:
            # Freeze batch norm mean and variance estimates
            quantized_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    # 7. 將 QAT 模型轉換為量化模型
    # quantized_model.eval()
    # quantized_model = convert_fx(quantized_model)
    # Check the accuracy after each epoch
    quantized_model=quantized_model.eval()
    # fuse_all_layers(quantized_model)
    quantized_model = torch.ao.quantization.convert(quantized_model, inplace=True)
    scripted_model = torch.jit.trace(quantized_model.cpu(),test_input.cpu())
    # custom_convert(quantized_model)
    with BytesIO() as f:
        # output_names = ['num_dets', 'boxes', 'scores', 'labels']
        # output_names = ['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        output_names = ['topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        # output_names = ['scores','boxes']
        torch.onnx.export(
            quantized_model.cpu(),
            test_input.cpu(),
            f,
            input_names=['images'],
            output_names=output_names,
            do_constant_folding=True,
            opset_version=13)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, "after_quan.onnx")
    # 1. 保存量化後模型為 TorchScript
    # scripted_model = torch.jit.script(quantized_model)
    device = torch.device("cuda")
    def move_model_to_cpu(model):
        # 將所有參數和緩衝區移動到 CPU
        for param in model.parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)

        for buffer in model.buffers():
            buffer.data = buffer.data.to(device)

        # 遞歸處理所有子模塊
        for submodule in model.children():
            move_model_to_cpu(submodule)
    move_model_to_cpu(quantized_model)
    quantized_model = quantized_model.to(device)
    scripted_model= torch.jit.trace(quantized_model, test_input.to(device))
    scripted_model.save("scripted_model.pt")
    print("Quantized model saved as TorchScript (quantized_model.pt)")
    mtk_converter.PyTorchV2Converter.from_exported_program(scripted_model,input_shapes=[(1,640,640,3)])
    
    # 8. 測試量化後的模型
    with torch.no_grad():
        # test_input = torch.randn(1, 3, 8, 8)
        output = quantized_model(test_input)
        print("Quantized Model Output:", output)

    # 9. 保存量化後模型為 TorchScript
    scripted_model = torch.jit.script(quantized_model)
    scripted_model.save("quantized_model.pt")
    print("Quantized model saved as TorchScript (quantized_model.pt)")
    
    with BytesIO() as f:
        # output_names = ['num_dets', 'boxes', 'scores', 'labels']
        # output_names = ['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        output_names = ['topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        # output_names = ['scores','boxes']
        torch.onnx.export(
            scripted_model,
            test_input,
            f,
            input_names=['images'],
            output_names=output_names,
            do_constant_folding=True,
            opset_version=13)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, "after_quan.onnx")
    #################
    graph = onnx_model.graph

    # 修改輸出名稱
    for output in graph.output:
        print(output.name)
    # 遍历节点，找到 TopK 操作
    for node in graph.node:
        if node.op_type == "TopK":
            print(f"Found TopK Node: {node.name}")
            
            # 确定 TopK 的第二个输入（k 值）的名字
            k_input_name = node.input[1]
            print(f"K input name: {k_input_name}")
            
            indices_output_name = node.output[1]
            print(f"Indices output name: {indices_output_name}")
            # 遍历初始化器，找到对应的 k
            for initializer in graph.initializer:
                if initializer.name == k_input_name:
                    print(f"Found initializer for K: {initializer.name}")
                    
                    # 转换为 NumPy 数组
                    k_array = to_array(initializer)
                    
                    # 检查 k 是否是形状为 [1] 的张量
                    if k_array.shape == (1,):
                        print(f"Original K value: {k_array}")

                        # 修改为形状为 [] 的标量
                        k_scalar = k_array.item()  # 提取标量值
                        new_initializer = from_array(
                            np.array(k_scalar, dtype=k_array.dtype).reshape(()),
                            name=initializer.name
                        )                        
                        # 替换原来的初始化器
                        graph.initializer.remove(initializer)
                        graph.initializer.append(new_initializer)
                        print(f"Modified K to scalar: {k_scalar}")
                # elif initializer.name == indices_output_name:
                #     nitializer_found = True
                #     print(f"Modifying initializer {initializer.name}")
                    
                #     # 提取數據並轉換為 uint8
                #     original_data = to_array(initializer)
                #     converted_data = original_data.astype(np.uint8)
                    
                #     # 創建新的初始化器
                #     new_initializer = from_array(
                #         converted_data, name=initializer.name)
                    
                #     # 替換舊的初始化器
                #     graph.initializer.remove(initializer)
                #     graph.initializer.append(new_initializer)
                    
            # 修改形状信息
            for value_info in graph.value_info:
                if value_info.name == k_input_name:
                    print(f"Before modification: {value_info}")
                    value_info.type.tensor_type.shape.dim.clear()  # 清空形状，表示标量
                    value_info.type.tensor_type.elem_type = TensorProto.INT64  # 设置数据类型
                    print(f"After modification: {value_info}")
                # elif value_info.name == indices_output_name:
                #     print(f"Before modification: {value_info}")
                #     # value_info.type.tensor_type.shape.dim.clear()  # 清空形状，表示标量
                #     value_info.type.tensor_type.elem_type = TensorProto.UINT8  # 设置数据类型
                    # print(f"After modification: {value_info}")
    # 保存模型
    modified_filename = "modified_model.onnx"
    onnx.save(onnx_model, modified_filename) 
    print("Model saved as modified_model.onnx")
    print("Model exported to ONNX.")
    converter = mtk_converter.OnnxConverter.from_model_proto(onnx_model, input_names=["images"], input_shapes=[(1,640,640,3)],output_names=['topk_scores', 'topk_classes','topk_indices','topk_bboxes'])
    converter.quantize = True
    converter.input_value_ranges = [(0, 255)]
    converter.use_unsigned_quantization_type=True
    converter.use_hessian_opt = True
    
def mtk_torchao(runner, model, text, test_input):
    from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer    
    qat_quantizer = Int32DynActInt4WeightQATQuantizer()
    original_model = model
    original_model.eval()

    # 使用 deepcopy 複製模型
    quantized_model = copy.deepcopy(original_model)
    quantized_model = qat_quantizer.prepare(quantized_model)
    quantized_model.to(test_input.device).train()
    # 模擬訓練
    optimizer = optim.Adam(quantized_model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    num_data = 10
    def calibration_data_function():
        return iter(get_calibrattion_data(runner, text, num_samples=num_data))
    
    total_loss=0
    batch = 4
    # 訓練迴圈
    for i, input_tensor in enumerate(calibration_data_function()):
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).to(test_input.device)
        with torch.no_grad():
            original_output = original_model(input_tensor)  # 原始模型輸出

        quantized_output = quantized_model(input_tensor)  # 量化模型輸出

        # 逐元素計算損失
        for j in range(len(original_output)):
            element_loss = mse_loss(quantized_output[j].float().to('cpu'), original_output[j].float().to('cpu'))  # 計算每個元素的 MSE
            total_loss += element_loss  # 累加損失
        if i%batch==(batch-1):
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"Step {i + 1}, Loss: {total_loss.item()}")
            total_loss=0
    model = qat_quantizer.convert(quantized_model)
    print(model)    
                     
    
def mtk_QAT_and_export_tflite(runner, model, text, test_input):
    # 準備模型
    
    
    original_model = model
    original_model.eval()


    # 使用 deepcopy 複製模型
    quantized_model = copy.deepcopy(original_model)
    quantized_model.to(test_input.device).train()
    quantized_model.to('cpu')  # 模型轉移到 CPU
    # torch.backends.quantized.engine = 'x86' # 'fbgemm'

    replace_activation(quantized_model, torch.nn.SiLU, SiLUApprox)
    replace_activation(quantized_model, torch.nn.Sigmoid, nnq.Sigmoid)
    set_qat_qconfig_all_layers(quantized_model)
    # 驗證所有層的 QConfig
    print("\n[INFO] Model QConfig settings:")
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'qconfig') and module.qconfig:
            print(f"{name}: {module.qconfig}")
    replace_embedding_with_quantized(quantized_model)
    quantized_model  = quantization.prepare_qat(quantized_model, inplace=True)
    # 模擬訓練
    optimizer = optim.Adam(quantized_model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    num_data = 10
    def calibration_data_function():
        return iter(get_calibrattion_data(runner, text, num_samples=num_data))
    
    total_loss=0
    batch = 4
    # 訓練迴圈
    for i, input_tensor in enumerate(calibration_data_function()):
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).to(test_input.device)
        with torch.no_grad():
            original_output = original_model(input_tensor)  # 原始模型輸出

        quantized_output = quantized_model(input_tensor.to('cpu'))  # 量化模型輸出

        # 逐元素計算損失
        for j in range(len(original_output)):
            element_loss = mse_loss(quantized_output[j].float().to('cpu'), original_output[j].float().to('cpu'))  # 計算每個元素的 MSE
            total_loss += element_loss  # 累加損失
        if i%batch==(batch-1):
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"Step {i + 1}, Loss: {total_loss.item()}")
            total_loss=0


    # 轉換量化模型為最終格式
    quantized_model.eval()
    quantized_model = quantization.convert(quantized_model, inplace=True).to('cpu')
    # float_model = torch.quantization.convert(quantized_model, remove_qconfig=True)
    # 儲存量化後的模型
    torch.save(quantized_model, 'quantized_model.pth')
    with BytesIO() as f:
        # output_names = ['num_dets', 'boxes', 'scores', 'labels']
        # output_names = ['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        output_names = ['topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        # output_names = ['scores','boxes']
        torch.onnx.export(
            quantized_model,
            test_input.to('cpu'),
            f,
            input_names=['images'],
            output_names=output_names,
            opset_version=13)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, "quantized_model.onnx")
    # 驗證輸出
    
    # original_output = original_model(test_input)
    # quantized_output = quantized_model(test_input.to('cpu'))

    # print("輸出誤差:", torch.norm(original_output[0].cpu() - quantized_output[0].cpu()).item())

    
    
def mtk_calibration_and_export_tflite(runner, model, text, test_input):
    import mtk_quantization
    import mtk_converter
    
    with BytesIO() as f:
        # output_names = ['num_dets', 'boxes', 'scores', 'labels']
        # output_names = ['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        output_names = ['topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        # output_names = ['scores','boxes']
        torch.onnx.export(
            model,
            test_input,
            f,
            input_names=['images'],
            output_names=output_names,
            do_constant_folding=True,
            opset_version=13)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, "after_quan.onnx")
    #################
    graph = onnx_model.graph

    # 修改輸出名稱
    for output in graph.output:
        print(output.name)
    # 遍历节点，找到 TopK 操作
    for node in graph.node:
        if node.op_type == "TopK":
            print(f"Found TopK Node: {node.name}")
            
            # 确定 TopK 的第二个输入（k 值）的名字
            k_input_name = node.input[1]
            print(f"K input name: {k_input_name}")
            
            indices_output_name = node.output[1]
            print(f"Indices output name: {indices_output_name}")
            # 遍历初始化器，找到对应的 k
            for initializer in graph.initializer:
                if initializer.name == k_input_name:
                    print(f"Found initializer for K: {initializer.name}")
                    
                    # 转换为 NumPy 数组
                    k_array = to_array(initializer)
                    
                    # 检查 k 是否是形状为 [1] 的张量
                    if k_array.shape == (1,):
                        print(f"Original K value: {k_array}")

                        # 修改为形状为 [] 的标量
                        k_scalar = k_array.item()  # 提取标量值
                        new_initializer = from_array(
                            np.array(k_scalar, dtype=k_array.dtype).reshape(()),
                            name=initializer.name
                        )                        
                        # 替换原来的初始化器
                        graph.initializer.remove(initializer)
                        graph.initializer.append(new_initializer)
                        print(f"Modified K to scalar: {k_scalar}")
                # elif initializer.name == indices_output_name:
                #     nitializer_found = True
                #     print(f"Modifying initializer {initializer.name}")
                    
                #     # 提取數據並轉換為 uint8
                #     original_data = to_array(initializer)
                #     converted_data = original_data.astype(np.uint8)
                    
                #     # 創建新的初始化器
                #     new_initializer = from_array(
                #         converted_data, name=initializer.name)
                    
                #     # 替換舊的初始化器
                #     graph.initializer.remove(initializer)
                #     graph.initializer.append(new_initializer)
                    
            # 修改形状信息
            for value_info in graph.value_info:
                if value_info.name == k_input_name:
                    print(f"Before modification: {value_info}")
                    value_info.type.tensor_type.shape.dim.clear()  # 清空形状，表示标量
                    value_info.type.tensor_type.elem_type = TensorProto.INT64  # 设置数据类型
                    print(f"After modification: {value_info}")
                # elif value_info.name == indices_output_name:
                #     print(f"Before modification: {value_info}")
                #     # value_info.type.tensor_type.shape.dim.clear()  # 清空形状，表示标量
                #     value_info.type.tensor_type.elem_type = TensorProto.UINT8  # 设置数据类型
                    # print(f"After modification: {value_info}")
    # 保存模型
    modified_filename = "modified_model.onnx"
    onnx.save(onnx_model, modified_filename) 
    print("Model saved as modified_model.onnx")
    print("Model exported to ONNX.")
    converter = mtk_converter.OnnxConverter.from_model_proto(onnx_model, input_names=["images"], input_shapes=[(1,640,640,3)],output_names=['topk_scores', 'topk_classes','topk_indices','topk_bboxes'])
    converter.quantize = True
    converter.input_value_ranges = [(0, 255)]
    converter.use_unsigned_quantization_type=True
    converter.use_hessian_opt = True
    # data_reader = DataReader(generator=get_calibrattion_data(runner=runner, text=text))
    
    def data_gen():
        for i in range(100):
            yield [np.random.randn(1,640,640,3).astype(np.float32)] 
    # # calibr_data = lambda: iter(data_reader)
    # converter._calibration_data_gen = lambda: iter(data_reader)
    num_data =200
    # data_reader = DataReader(generator=get_calibrattion_data(runner, text, num_samples=num_data), data_length=num_data)
    # 確保 DataReader 運行正常
    # for idx, sample in enumerate(iter(data_reader)):
    #     if idx >= 10:
    #         break
    #     print(f"Read sample {idx}: {sample}")
    # generator=get_calibrattion_data(runner, text, num_samples=num_data)
    # 包裝 calibration data 為函數
    # num_data = 5000
    def calibration_data_function():
        return iter(get_calibrattion_data(runner, text, num_samples=num_data))

    # 正確設置 calibration_data_gen
    converter.append_output_dequantize_ops=True
    converter.calibration_data_gen = calibration_data_function
    tflite_file = './work_dirs/quantized_model.tflite'
    _ = converter.convert_to_tflite(output_file=tflite_file, tflite_op_export_spec='npsdk_v7')
    # output_file = f'mtk_yolow_{text.replace(",","_").replace(" ","")}'
    # output_file = f'{output_file[:90]}{'..' if len(output_file)>90 else ''}.dla'
    output_file = './work_dirs/quantized_model.dla'
    # cmd = f'LD_LIBRARY_PATH=neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/host/lib neuropilot-sdk-basic-8.0.5-build20241127/neuron_sdk/host/bin/ncc-tflite --arch=mdla5.1,mvpu2.5 -O3 --show-exec-plan {tflite_file} -o {output_file}'
    import subprocess

    # 設置環境變量
    env = {
        "LD_LIBRARY_PATH": "neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/host/lib",
    }

    # 命令拆分為列表
    cmd = [
        "neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/host/bin/ncc-tflite",
        "--arch=mdla5.1,mvpu2.5",
        "-O3",
        "--show-exec-plan",
        f"{tflite_file}",
        "-o",
        f"{output_file}",
    ]

    # 使用 subprocess 執行命令
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)


    # 打印執行結果
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return Code:", result.returncode)
    del onnx_model
    del converter
    return (result,output_file, tflite_file)
    
    
    
    
    
    
    
    
    # from torch.quantization import prepare, convert
    # from torch.ao.quantization import quantize, get_default_qconfig

    
    # # 模型準備
    # model.eval()  # 進入評估模式
    # # 設置後端為 qnnpack
    # torch.backends.quantized.engine = 'fbgemm'

    # # 配置量化參數
    # # quantization_config = torch.quantization.get_default_qconfig('qnnpack')
    # quantization_config = get_default_qconfig('fbgemm')
    
    # model.qconfig = quantization_config
    # model.baseModel.qconfig = quantization_config
    # model.baseModel.backbone.qconfig = quantization_config
    # model.baseModel.neck.qconfig = quantization_config
    # model.baseModel.bbox_head.qconfig = quantization_config

    # # 遍歷所有子模塊，禁用名稱中包含 "text" 的模塊的量化
    # for name, module in model.named_modules():
    #     if 'text' in name or 'data_preprocessor' in name:  # 判斷條件
    #         module.qconfig = None  # 禁用量化
    #         print(f"Disabled quantization for module: {name}")
    #     else:
    #         module.qconfig = quantization_config
    #         print(f"enable quantization for module: {name}")
    # data_preprocessor = model.baseModel.data_preprocessor
    # model.baseModel.data_preprocessor = None
    # data_reader = DataReader(generator=get_calibrattion_data(runner=runner, text=text))
    
    # # calibr_data = lambda: iter(data_reader)
    # quantized_model = quantize(
    #     model.to('cpu'),
    #     run_fn=lambda m: m(torch.randn(1, 3, 640, 640).to('cpu')),  # 校準函數
    #     run_args=()
    #     )
    # model.to(device=test_input.device)
    # with BytesIO() as f:
    #     # output_names = ['num_dets', 'boxes', 'scores', 'labels']
    #     output_names = ['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes']
    #     # output_names = ['scores','boxes']
    #     torch.onnx.export(
    #         quantized_model,
    #         test_input.cpu(),
    #         f,
    #         input_names=['images'],
    #         output_names=output_names,
    #         opset_version=13)
    #     f.seek(0)
    #     onnx_model = onnx.load(f)
    #     onnx.checker.check_model(onnx_model)
    #     onnx_model, check = onnxsim.simplify(onnx_model)
    #     onnx.save(onnx_model, "after_quan.onnx")
    # print("Model exported to ONNX.")
    # # 準備模型進行量化
    # # model_prepared = prepare(model)
    
    # # quan_model = mtk_quantization.pytorch.fuse_modules(model_prepared,test_input)
    # # # 模型融合：將 Conv-BN-ReLU 融合為一個操作
    
    # # config_generator = mtk_quantization.pytorch.ConfigGenerator(quan_model)
    # # config_generator.export_config('./work_dirs/quant_config.json', example_inputs=test_input)
    # # quan_model.eval()
    # # correct=0
    # # data_reader = DataReader(generator=get_calibrattion_data(runner=runner, text=text))
    # # calibr_data = lambda: iter(data_reader)
    # # for data in calibr_data:
    # #     quan_model(data)
    # # # 轉換為量化模型
    # # model_quantized = convert(model_prepared)
    
    
    # qat_model_file = "./work_dirs/qat_model.pt"
    # tflite_file = "./work_dirs/qat_model.tflite"
    # test_input_cpu = test_input.cpu()
    # # scripted_model = torch.jit.script(quantized_model.cpu().eval())
    # # torch.jit.save(scripted_model, qat_model_file)
    # # print("Model successfully scripted and saved!")
    
    # torch.jit.save(torch.jit.trace(quantized_model.cpu().eval(), test_input_cpu), qat_model_file)
    # print('finish calibration and output model {qat_model_file}')
    # converter = mtk_converter.PyTorchConverter.from_script_module_file(qat_model_file, test_input_cpu)
    # converter.quantize=True
    # converter.input_value_ranges=[(0,1)]
    # converter.convert_to_tflite(output_file=tflite_file)
        
def calibration_and_tflite(runner, model, text, fake_input):
    from onnx_tf.backend import prepare
    save_onnx_path='before_quan.onnx'    
    
    # Step 1: Export PyTorch model to ONNX
    with BytesIO() as f:
        # output_names = ['num_dets', 'boxes', 'scores', 'labels']
        output_names = ['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes']
        # output_names = ['scores','boxes']
        torch.onnx.export(
            model,
            fake_input,
            f,
            input_names=['images'],
            output_names=output_names,
            opset_version=13)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, save_onnx_path)
    print("Model exported to ONNX.")
    
    # 遍歷所有節點，檢查 Squeeze 操作
    for node in onnx_model.graph.node:
        if node.op_type == "Squeeze":
            print(f"Modifying Squeeze node: {node.name}")
            node.domain = ""  # 確保域是默認域
            # 更新為支持的版本
            node.attribute.clear()  # 清理無效的屬性

    # 保存修改後的模型
    onnx.save(onnx_model, "model_modified.onnx")
    # 列出所有節點及其屬性
    for idx, node in enumerate(onnx_model.graph.node):
        print(f"Node {idx}:")
        print(f"  Name: {node.name}")
        print(f"  Type: {node.op_type}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
    # export onnx
    # from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat

    # # 静态量化
    # data_reader = DataReader(get_calibrattion_data(runner=runner, text=text))
    # quantized_model = quantize_static(
    #     model_input="before_quan.onnx",
    #     model_output="after_quan.onnx",
    #     calibration_data_reader=data_reader,
    #     quant_format=QuantType.QInt8,   # 权重量化为 INT8
    #     per_channel=True,  # 使用通道级量化
    #     activation_type=QuantType.QInt8,
    #     weight_type=QuantType.QInt8,
    #     calibrate_method=CalibrationMethod.MinMax  # 校准方法
    # )
    # print("Model statically quantized.")

    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # 顯示所有日誌，包括 DEBUG
    # os.environ["TF_TFLITE_LOG_LEVEL"] = "3"  # 啟用 TFLite 轉換的詳細日誌

    # 加载量化后的 ONNX 模型
    # onnx_model = onnx.load("after_quan.onnx")
    # onnx_model, check = onnxsim.simplify(onnx_model)
    # 转换为 TensorFlow 格式
    
    # Step 2: Convert ONNX to TensorFlow
    from onnx_tf.backend import prepare
    try:
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph("model_tf")
        print("ONNX model converted to TensorFlow format.")
    except Exception as e:
        print(f"Error during ONNX-TF conversion: {e}")
        return
    
    # 加载 TensorFlow 模型
    model = tf.saved_model.load("model_tf")
    graph = model.signatures["serving_default"].graph

    # 列出所有操作和屬性
    for op in graph.get_operations():
        print(f"Operation name: {op.name}")
        print(f"Operation type: {op.type}")
        print(f"Inputs: {[t.shape for t in op.inputs]}")
        print(f"Outputs: {[t.shape for t in op.outputs]}")
        break
    
    # Convert TensorFlow SavedModel to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    # # Use dynamic range quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # 输入类型量化为 uint8
    converter.inference_output_type = tf.int8  # 输出类型量化为 int8
    
    # # Step 3: Convert TensorFlow SavedModel to TFLite
    # # 加载 TensorFlow SavedModel
    # converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
    # # print(converter.get_supported_operations())
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,  # 默認的內置操作
    #     tf.lite.OpsSet.SELECT_TF_OPS     # 添加 TensorFlow 的兼容操作支持
    #     ]
    # # for op in tf.saved_model.load("model_tf").signatures["serving_default"].graph.get_operations():
    # #     print(op.name, op.type, [input.shape for input in op.inputs])

    # # 指定量化类型
    # # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # # converter.inference_input_type = tf.uint8  # 输入类型
    # # converter.inference_output_type = tf.int8  # 输出类型
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8  # 输入类型
    # converter.inference_output_type = tf.int8  # 输出类型
    # converter.experimental_new_converter = True
    # converter.experimental_enable_resource_variables = True
    # 代表性数据集（仅用于静态量化）
    # def representative_dataset():
    #     for _ in range(100):
    #         data = np.random.rand(1, 640, 640, 3).astype(np.float32) 
    #         yield [data]

    data_reader = DataReader(get_calibrattion_data(runner=runner, text=text))
    # 设置代表性数据集
    converter.representative_dataset = lambda: iter(data_reader)
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 3, 640, 640).astype(np.float32)
            yield [data]

    # for data in data_reader:
    #     print(data.shape)
    #     try:
    #         interpreter = tf.lite.Interpreter(model_content=converter.convert())
    #         interpreter.allocate_tensors()
    #         input_details = interpreter.get_input_details()
    #         print(f"Input shape: {input_details[0]['shape']}, type: {input_details[0]['dtype']}")
    #     except Exception as e:
    #         print(f"Error during inference: {e}")
    #     break

    converter.representative_dataset = representative_dataset

    # 转换为 TFLite
    try:
        # tf.saved_model.save(converter, "debug_saved_model")
        # print("Input arrays: ", converter.get_input_arrays())
        # print("Optimizations: ", converter.optimizations)
        # print("Representative dataset: ", converter.representative_dataset)

        # tf.debugging.set_log_device_placement(True)
        # with tf.profiler.experimental.Profile("logs"):
        #     tflite_model = converter.convert()

        # converter.experimental_new_converter = False
        
        tflite_model = converter.convert()
        # 保存 TFLite 模型
        with open("model_quantized.tflite", "wb") as f:
            f.write(tflite_model)

        print("Model converted to TFLite successfully.")
        
        # # 加载量化后的 TFLite 模型
        # interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
        # interpreter.allocate_tensors()

        # # 检查输入张量类型
        # input_details = interpreter.get_input_details()
        # print("Input type:", input_details[0]['dtype'])  # 应该是 uint8

        # # 检查输出张量类型
        # output_details = interpreter.get_output_details()
        # print("Output type:", output_details[0]['dtype'])  # 应该是 int8 或 uint8
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        import traceback
        traceback.print_exc()
                
# def calibration_and_tflite(runner, model, text, fake_input):
#     from torch.quantization import prepare, convert
#     from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig
#     from torch.ao.quantization import get_default_qconfig, prepare, convert
    
#     # 模型準備
#     torch.backends.quantized.engine = 'qnnpack'
#     model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
#     # 指定嵌入層的量化配置
#     model.embeddings.qconfig = float_qparams_weight_only_qconfig
#     model_prepared = prepare(model)
#     calibrate(model_prepared, get_calibrattion_data(runner, text))
#     quantized_model = convert(model_prepared)
    
#     os.makedirs('work_dirs', exist_ok=True)
#     save_onnx_path = os.path.join(
#         'work_dirs', 'yolow-l-quantized.onnx')
#     # export onnx
#     with BytesIO() as f:
#         # output_names = ['num_dets', 'boxes', 'scores', 'labels']
#         output_names = ['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes']
#         # output_names = ['scores','boxes']
#         torch.onnx.export(
#             quantized_model,
#             fake_input,
#             f,
#             input_names=['images'],
#             output_names=output_names,
#             opset_version=13)
#         f.seek(0)
#         onnx_model = onnx.load(f)
#         onnx.checker.check_model(onnx_model)
#     onnx_model, check = onnxsim.simplify(onnx_model)
#     onnx.save(onnx_model, save_onnx_path)


    
    
def generate_calibration_data(runner, text:str):
    root = "/home/yptsai/program/object_detection/YOLO-World/data/coco/val2017/"
    image_list = os.listdir(root)
    image_list = [os.path.join(root, f) for f in image_list]
    random.shuffle(image_list)
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    img_datas = []
    # device = runner.model.device
    img_uint8 = []
    for idx, file in enumerate(image_list[:500]):
        
        data_info = dict(img_id=0, img_path=file, texts=texts)
        data_info = runner.pipeline(data_info)
        img_uint8.append(data_info['inputs'].unsqueeze(0).permute(0,2,3,1).cpu().numpy()) #(1,3,640,640, dtype=uint8)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0), #.to(device=device),
                        data_samples=[data_info['data_samples']])
        fake_input_2 = runner.model.data_preprocessor(data_batch,False)
        img_datas.append(fake_input_2['inputs'].permute(0,2,3,1).cpu().numpy())

    calib_datas = np.vstack(img_datas)
    calib_uint8 = np.vstack(img_uint8)
    print(f'calib_datas.shape: {calib_datas.shape}')
    np.save(file='tflite_calibration_data_500_images_640_1x640x640x3.npy', arr=calib_datas)
    np.save(file='tflite_calibration_data_500_images_640_uint8_1x640x640x3.npy', arr=calib_uint8)
    
    
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def preprocess_with_cv2_only(image_path, target_size=(640, 640), pad_value=114, convert_to_rgb=True):
    """
    使用 OpenCV 完成数据预处理。

    Args:
        image_path (str): 输入图像的文件路径。
        target_size (tuple): 目标尺寸 (width, height)。
        pad_value (int): 填充值，默认为 114。
        convert_to_rgb (bool): 是否将 BGR 转换为 RGB，默认 True。

    Returns:
        torch.Tensor: 处理后的张量，形状为 (C, H, W)。
        tuple: 原始图像的大小 (width, height)。
        tuple: 缩放比例 (scale_w, scale_h)。
    """
    # 1. 使用 OpenCV 加载图像 (默认 BGR 格式)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    
    original_height, original_width = img.shape[:2]
    target_width, target_height = target_size

    # 2. 是否转换为 RGB 格式
    if convert_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. 等比例缩放
    scale = min(target_width / original_width, target_height / original_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # 4. 填充到目标大小
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    padded_img = cv2.copyMakeBorder(
        resized_img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
    )

    # 5. 转换为 PyTorch 张量
    img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)

    return img_tensor

g_scores =None
g_scores2 = None
g_confidence = None
def run_image(runner,
              image,
              text,
              max_num_boxes,
              score_thr,
              nms_thr,
              image_path='./work_dirs/demo.png'):
    os.makedirs('./work_dirs', exist_ok=True)
    image.save(image_path)
    
    # generate_calibration_data(runner=runner, text=text)
    
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    img = preprocess_with_cv2_only(image_path,convert_to_rgb=False)
    data_info = runner.pipeline(data_info)
    
    img2= data_info['inputs']
    # 將 PyTorch 張量轉換為 NumPy 陣列（如果需要）
    diff = torch.abs(img - img2).cpu().numpy()

    # 找到最大值和對應的索引
    max_value = np.max(diff)
    max_position = np.unravel_index(np.argmax(diff), diff.shape)

    print(f"最大差異值: {max_value}")
    print(f"最大差異位置: {max_position}")
    
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    global g_scores
    g_scores=pred_instances
    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    global g_scores2
    g_scores2 = pred_instances
    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    global g_confidence
    g_confidence = detections
    
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def export_model(runner,
                 checkpoint,
                 image,
                 text,
                 max_num_boxes,
                 score_thr,
                 nms_thr):
    backend = MMYOLOBackend.ONNXRUNTIME
    postprocess_cfg = ConfigDict(
        pre_top_k=10 * max_num_boxes,
        keep_top_k=max_num_boxes,
        iou_threshold=nms_thr,
        score_threshold=score_thr,
        multi_label=True)

    base_model = deepcopy(runner.model)
    texts = [[t.strip() for t in text.split(',')] + [' ']]
    base_model.reparameterize(texts)
    deploy_model = DeployModel(
        baseModel=base_model,
        backend=backend,
        # postprocess_cfg=postprocess_cfg,
        postprocess_cfg=None,
        without_bbox_decoder=True)
    deploy_model.eval()

    device = (next(iter(base_model.parameters()))).device
    fake_input = torch.ones([1, 3, 640, 640], device=device)
    # dry run
    image_path='./work_dirs/demo.png'
    fake_input = preprocess_with_cv2_only(image_path,convert_to_rgb=False).unsqueeze(0).to(dtype=torch.float32,device=device)
    fake_input_cv2 = fake_input[:, [2, 1, 0], ...]/255.0
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    label_file = './work_dirs/label.txt'
    with open(label_file,"w") as fp:
        for l in texts:
            for k in l:
                fp.write(k+"\n")
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = runner.pipeline(data_info)
    torch.save(data_info,"data_input_uint8.pt")
    fake_input_pipeline=data_info['inputs'].unsqueeze(0).to(device=device)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0).to(device=device),
                      data_samples=[data_info['data_samples']])
    fake_input_2 = runner.model.data_preprocessor(data_batch,False)
    torch.save(fake_input_2, "data_input_tensor.pt")
    # fake_input = fake_input_2['inputs'].permute(0,2,3,1)
    fake_input = fake_input_pipeline[:1,[2,1,0],...].permute(0,2,3,1).to(dtype=torch.float32)
    
    results=deploy_model(fake_input)
    
    batch_img_metas = [
            data_samples.metainfo for data_samples in [data_info['data_samples']]
        ]
    
    tensor3 = fake_input_2['inputs'].cpu()
    tensor4 = (fake_input_pipeline[:1,[2,1,0],...]/255.0).cpu()
    # # 计算差异 (绝对值)
    # difference = torch.abs(tensor1 - tensor2)
    difference = torch.abs(tensor3 - tensor4)
    # # 找到差异最大的 5 个值及其索引
    top_k_values, top_k_indices = torch.topk(difference.flatten(), k=5)

    # # 转换为二维索引 (行, 列) 使用 numpy.unravel_index
    top_k_indices_2d = np.unravel_index(top_k_indices.cpu().numpy(), difference.shape)

    # # 打印差异最大的 5 个值及其位置
    for i in range(5):
        idx = tuple([ii[i] for ii in top_k_indices_2d])
        # row, col = row, col = int(top_k_indices_2d[0][i]), int(top_k_indices_2d[1][i])
        print(f"Difference: {top_k_values[i]} at position ({idx[-2]}, {idx[-1]})")
        # print(f"Tensor1 Value: {tensor1[row, col]}, Tensor2 Value: {tensor2[row, col]}")
        print(f"Tensor3 Value: {tensor3[idx]}, Tensor4 Value: {tensor4[idx]}")

    # predict_head_predict_forword = torch.load("predict_head_predict_forword.pt")
    # runner_pred_by_feat = torch.load("predict_head_predict_by_feat.pt")
    # runner_before_deocde = torch.load("runner_before_deocde.pt")
    # predictions = runner.model.bbox_head.predict_by_feat([results[0][:1,:6,...],results[1][:1,:6,...],results[2][:1,:6,...]],[results[0][:1,6:,...],results[1][:1,6:,...],results[2][:1,6:,...]],None,batch_img_metas)
    # permute_scores = results[0].permute(0,2,1)
    # scores_level_80x80 = permute_scores[:,:,:80*80].reshape(1,-1,80,80)
    # scores_level_40x40 = permute_scores[:,:,80*80:80*80+40*40].reshape(1,-1,40,40)
    # scores_level_20x20 = permute_scores[:,:,80*80+40*40:80*80+40*40+20*20].reshape(1,-1,20,20)
    
    # permute_bboxes = results[1].permute(0,2,1)
    # bboxes_level_80x80 = permute_bboxes[:,:,:80*80].reshape(1,-1,80,80)
    # bboxes_level_40x40 = permute_bboxes[:,:,80*80:80*80+40*40].reshape(1,-1,40,40)
    # bboxes_level_20x20 = permute_bboxes[:,:,80*80+40*40:80*80+40*40+20*20].reshape(1,-1,20,20)
    
    # predictions = runner.model.bbox_head.predict_by_feat([scores_level_80x80,scores_level_40x40,scores_level_20x20],[bboxes_level_80x80,bboxes_level_40x40,bboxes_level_20x20],None,batch_img_metas)
    
    # pred_instances = predictions[0]
    # # print(f'topk:\nscores={results[2]}\nlabels={results[3]}\nindex={results[4]}')
    # # global g_scores
    # # g_scores=pred_instances
    # keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    # pred_instances = pred_instances[keep]
    # pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    # # global g_scores2
    # # g_scores2 = pred_instances
    # if len(pred_instances.scores) > max_num_boxes:
    #     indices = pred_instances.scores.float().topk(max_num_boxes)[1]
    #     pred_instances = pred_instances[indices]

    # pred_instances = pred_instances.cpu().numpy()
    # detections = sv.Detections(
    #     xyxy=pred_instances['bboxes'],
    #     class_id=pred_instances['labels'],
    #     confidence=pred_instances['scores']
    # )
    # # global g_confidence
    # # g_confidence = detections
    
    # labels = [
    #     f"{texts[class_id][0]} {confidence:0.2f}"
    #     for class_id, confidence
    #     in zip(detections.class_id, detections.confidence)
    # ]

    # image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    # image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)
    #########
    # # # load_data = torch.load("result.pt")
    # # yolow_predict = torch.load("predict.pt")    
    # # predict_head_predict_features = torch.load( "predict_head_predict_features.pt")
    # predict_head_predict_forword = torch.load("predict_head_predict_forword.pt")
    # # predict_head_predict = torch.load("predict_head_predict.pt")
    
    # runner_pred_by_feat = torch.load("predict_head_predict_by_feat.pt")
    # # deploy_pred_by_feat = torch.load("deploy_predict_by_feat.pt")
    
    # runner_before_deocde = torch.load("runner_before_deocde.pt")
    # deploy_before_deocde = torch.load("deploy_before_deocde.pt")
    
    # yolow_forward = results
    # forward_head_forward_features = torch.load('forward_head_forward_features.pt')
    # forward_head_forward = torch.load('forward_head_forward.pt')
    
    # # feat_extact2 = torch.load("features2.pt")
    # # head_results = torch.load("head_result.pt")
    
    # # tensor1 = load_data['scores'].cpu()
    # # tensor2 = results[0].cpu()
    # # tensor3 = load_data['bbox'].cpu()
    # # tensor4 = results[1].cpu()
    # tensor3 = yolow_predict['input'].cpu()
    # tensor4 = yolow_forward['input'].cpu()
    # # 计算差异 (绝对值)
    # # difference = torch.abs(tensor1 - tensor2)
    # difference = torch.abs(tensor3 - tensor4)
    # # 找到差异最大的 5 个值及其索引
    # top_k_values, top_k_indices = torch.topk(difference.flatten(), k=5)

    # # 转换为二维索引 (行, 列) 使用 numpy.unravel_index
    # top_k_indices_2d = np.unravel_index(top_k_indices.cpu().numpy(), difference.shape)

    # # 打印差异最大的 5 个值及其位置
    # for i in range(5):
    #     idx = tuple([ii[i] for ii in top_k_indices_2d])
    #     # row, col = row, col = int(top_k_indices_2d[0][i]), int(top_k_indices_2d[1][i])
    #     print(f"Difference: {top_k_values[i]} at position ({idx[-2]}, {idx[-1]})")
    #     # print(f"Tensor1 Value: {tensor1[row, col]}, Tensor2 Value: {tensor2[row, col]}")
    #     print(f"Tensor3 Value: {tensor3[idx]}, Tensor4 Value: {tensor4[idx]}")

    #########

    #########yptsai #############################
    # calibration_and_tflite(runner=runner,model=deploy_model, text=text, fake_input=fake_input_2['inputs'])
    # result, dla_file, tflite_file = mtk_calibration_and_export_tflite(runner=runner,model=deploy_model, text=text, test_input=fake_input)
    # result, dla_file, tflite_file = mtk_QAT_and_export_tflite(runner=runner,model=deploy_model, text=text, test_input=fake_input)
    result, dla_file, tflite_file = mtk_qat(runner=runner,model=deploy_model, text=text, test_input=fake_input)
    #############################################
    # os.makedirs('work_dirs', exist_ok=True)
    # save_onnx_path = os.path.join(
    #     'work_dirs', 'yolow-l.onnx')
    # # export onnx
    # with BytesIO() as f:
    #     # output_names = ['num_dets', 'boxes', 'scores', 'labels']
    #     output_names = ['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes']
    #     # output_names = ['scores','boxes']
    #     torch.onnx.export(
    #         deploy_model,
    #         fake_input,
    #         f,
    #         input_names=['images'],
    #         output_names=output_names,
    #         opset_version=13)
    #     f.seek(0)
    #     onnx_model = onnx.load(f)
    #     onnx.checker.check_model(onnx_model)
    # onnx_model, check = onnxsim.simplify(onnx_model)
    # onnx.save(onnx_model, save_onnx_path)
    
    
    # # import netron
    # # netron.start(save_onnx_path)
    
    # # #################
    # # graph = onnx_model.graph

    # # # 修改輸出名稱
    # # for output in graph.output:
    # #     print(output.name)
    # # # 遍历节点，找到 TopK 操作
    # # for node in graph.node:
    # #     if node.op_type == "TopK":
    # #         print(f"Found TopK Node: {node.name}")
            
    # #         # 确定 TopK 的第二个输入（k 值）的名字
    # #         k_input_name = node.input[1]
    # #         print(f"K input name: {k_input_name}")
            
    # #         indices_output_name = node.output[1]
    # #         print(f"Indices output name: {indices_output_name}")
    # #         # 遍历初始化器，找到对应的 k
    # #         for initializer in graph.initializer:
    # #             if initializer.name == k_input_name:
    # #                 print(f"Found initializer for K: {initializer.name}")
                    
    # #                 # 转换为 NumPy 数组
    # #                 k_array = to_array(initializer)
                    
    # #                 # 检查 k 是否是形状为 [1] 的张量
    # #                 if k_array.shape == (1,):
    # #                     print(f"Original K value: {k_array}")

    # #                     # 修改为形状为 [] 的标量
    # #                     k_scalar = k_array.item()  # 提取标量值
    # #                     new_initializer = from_array(
    # #                         np.array(k_scalar, dtype=k_array.dtype).reshape(()),
    # #                         name=initializer.name
    # #                     )                        
    # #                     # 替换原来的初始化器
    # #                     graph.initializer.remove(initializer)
    # #                     graph.initializer.append(new_initializer)
    # #                     print(f"Modified K to scalar: {k_scalar}")
    # #             # elif initializer.name == indices_output_name:
    # #             #     nitializer_found = True
    # #             #     print(f"Modifying initializer {initializer.name}")
                    
    # #             #     # 提取數據並轉換為 uint8
    # #             #     original_data = to_array(initializer)
    # #             #     converted_data = original_data.astype(np.uint8)
                    
    # #             #     # 創建新的初始化器
    # #             #     new_initializer = from_array(
    # #             #         converted_data, name=initializer.name)
                    
    # #             #     # 替換舊的初始化器
    # #             #     graph.initializer.remove(initializer)
    # #             #     graph.initializer.append(new_initializer)
                    
    # #         # 修改形状信息
    # #         for value_info in graph.value_info:
    # #             if value_info.name == k_input_name:
    # #                 print(f"Before modification: {value_info}")
    # #                 value_info.type.tensor_type.shape.dim.clear()  # 清空形状，表示标量
    # #                 value_info.type.tensor_type.elem_type = TensorProto.INT32  # 设置数据类型
    # #                 print(f"After modification: {value_info}")
    # #             # elif value_info.name == indices_output_name:
    # #             #     print(f"Before modification: {value_info}")
    # #             #     # value_info.type.tensor_type.shape.dim.clear()  # 清空形状，表示标量
    # #             #     value_info.type.tensor_type.elem_type = TensorProto.UINT8  # 设置数据类型
    # #                 # print(f"After modification: {value_info}")
            
            

    # # 验证模型
    # # onnx.checker.check_model(onnx_model)

    # # 保存模型
    # # modified_filename = "modified_model.onnx"
    # # onnx.save(onnx_model, modified_filename)
    # # print("Model saved as modified_model.onnx")
    # # netron.start(modified_filename)
    # #################
    
    
    # from onnxruntime import InferenceSession
    # sess = InferenceSession(save_onnx_path)
    # # 检查当前使用的执行提供程序
    # print("Available providers:", sess.get_providers())
    # outs = sess.run(['scores','boxes','topk_scores', 'topk_classes','topk_indices','topk_bboxes'], {"images": fake_input_2['inputs'].cpu().numpy().astype(np.float32)})
    # print(outs)
    
    del base_model
    del deploy_model
    # del onnx_model
    return gr.update(visible=True), dla_file, tflite_file, label_file, image


def demo(runner, args, cfg):
    
    with gr.Blocks(title="YOLO-World") as demo:
        with gr.Row():
            gr.Markdown('<h1><center>YOLO-World: Real-Time Open-Vocabulary '
                        'Object Detector</center></h1>')
        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Row():
                    image = gr.Image(type='pil', label='input image')
                input_text = gr.Textbox(
                    lines=7,
                    label='Enter the classes to be detected, '
                          'separated by comma',
                    value=', '.join(CocoDataset.METAINFO['classes']),
                    elem_id='textbox')
                with gr.Row():
                    submit = gr.Button('Detect')
                    clear = gr.Button('Clear')
                with gr.Row():
                    export = gr.Button('Deploy and Export Model')
                out_download = gr.File(
                    label='Download DLA link',
                    visible=True,
                    height=30,
                    interactive=False)
                out_download_2 = gr.File(
                    label='Download TfLite Model',
                    visible=True,
                    height=30,
                    interactive=False
                )
                out_download_3 = gr.File(
                    label='Download class Label',
                    visible=True,
                    height=30,
                    interactive=False
                )
                max_num_boxes = gr.Slider(
                    minimum=1,
                    maximum=300,
                    value=100,
                    step=1,
                    interactive=True,
                    label='Maximum Number Boxes')
                score_thr = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.05,
                    step=0.001,
                    interactive=True,
                    label='Score Threshold')
                nms_thr = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.001,
                    interactive=True,
                    label='NMS Threshold')
            with gr.Column(scale=0.7):
                output_image = gr.Image(
                    type='pil',
                    label='output image')

        submit.click(partial(run_image, runner),
                     [image, input_text, max_num_boxes,
                      score_thr, nms_thr],
                     [output_image])
        clear.click(lambda: [[], '', ''], None,
                    [image, input_text, output_image])
        export.click(partial(export_model, runner, args.checkpoint),
                     [image, input_text, max_num_boxes, score_thr, nms_thr],
                     [out_download, out_download, out_download_2, out_download_3, output_image])
        
        demo.launch(server_name='0.0.0.0')


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    os.makedirs('./work_dirs', exist_ok=True)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    demo(runner, args, cfg)
