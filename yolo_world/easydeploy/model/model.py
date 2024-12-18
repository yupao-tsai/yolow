# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmdet.models.backbones.csp_darknet import Focus
from mmdet.models.layers import ChannelAttention
from mmdet.models.utils import (
    multi_apply,
    unpack_gt_instances,
    filter_scores_and_topk)
from mmengine.config import ConfigDict
from torch import Tensor

from mmyolo.models import RepVGGBlock
from mmyolo.models.dense_heads import (PPYOLOEHead, RTMDetHead, YOLOv5Head,
                                       YOLOv7Head, YOLOv8Head, YOLOXHead)
from mmyolo.models.layers import ImplicitA, ImplicitM
from ..backbone import DeployFocus, GConvFocus, NcnnFocus
from ..bbox_code import (rtmdet_bbox_decoder, yolov5_bbox_decoder,
                         yolox_bbox_decoder)
from ..nms import batched_nms, efficient_nms, onnx_nms
from .backend import MMYOLOBackend


class DeployModel(nn.Module):
    transpose = False

    def __init__(self,
                 baseModel: nn.Module,
                 backend: MMYOLOBackend,
                 postprocess_cfg: Optional[ConfigDict] = None, 
                 without_bbox_decoder=False):
        super().__init__()
        self.baseModel = baseModel
        self.baseHead = baseModel.bbox_head
        self.backend = backend
        self.features = None
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.without_bbox_decoder = without_bbox_decoder
        self.data_norm = torch.tensor([[[[255, 255, 255]]]], dtype=torch.float32)
        

        if postprocess_cfg is None:
            self.with_postprocess = False
        else:
            self.with_postprocess = True
            self.__init_sub_attributes()
            self.detector_type = type(self.baseHead)
            self.pre_top_k = postprocess_cfg.get('pre_top_k', 1000)
            self.keep_top_k = postprocess_cfg.get('keep_top_k', 100)
            self.iou_threshold = postprocess_cfg.get('iou_threshold', 0.65)
            self.score_threshold = postprocess_cfg.get('score_threshold', 0.25)
        self.__switch_deploy()

    def __init_sub_attributes(self):
        self.bbox_decoder = self.baseHead.bbox_coder.decode
        self.prior_generate = self.baseHead.prior_generator.grid_priors
        self.num_base_priors = self.baseHead.num_base_priors
        self.featmap_strides = self.baseHead.featmap_strides
        self.num_classes = self.baseHead.num_classes

    def __switch_deploy(self):
        headType = type(self.baseHead)
        if not self.with_postprocess:
            if headType in (YOLOv5Head, YOLOv7Head):
                self.baseHead.head_module.forward_single = self.forward_single
            elif headType in (PPYOLOEHead, YOLOv8Head):
                self.baseHead.head_module.reg_max = 0

        if self.backend in (MMYOLOBackend.HORIZONX3, MMYOLOBackend.NCNN,
                            MMYOLOBackend.TORCHSCRIPT):
            self.transpose = True
        for layer in self.baseModel.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, ChannelAttention):
                layer.global_avgpool.forward = self.forward_gvp
            elif isinstance(layer, Focus):
                # onnxruntime openvino tensorrt8 tensorrt7
                if self.backend in (MMYOLOBackend.ONNXRUNTIME,
                                    MMYOLOBackend.OPENVINO,
                                    MMYOLOBackend.TENSORRT8,
                                    MMYOLOBackend.TENSORRT7):
                    self.baseModel.backbone.stem = DeployFocus(layer)
                # ncnn
                elif self.backend == MMYOLOBackend.NCNN:
                    self.baseModel.backbone.stem = NcnnFocus(layer)
                # switch focus to group conv
                else:
                    self.baseModel.backbone.stem = GConvFocus(layer)

    def pred_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     objectnesses: Optional[List[Tensor]] = None,
                     **kwargs):
        assert len(cls_scores) == len(bbox_preds)
        dtype = cls_scores[0].dtype
        device = cls_scores[0].device

        nms_func = self.select_nms()
        if self.detector_type in (YOLOv5Head, YOLOv7Head):
            bbox_decoder = yolov5_bbox_decoder
        elif self.detector_type is RTMDetHead:
            bbox_decoder = rtmdet_bbox_decoder
        elif self.detector_type is YOLOXHead:
            bbox_decoder = yolox_bbox_decoder
        else:
            bbox_decoder = self.bbox_decoder

        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        mlvl_priors = self.prior_generate(
            featmap_sizes, dtype=dtype, device=device)

        flatten_priors = torch.cat(mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size[0] * featmap_size[1] * self.num_base_priors, ),
                stride) for featmap_size, stride in zip(
                    featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        
        ########
        # data = {"flatten_stride":flatten_stride, "flatten_cls_scores":flatten_cls_scores, "flatten_bbox_preds":flatten_bbox_preds, "mlvl_priors":mlvl_priors}
        # torch.save(data, "deploy_predict_by_feat.pt")
        ########
        cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        ########
        # data = {"scores":cls_scores, "bbox":flatten_bbox_preds,"objectnesses":objectnesses}
        # torch.save(data, "deploy_before_deocde.pt")
        # bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds,
        #                       flatten_stride)
        # data = bboxes
        # torch.save(data, "deploy_after_deocde.pt")
        ########
        if objectnesses is not None:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
            cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

        scores = cls_scores

        bboxes = flatten_bbox_preds

        if self.without_bbox_decoder:
            return scores, bboxes
        
        bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds,
                              flatten_stride)

        return nms_func(bboxes, scores, self.keep_top_k, self.iou_threshold,
                        self.score_threshold, self.pre_top_k, self.keep_top_k)

    def select_nms(self):
        if self.backend in (MMYOLOBackend.ONNXRUNTIME, MMYOLOBackend.OPENVINO):
            nms_func = onnx_nms
        elif self.backend == MMYOLOBackend.TENSORRT8:
            nms_func = efficient_nms
        elif self.backend == MMYOLOBackend.TENSORRT7:
            nms_func = batched_nms
        else:
            raise NotImplementedError
        if type(self.baseHead) in (YOLOv5Head, YOLOv7Head, YOLOXHead):
            nms_func = partial(nms_func, box_coding=1)

        return nms_func

    def forward(self, inputs: Tensor):
        
        # if hasattr(inputs, "node") and "device" in inputs.node.meta:
        #     device = inputs.node.meta["device"]
        #     self.data_norm = self.data_norm.to(device)
        # else:
        #     self.data_norm = self.data_norm.to(inputs)
        # print(self.data_norm.device)
        self.data_norm = self.data_norm.to(inputs.device)
        # self.data_norm = self.data_norm.to(device=self.baseHead.device)
        inputs = inputs.div(self.data_norm)
        inputs = self.quant(inputs)
        inputs = inputs.permute(0,3,1,2)
        # inputs = inputs[0, [2,1,0], ...]
        neck_outputs = self.baseModel(inputs)
        # self.features = neck_outputs
        if self.with_postprocess:
            return self.pred_by_feat(*neck_outputs)
        else:
            outputs = []
            if self.transpose:
                for feats in zip(*neck_outputs):
                    if self.backend in (MMYOLOBackend.NCNN,
                                        MMYOLOBackend.TORCHSCRIPT):
                        outputs.append(
                            torch.cat(
                                [feat.permute(0, 2, 3, 1) for feat in feats],
                                -1))
                    else:
                        outputs.append(torch.cat(feats, 1).permute(0, 2, 3, 1))
            else:
                # for feats in zip(*neck_outputs):
                #     outputs.append(torch.cat(feats, 1))
                self.num_classes = self.baseHead.num_classes
                flatten_cls_scores = [
                    cls_score.permute(0, 2, 3, 1).reshape(1, -1,
                                                        self.num_classes)
                    # cls_score.transpose(1,3).transpose(1,2).reshape(1, -1,
                    #                                     self.num_classes)
                    for cls_score in neck_outputs[0]
                ]
                flatten_bbox_preds = [
                    bbox_pred.permute(0, 2, 3, 1).reshape(1, -1, 4)
                    # bbox_pred.transpose(1,3).transpose(1,2).reshape(1, -1, 4)
                    for bbox_pred in neck_outputs[1]
                ]
                
                scores = torch.cat(flatten_cls_scores,1)
                bboxes = torch.cat(flatten_bbox_preds,1)
                score_thr = 0.05
                nms_number = 64
                # K = torch.tensor([1,2,4,8,16,],dtype=torch.uint8)
                K = 64 #torch.tensor(32, dtype=torch.uint8)
                # topk_scores, topk_labels, keep_idxs, _ = filter_scores_and_topk(
                #     scores[0], score_thr, nms_number)
                # values, indices = torch.topk(scores.reshape(-1), k=nms_number, dim=-1, largest=True, sorted=True)
                
                squeeze_scores = scores.squeeze(0)
                squeeze_bboxes = bboxes.squeeze(0)
                # Step 1: 对每个 anchor 的最大分数进行排序（仅在 topk_indices 范围内操作）
                max_scores_per_anchor, max_class_per_anchor = torch.max(scores, dim=-1)  # shape: [1, 8400]
                max_class_per_anchor=max_class_per_anchor.reshape(-1)
                max_scores_per_anchor = torch.clamp(max_scores_per_anchor, min=-8, max=8) 
                # max_class_per_anchor = torch.clamp(max_class_per_anchor, min=0, max=K)
                max_class_per_anchor = max_class_per_anchor.to(dtype=torch.uint8)
                # max_scores_per_anchor = max_scores_per_anchor.sigmoid()
                max_scores_per_anchor2=max_scores_per_anchor.reshape(1,1,1,-1)
                _, topk_indices = max_scores_per_anchor2.topk(K,dim=-1)  # shape: [1, 15]
                topk_indices = topk_indices.reshape(-1)
                topk_scores = max_scores_per_anchor.reshape(-1)[topk_indices].sigmoid()
                topk_classes = max_class_per_anchor[topk_indices]
                topk_bboxes = squeeze_bboxes[topk_indices,:]
                
                # topk_indices=topk_indices.reshape(-1)
                # # Step 2: 仅在 topk_indices 范围内计算 sigmoid 并提取类别
                # # 根据 topk_indices 提取 scores 的子集以减少计算量
                
                # selected_scores = squeeze_scores[topk_indices, :]  # shape: [15, 6]
                # selected_sigmoid_scores = selected_scores.reshape(-1)  # 对子集计算 sigmoid，形状 [15, 6]
                
                # # 根据最大分数的类别索引，提取最终的分数
                # squeeze_classes=max_class_per_anchor.squeeze(0)
                # topk_classes = squeeze_classes[topk_indices]  # shape: [15]
                # new_indices = torch.arange(K).to(device=topk_classes.device, dtype = torch.uint8)*selected_scores.shape[-1]+topk_classes.to(dtype=torch.uint8)
                # new_indices = new_indices.to(dtype = torch.uint8)
                # topk_scores = selected_sigmoid_scores[new_indices]  # shape: [15]
                
                # topk_bboxes = squeeze_bboxes[topk_indices,:]
                # 如果需要扩展维度为 [1, 15]
                # final_topk_scores = final_topk_scores.unsqueeze(0)  # shape: [1, 15]

                # 输出
                # outputs = [scores, bboxes, topk_scores, topk_classes, topk_indices, topk_bboxes]
                # outputs = [self.dequant(topk_scores), self.dequant(topk_classes), self.dequant(topk_indices), self.dequant(topk_bboxes)]
                outputs = [topk_scores, topk_classes, topk_indices, topk_bboxes]
                # outputs = [scores, bboxes, max_scores_per_anchor, max_class_per_anchor, topk_indices]
                # outputs = [scores, bboxes]
            return tuple(outputs)

    @staticmethod
    def forward_single(x: Tensor, convs: nn.Module) -> Tuple[Tensor]:
        if isinstance(convs, nn.Sequential) and any(
                type(m) in (ImplicitA, ImplicitM) for m in convs):
            a, c, m = convs
            aw = a.implicit.clone()
            mw = m.implicit.clone()
            c = deepcopy(c)
            nw, cw, _, _ = c.weight.shape
            na, ca, _, _ = aw.shape
            nm, cm, _, _ = mw.shape
            c.bias = nn.Parameter(c.bias + (
                c.weight.reshape(nw, cw) @ aw.reshape(ca, na)).squeeze(1))
            c.bias = nn.Parameter(c.bias * mw.reshape(cm))
            c.weight = nn.Parameter(c.weight * mw.transpose(0, 1))
            convs = c
        feat = convs(x)
        return (feat, )

    @staticmethod
    def forward_gvp(x: Tensor) -> Tensor:
        return torch.mean(x, [2, 3], keepdim=True)
