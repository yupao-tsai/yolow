# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
import torch


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        print(f'{__file__}:40 - predict')
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)


        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)
        ##########
        # data =results_list
        # torch.save(data, "predict.pt")
        # data = {"image":img_feats, "text":txt_feats}
        # torch.save(data, "predict_head_predict_features.pt")
        ##########
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        self.texts = texts

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        print(f'{__file__}:60 - _forward')
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        
        results = self.bbox_head.forward(img_feats, txt_feats)
        ##########
        # data = {"image":img_feats, "text":txt_feats}
        # torch.save(data, "forward_head_forward_features.pt")
        # data = results
        # torch.save(data, "forward_head_forward.pt")
        ##########
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        print(f'{__file__}:79 - extract_feat')
        if batch_data_samples is None:
            texts = self.texts
        elif isinstance(batch_data_samples, dict):
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        # print(f'type of self.backbone = {type(self.backbone)}, self.mm_neck = {self.mm_neck}')
        img_feats, txt_feats = self.backbone(batch_inputs, texts)
        
        # print(f'type of self.neck = {type(self.neck)}')
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
