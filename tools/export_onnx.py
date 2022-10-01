#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import sys
import random

from loguru import logger

import torch
from torch import nn

sys.path.append("./")

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


class ORT_NMS(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        max_output_boxes_per_class=torch.tensor([100]),
        iou_threshold=torch.tensor([0.45]),
        score_threshold=torch.tensor([0.25]),
    ):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )


class TRT_NMS(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32
        )
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        out = g.op(
            "TRT::EfficientNMS_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4,
        )
        det_nums, det_boxes, det_scores, det_classes = out
        return det_nums, det_boxes, det_scores, det_classes


class TRT_NMS2(torch.autograd.Function):
    """TensorRT NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        shareLocation=1,
        numClasses=80,
        backgroundLabelId=-1,
        topK=100,
        keepTopK=100,
        scoreThreshold=0.35,
        iouThreshold=0.65,
        isNormalized=0,
        clipBoxes=0,
        scoreBits=16,
        caffeSemantics=0,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, keepTopK, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, keepTopK, 4)
        det_scores = torch.randn(batch_size, keepTopK)
        det_classes = torch.randint(
            0, num_classes, (batch_size, keepTopK), dtype=torch.int32
        ).float()
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        shareLocation=1,
        numClasses=80,
        backgroundLabelId=-1,
        topK=100,
        keepTopK=100,
        scoreThreshold=0.35,
        iouThreshold=0.65,
        isNormalized=0,
        clipBoxes=0,
        scoreBits=16,
        caffeSemantics=0,
    ):
        out = g.op(
            "TRT::BatchedNMSDynamic_TRT",
            boxes,
            scores,
            shareLocation_i=shareLocation,
            numClasses_i=numClasses,
            backgroundLabelId_i=backgroundLabelId,
            topK_i=topK,
            keepTopK_i=keepTopK,
            scoreThreshold_f=scoreThreshold,
            iouThreshold_f=iouThreshold,
            isNormalized_i=isNormalized,
            clipBoxes_i=clipBoxes,
            scoreBits_i=scoreBits,
            caffeSemantics_i=caffeSemantics,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(
        self,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=640,
        device=None,
        n_classes=80,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh  # if max_wh != 0 : non-agnostic else : agnostic
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=self.device,
        )
        self.n_classes = n_classes

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        if self.n_classes == 1:
            scores = conf  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            scores *= conf  # conf = obj_conf * cls_conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(
            nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold
        )
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.cat([X, selected_boxes, selected_categories, selected_scores], 1)


class ONNX_TRT(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(
        self,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        n_classes=80,
        type_nms=0,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert type_nms in [0, 1]
        self.device = device if device else torch.device("cpu")
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = "1"
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes = n_classes

        self.type_nms = type_nms

        if self.type_nms == 1:
            self.convert_matrix = torch.tensor(
                [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                dtype=torch.float32,
                device=self.device,
            )

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        if self.n_classes == 1:
            scores = conf  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            scores *= conf  # conf = obj_conf * cls_conf

        if self.type_nms == 0:
            num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(
                boxes,
                scores,
                self.background_class,
                self.box_coding,
                self.iou_threshold,
                self.max_obj,
                self.plugin_version,
                self.score_activation,
                self.score_threshold,
            )
        else:
            boxes @= self.convert_matrix
            boxes = boxes.unsqueeze(dim=2)
            num_det, det_boxes, det_scores, det_classes = TRT_NMS2.apply(
                boxes,
                scores,
                1,
                self.n_classes,
                -1,
                self.max_obj,
                self.max_obj,
                self.score_threshold,
                self.iou_threshold,
                0,
                0,
                16,
                0,
            )

        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(
        self,
        model,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        device=None,
        n_classes=80,
        type_nms=0,
        trt=False,
    ):
        super().__init__()
        device = device if device else torch.device("cpu")
        assert isinstance(max_wh, (int)) or max_wh is None
        self.model = model.to(device)
        self.patch_model = ONNX_TRT if trt else ONNX_ORT
        self.end2end = self.patch_model(
            max_obj=max_obj,
            iou_thres=iou_thres,
            score_thres=score_thres,
            max_wh=max_wh,
            device=device,
            n_classes=n_classes,
            type_nms=type_nms,
        )
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument("--input", default="images", type=str, help="input node name of onnx model")
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument("-o", "--opset", default=14, type=int, help="onnx opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--topk-all", type=int, default=100, help="ONNX NMS: topk for all classes to keep"
    )
    parser.add_argument("--iou-thres", type=float, default=0.45, help="ONNX NMS: IoU threshold")
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="ONNX NMS: confidence threshold"
    )
    parser.add_argument("--end2end", action="store_true", help="ONNX: add NMS to model")
    parser.add_argument("--trt", action="store_true", help="ONNX: use to export trt")
    parser.add_argument(
        "--type-nms",
        type=int,
        default=0,
        help="TensorRT: EfficientNMS (type-nms=0) or BatchedNMS(type-nms=1)",
    )
    parser.add_argument(
        "--dynamic-batch", action="store_true", help="ONNX/TF/TensorRT: dynamic batch"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="ONNX: using onnx_graphsurgeon to cleanup"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    # model.head.decode_in_inference = False
    model.head.end2end_in_inference = True

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    if args.end2end:
        if args.trt:
            output_names = ["num_dets", "det_boxes", "det_scores", "det_classes"]
        else:
            output_names = ["output"]
    else:
        output_names = ["output"]

    dynamic_axes = None

    if args.dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        dynamic_axes["output"] = {0: "batch", 1: "anchors"}

    if args.dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}}

        if args.end2end:
            if args.trt:
                dynamic_axes.update(
                    {
                        "num_dets": {0: "batch"},
                        "det_boxes": {0: "batch"},
                        "det_scores": {0: "batch"},
                        "det_classes": {0: "batch"},
                    }
                )
            else:
                dynamic_axes["output"] = {0: "batch"}
        else:
            dynamic_axes["output"] = {0: "batch"}

    if args.end2end:
        model = End2End(
            model=model,
            max_obj=args.topk_all,
            iou_thres=args.iou_thres,
            score_thres=args.conf_thres,
            max_wh=max(dummy_input.shape[2:]),
            device=torch.device("cpu"),
            n_classes=model.head.num_classes,
            type_nms=args.type_nms,
            trt=args.trt,
        )

    torch.onnx.export(
        model,
        dummy_input,
        args.output_name,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if args.simplify:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))

    if args.cleanup:
        try:
            logger.info("Starting to cleanup ONNX using onnx_graphsurgeon...")
            import onnx_graphsurgeon as gs

            graph = gs.import_onnx(onnx_model)
            graph = graph.cleanup().toposort()
            onnx_model = gs.export_onnx(graph)
            onnx.save(onnx_model, args.output_name)
            logger.info("generated simplified onnx model named {}".format(args.output_name))
        except Exception as e:
            print(f"Cleanup failure: {e}")


if __name__ == "__main__":
    main()
