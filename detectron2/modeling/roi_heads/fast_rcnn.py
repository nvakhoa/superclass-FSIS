# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
import random
from typing import Tuple, List
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.data import MetadataCatalog, DatasetCatalog
from torch.nn.modules.loss import _WeightedLoss

logger = logging.getLogger(__name__)

ROI_HEADS_OUTPUT_REGISTRY = Registry("ROI_HEADS_OUTPUT")
ROI_HEADS_OUTPUT_REGISTRY.__doc__ = """
Registry for the output layers in ROI heads in a generalized R-CNN model."""

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta,
            model_method = 'default', eval_method = None, eval_gt_classes = None, eval_ways = 1, cosine_scale = -1.0 
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            model_method (str): The method we're using to do inference. Can be metric-averaged,
                'default' etc. See  main-script 'args.method' for the posibilities.
            eval_method (str): The method we're using to do evaluation. Based on a few competing
                papers, we needed to implement different evaluation schemes. Can be 'like_fgn', 
                'like_oneshot' or 'default'fgn' methodfgn' method
            eval_gt_classes (Tensor):
                The ground truth classes in the image as a torch.Tensor.
            eval_ways (int):
                For the `like_fgn' method, specifies how many 'ways' we're emulating
            cosine_scale (float): Cosine scale used in cosine similarity model. -1 means learnable
                and is not yet handled
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.eval_gt_classes = eval_gt_classes 
        self.is_metric_averaging = 'metric-averaged' in model_method 
        self.is_eval_like_fgn = 'like_fgn' in eval_method
        self.eval_ways = eval_ways
        self.cosine_scale = cosine_scale

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.0
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_boxes_for_gt_classes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        predicted_boxes = self._predict_boxes()
        B = self.proposals.tensor.shape[1]
        # If the box head is class-agnostic, then the method is equivalent to `predicted_boxes`.
        if predicted_boxes.shape[1] > B:
            num_pred = len(self.proposals)
            num_classes = predicted_boxes.shape[1] // B
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = torch.clamp(self.gt_classes, 0, num_classes - 1)
            predicted_boxes = predicted_boxes.view(num_pred, num_classes, B)[
                torch.arange(num_pred, dtype=torch.long, device=predicted_boxes.device), gt_classes
            ]
        return predicted_boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()

        if self.is_metric_averaging:
            raise NotImplementedError("NOT USED - Previous experiment on using cosine scale before softmax.")
            assert self.cosine_scale > 0
            # Scores in [-20,20]
            scores = self.pred_class_logits / self.cosine_scale
            # Scores in [-1,1] -> Turn to [0,1] 
            scores = (scores  + 1.0) / 2.0 
            scores = (scores,)
            score_thresh = 0.65
        else:
            scores = self.predict_probs() # Simple Softmax

        # If we have GT information here that means we are in the GTOE evaluation procedure
        # Then we filter the softmax scores based on GT clases to emulate
        # Siamese Mask R-CNN and FGN
        if self.eval_gt_classes is not None:
            scores = self._filter_scores_on_gts(scores) 

        assert len(scores) == 1 
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

    def _filter_scores_on_gts(self, scores : Tuple[torch.Tensor, ...]):
        """
        Filter the scores to only contain gt classes if we've been passed this option 

        NOTE: This code is only executed if we're in LIKE_ONESHOT or LIKE_FGN mode, and gt_classes
        have been passed here. This emulates behaviour of two competing papers, where they do N-way
        evaluation for every class in the images.

        For normal evaluation we DO NOT execute this.
        """
        assert len(scores) == 1 #Only run in inference mode with 1 img per gpu
        gt_classes = self.eval_gt_classes
        if self.is_eval_like_fgn:
            # Emulating FGN goes like this:
            # For every gt class, we do inference with it and N - 1 random other classes.
            # The issues are whether or not the N - 1 classes are necessarily NOT GT or not.

            # The following tries to give an emulation of how that would look like, but since
            # the details are unsure, we only do evaluation for the 1-way 1-shot case.
            # I.e this means we expect self.eval_ways to ALWAYS BE 1 for the experiments we've done.

            picked_classes = set()
            picked_classes.update(self.eval_gt_classes.tolist())
            for _ in range(len(self.eval_gt_classes)): 
                score, *_ = scores # Get only the scores for one image. Should only be 1 anyway
                total_classes = score.shape[-1]
                all_classes = set(range(total_classes))
                non_gt_classes = all_classes - set(self.eval_gt_classes.tolist())

                episode_picks = set(random.choices(tuple(non_gt_classes), k = self.eval_ways - 1))
                picked_classes.update(episode_picks)

            gt_classes = torch.tensor(list(picked_classes), device=self.eval_gt_classes.device, dtype = self.eval_gt_classes.dtype)


        # Only happens if we're in fgn or one-shot eval mode
        # if self.eval_gt_classes is not None and len(self.eval_gt_classes) > 0:
        if self.eval_gt_classes is not None:
            score, *_ = scores
            # All classes != gts are zeroed. This is the GTOE evaluation procedure
            # This allows us to emulate the evaluation method from Siamese Mask R-CNN and FGN
            new_score = torch.zeros_like(score) 
            new_score[:, gt_classes] = score[: , gt_classes]
            new_score[:, -1] = score[: , -1] # Don't forget background class! 
            if len(self.eval_gt_classes) == 0: # NO GT Classes. All preds are BG
                new_score[:, -1] = torch.ones_like(score[: , -1]) # Don't forget background class! 


            scores = (new_score,) # Scores expected to be list per Image, but here we kno
        
        return scores


@ROI_HEADS_OUTPUT_REGISTRY.register()
class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def _cosine_sim_test(self, feature_vector):
        """
            Computes the cosine similarity between the current head and a normalized vector.
            Is not meant for training, instead, use as a comparison between FC head and CosineHead
        """
        feat_vect_norm = torch.norm(feature_vector, 2, dim=0).expand_as(feature_vector)
        feature_vector_normalized = feature_vector.div(feat_vect_norm + 1e-5)

        # TODO(): Can be easily vectorized.
        scores = []
        for idx in range(self.cls_score.weight.shape[0]):
            norm = torch.norm(self.cls_score.weight[idx], p=2, dim=0).expand_as(self.cls_score.weight[idx])
            normalized_weight = self.cls_score.weight[idx].div(norm + 1e-5)
            scores.append(torch.dot(feature_vector_normalized, normalized_weight))

        scores = torch.FloatTensor(scores)

        return scores, scores.argmax()


@ROI_HEADS_OUTPUT_REGISTRY.register()
class CosineSimOutputLayers(nn.Module):
    """
    Two outputs
    (1) proposal-to-detection box regression deltas (the same as
        the FastRCNNOutputLayers)
    (2) classification score is based on cosine_similarity
    """

    def __init__(
            self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(CosineSimOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1, bias=False)
        self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        self.alpha_weighting = None
        self.test_dataset = cfg.DATASETS.TEST[0]
        if 'all' in self.test_dataset and cfg.MODEL.ROI_HEADS.ALPHA_WEIGHTING != 1.0:
            alpha = cfg.MODEL.ROI_HEADS.ALPHA_WEIGHTING
            metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            # Find where the 'novel' keys are in the 'all' model as weight placements.

            base_classes = list(metadata.base_dataset_id_to_contiguous_id.keys())
            base_idx = [metadata.thing_dataset_id_to_contiguous_id[v] for v in base_classes]

            novel_classes = list(metadata.novel_dataset_id_to_contiguous_id.keys())
            novel_idx = [metadata.thing_dataset_id_to_contiguous_id[v] for v in novel_classes]

            self.alpha_weighting = torch.ones(1, num_classes + 1)
            self.alpha_weighting[0, base_classes] = self.alpha_weighting[0,
                                                                         base_classes] * (1 - alpha)
            self.alpha_weighting[0, novel_classes] = self.alpha_weighting[0, base_classes] * alpha


        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        # First dim is now batch size, second dim features ( typically 1024)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = torch.norm(self.cls_score.weight.data,
                               p=2, dim=1).unsqueeze(1).expand_as(
            self.cls_score.weight.data)
        self.cls_score.weight.data = self.cls_score.weight.data.div(temp_norm + 1e-5)
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        # if 'all' in self.test_dataset and not self.training and self.alpha_weighting:
        #     scores = scores * self.alpha_weighting
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas




class SuperClassOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta,
            model_method = 'default', eval_method = None, eval_gt_classes = None, eval_ways = 1, cosine_scale = -1.0 , 
            mapper=None, use_super_cls_acti=True, is_multi_super_cls=False, use_fine_grained_cls_acti=True, use_margin_loss=False, 
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            model_method (str): The method we're using to do inference. Can be metric-averaged,
                'default' etc. See  main-script 'args.method' for the posibilities.
            eval_method (str): The method we're using to do evaluation. Based on a few competing
                papers, we needed to implement different evaluation schemes. Can be 'like_fgn', 
                'like_oneshot' or 'default'fgn' methodfgn' method
            eval_gt_classes (Tensor):
                The ground truth classes in the image as a torch.Tensor.
            eval_ways (int):
                For the `like_fgn' method, specifies how many 'ways' we're emulating
            cosine_scale (float): Cosine scale used in cosine similarity model. -1 means learnable
                and is not yet handled
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]

        self.pred_super_class_logits = pred_class_logits[0]
        self.pred_class_logits = pred_class_logits[1]


        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.eval_gt_classes = eval_gt_classes 
        self.is_metric_averaging = 'metric-averaged' in model_method 
        self.is_eval_like_fgn = 'like_fgn' in eval_method
        self.eval_ways = eval_ways
        self.cosine_scale = cosine_scale

        self.mapper = mapper
        self.use_super_cls_acti = use_super_cls_acti
        self.is_multi_super_cls = is_multi_super_cls
        self.use_fine_grained_cls_acti = use_fine_grained_cls_acti
        self.use_margin_loss = use_margin_loss

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        
    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        # print('gt_classes:', self.gt_classes[:10])
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def superclass_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        
        assert self.mapper
        # print(self.mapper)
        gt_super_class = [self.mapper[int(i)] for i in self.gt_classes]
        # print('gt_super_class:', gt_super_class[:10])
        # assert 2==1
        if not self.is_multi_super_cls:
            gt_super_class = torch.cat(gt_super_class, dim=0).reshape(-1)
            gt_super_class = gt_super_class.cuda()
            return F.cross_entropy(self.pred_super_class_logits, gt_super_class.cuda(), reduction="mean")

        return super_cls_loss(self.pred_super_class_logits, gt_super_class)
    
    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.0
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        loss = {
            "loss_cls": self.softmax_cross_entropy_loss()*0, #*1e1,
            # "loss_cls2": self.softmax_cross_entropy_loss2(),
            "loss_super_cls": self.superclass_cross_entropy_loss()*1e2,
            # "loss_super_cls": self.superclass_cross_entropy_loss2(),
            # "loss_dynamic": self.dynamic_softmax_cross_entropy_loss(),
            # "loss_similar_base": self.Simmilar_base_loss(),
            # "loss_final_score": self.softmax_cross_entropy_loss_final(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

        # if self.use_margin_loss:
        #     class_loss = {"MarginCosine_loss": self.MarginCosine_loss()}
        # else:
        #     class_loss ={"loss_cls": self.softmax_cross_entropy_loss()*0,}

        # loss.update(class_loss)
        return loss

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_boxes_for_gt_classes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        predicted_boxes = self._predict_boxes()
        B = self.proposals.tensor.shape[1]
        # If the box head is class-agnostic, then the method is equivalent to `predicted_boxes`.
        if predicted_boxes.shape[1] > B:
            num_pred = len(self.proposals)
            num_classes = predicted_boxes.shape[1] // B
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = torch.clamp(self.gt_classes, 0, num_classes - 1)
            predicted_boxes = predicted_boxes.view(num_pred, num_classes, B)[
                torch.arange(num_pred, dtype=torch.long, device=predicted_boxes.device), gt_classes
            ]
        return predicted_boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = self.pred_class_logits
        if self.use_fine_grained_cls_acti:
            probs = F.softmax(probs, dim=-1)

        return probs.split(self.num_preds_per_image, dim=0)
    
    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()

        if self.is_metric_averaging:
            raise NotImplementedError("NOT USED - Previous experiment on using cosine scale before softmax.")
            assert self.cosine_scale > 0
            # Scores in [-20,20]
            scores = self.pred_class_logits / self.cosine_scale
            # Scores in [-1,1] -> Turn to [0,1] 
            scores = (scores  + 1.0) / 2.0 
            scores = (scores,)
            score_thresh = 0.65
        else:
            scores = self.predict_probs() # Simple Softmax

        # If we have GT information here that means we are in the GTOE evaluation procedure
        # Then we filter the softmax scores based on GT clases to emulate
        # Siamese Mask R-CNN and FGN
        if self.eval_gt_classes is not None:
            scores = self._filter_scores_on_gts(scores) 

        assert len(scores) == 1 
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

    def _filter_scores_on_gts(self, scores : Tuple[torch.Tensor, ...]):
        """
        Filter the scores to only contain gt classes if we've been passed this option 

        NOTE: This code is only executed if we're in LIKE_ONESHOT or LIKE_FGN mode, and gt_classes
        have been passed here. This emulates behaviour of two competing papers, where they do N-way
        evaluation for every class in the images.

        For normal evaluation we DO NOT execute this.
        """
        assert len(scores) == 1 #Only run in inference mode with 1 img per gpu
        gt_classes = self.eval_gt_classes
        if self.is_eval_like_fgn:
            
            # Emulating FGN goes like this:
            # For every gt class, we do inference with it and N - 1 random other classes.
            # The issues are whether or not the N - 1 classes are necessarily NOT GT or not.

            # The following tries to give an emulation of how that would look like, but since
            # the details are unsure, we only do evaluation for the 1-way 1-shot case.
            # I.e this means we expect self.eval_ways to ALWAYS BE 1 for the experiments we've done.

            picked_classes = set()
            picked_classes.update(self.eval_gt_classes.tolist())
            for _ in range(len(self.eval_gt_classes)): 
                score, *_ = scores # Get only the scores for one image. Should only be 1 anyway
                total_classes = score.shape[-1]
                all_classes = set(range(total_classes))
                non_gt_classes = all_classes - set(self.eval_gt_classes.tolist())

                episode_picks = set(random.choices(tuple(non_gt_classes), k = self.eval_ways - 1))
                picked_classes.update(episode_picks)

            gt_classes = torch.tensor(list(picked_classes), device=self.eval_gt_classes.device, dtype = self.eval_gt_classes.dtype)


        # Only happens if we're in fgn or one-shot eval mode
        # if self.eval_gt_classes is not None and len(self.eval_gt_classes) > 0:
        if self.eval_gt_classes is not None:
            score, *_ = scores
            # print(gt_classes)
            # All classes != gts are zeroed. This is the GTOE evaluation procedure
            # This allows us to emulate the evaluation method from Siamese Mask R-CNN and FGN
            new_score = torch.zeros_like(score) 
            new_score[:, gt_classes] = score[: , gt_classes]
            new_score[:, -1] = score[: , -1] # Don't forget background class! 
            if len(self.eval_gt_classes) == 0: # NO GT Classes. All preds are BG
                new_score[:, -1] = torch.ones_like(score[: , -1]) # Don't forget background class! 

            scores = (new_score,) # Scores expected to be list per Image, but here we kno
        
        return scores

class SuperClass(nn.Module):
    def __init__(self, cfg, num_classes):
        super(SuperClass, self).__init__()
        # self.__init__()
        superclass = cfg.MODEL.ROI_HEADS.SUPERCLASS
        self.is_multi_super_cls = cfg.MODEL.ROI_HEADS.IS_MULTI_SUPER_CLS

        invert = {}
        for sup_id in range(len(superclass[:-1])):
            fine_grained_ids = superclass[sup_id]
            for index, id_ in enumerate(fine_grained_ids[1:]):
                list_id = invert.get(id_, [])
                list_id.append((sup_id, index+1))
                invert[id_] = list_id

        self.invert_superclass = invert
        self.superclass = superclass

        invert_all = {}
        for sup_id in range(len(superclass)):
            fine_grained_ids = superclass[sup_id]
            for index, id_ in enumerate(fine_grained_ids):

                list_id = invert_all.get(id_, [])
                list_id.append((sup_id, index))
                invert_all[id_] = list_id

        self.invert_all_class = invert_all
        self.num_classes = num_classes
        self.margin_loss_weight = cfg.MODEL.ROI_HEADS.MARGIN_LOSS_WEIGHT
        self.similar_base_loss_weight = cfg.MODEL.ROI_HEADS.SIMILAR_BASE_LOSS_WEIGHT
        mapper = {i:[] for i in range(81)}
        for super_cls, child_classes in enumerate(self.superclass):
            for i in child_classes:
                mapper[i].append(super_cls)
        mapper = {key: torch.tensor(val).reshape(1,-1) for key, val in mapper.items()}
        self.mapper= mapper

        logger.info(f'self.mapper: {self.mapper}')
        logger.info('-'*64)
        logger.info(f'superclass: {superclass}')
        logger.info('-'*64)
        logger.info(f'invert_superclass: {self.invert_superclass}')
        logger.info(f'len(self.invert_superclass): {len(self.invert_superclass)}')
        logger.info(f'self.invert_all_class: {self.invert_all_class}')
        logger.info(f'len(self.invert_all_class): {len(self.invert_all_class)}')

@ROI_HEADS_OUTPUT_REGISTRY.register()
class HardSuperClassOutputLayers(SuperClass):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super().__init__(cfg, num_classes)
        # print('num_classes:', num_classes)
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        self.cls_score = nn.Linear(input_size, len(self.superclass))
        self.fg_score = torch.nn.ModuleList([
            nn.Linear(input_size, len(self.superclass[i]))  # 1 bg and 1 base cls
            for i in range(len(self.superclass))
        ])

    
    def losses(self):
        losses = {}
        losses.update({'loss_fine_grained': self.loss_fine_grained()*1e1})
        return losses

    def loss_fine_grained(self, invert_all=True):
        # print(self.gt_classes)

        invert_cls = self.invert_all_class if invert_all else self.invert_superclass

        if isinstance(self.gt_classes, tuple):
            self.gt_classes = torch.cat(self.gt_classes, dim=0)

        loss = torch.tensor(0.0).cuda()
        N = 0
        for gt_cls in self.gt_classes.unique().cpu().numpy().tolist():
            if gt_cls not in invert_cls.keys(): continue
            super_cls_contains_gt_cls = invert_cls[gt_cls]
            indices = torch.where(self.gt_classes==gt_cls)[0]
            # if gt_cls == self.num_classes: continue
            if indices.shape[0]<1: continue
            for super_id, fg_id in super_cls_contains_gt_cls:
                # create label for softmax_cross entropy
                labels = torch.zeros(len(indices), dtype=int) + fg_id 

                
                # print(labels)
                pred_class_logits = self.scores[super_id][indices]

                # caculate loss via cross entropy
                loss += F.cross_entropy(pred_class_logits, labels.cuda(), reduction="sum")
 
        return loss/self.gt_classes.shape[0]

    def forward(self, x, proposals=None):
        if self.training and proposals:
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            self.gt_classes = gt_classes
        
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1) # bz, 1024
        super_cls_scores = self.cls_score(x) # bz x 60 # 1024 x 60


        dim  = super_cls_scores.shape
        num_ins = dim[0]


        fine_grained_cls_scores = torch.zeros((num_ins, self.num_classes + 1), device='cuda')
        
        
        idx = torch.argsort(super_cls_scores, dim=1, descending=True)[..., :1].reshape(-1)
        mask = torch.ones(dim, device='cuda')*(1e-2)
        rows = torch.arange(num_ins).repeat((1, 1)).T.reshape(-1)
        mask[rows, idx] = 1
        result = {key:[] for key in range(self.num_classes + 1)}

        self.scores = []
        for i in range(len(self.fg_score)):
            layer = self.fg_score[i]

            # sc_x = super_cls_scores[...,i, np.newaxis]
            mask_x = mask[...,i, np.newaxis]
    
            mask_x = mask_x.repeat((1, len(self.superclass[i])))

            score = layer(x)

            self.scores.append(score)
            score = torch.nn.Softmax(dim=-1)(score) * mask_x
            
            for index, j in enumerate(self.superclass[i]):
                # print(score[..., index, np.newaxis])
                result[j].append(score[..., index, np.newaxis])
    

        for i in range(len(self.fg_score)):
            for index, j in enumerate(self.superclass[i]):
                score = torch.cat(tuple(result[j]), dim=-1)
                fine_grained_cls_scores[..., j] = score.sum(dim=-1)

        fine_grained_cls_scores = torch.clamp(fine_grained_cls_scores, max=1)
        proposal_deltas = self.bbox_pred(x)

        return (super_cls_scores, fine_grained_cls_scores, self.scores, 0), proposal_deltas
       
@ROI_HEADS_OUTPUT_REGISTRY.register()
class SoftSuperClassOutputLayersNative(HardSuperClassOutputLayers):

    def forward(self, x, proposals=None):
        if self.training and proposals:
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            self.gt_classes = gt_classes
        
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1) # bz, 1024
        super_cls_scores = self.cls_score(x) # bz x 60 # 1024 x 60


        dim  = super_cls_scores.shape
        num_ins = dim[0]


        fine_grained_cls_scores = torch.zeros((num_ins, self.num_classes + 1), device='cuda')
        

        if self.is_multi_super_cls:
            prob_super_cls_scores = torch.nn.Sigmoid()(super_cls_scores)
        else:
            prob_super_cls_scores = torch.nn.Softmax(dim=-1)(super_cls_scores)

        result = {key:[] for key in range(self.num_classes + 1)}

        self.scores = []
        for i in range(len(self.fg_score)):
            layer = self.fg_score[i]
            mask_x = prob_super_cls_scores[...,i, None]
            mask_x = mask_x.repeat((1, len(self.superclass[i])))
            score = layer(x)

            self.scores.append(score)
            score = torch.nn.Softmax(dim=-1)(score) * mask_x
            
            for index, j in enumerate(self.superclass[i]):
                result[j].append(score[..., index, np.newaxis])
    

        for i in range(len(self.fg_score)):
            for index, j in enumerate(self.superclass[i]):
                score = torch.cat(tuple(result[j]), dim=-1)
                fine_grained_cls_scores[..., j] = score.sum(dim=-1)

        fine_grained_cls_scores = torch.clamp(fine_grained_cls_scores, max=1)
        proposal_deltas = self.bbox_pred(x)

        return (super_cls_scores, fine_grained_cls_scores, self.scores, 0), proposal_deltas
     
@ROI_HEADS_OUTPUT_REGISTRY.register()
class SoftSuperClassOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(SoftSuperClassOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        BASE_CLS = 60
        NOVEL_CLS = 20
        NUM_CHILD_CLS = 5
        previous_model = torch.load(cfg.MODEL.WEIGHTS)
        # super-class from 1-shot using cosin bwt p6 features
 
        super_class = cfg.MODEL.ROI_HEADS.SUPER_CLASS
        self.is_multi_super_cls = cfg.MODEL.ROI_HEADS.IS_MULTI_SUPER_CLS
        self.topk = cfg.MODEL.ROI_HEADS.TOP_K
        # print(super_class)
        # assert 2==1
        # self.cls_score = nn.Linear(input_size, BASE_CLS +  1)
        self.cls_score = CosineSimFCLayer(cfg, input_size, len(super_class))
        
        # self.reduction = nn.Linear(input_size, input_size//2, bias=False)

        self.child_cls_score = torch.nn.ModuleList([
            torch.nn.Sequential(
            CosineSimFCLayer(cfg, input_size, len(super_class[i]))

            # CosineSimFCLayer(cfg, input_size + 256*4, len(super_class[i]))
            )
            for i in range(len(super_class))
        ])

        
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        # self.refine_cls = nn.Linear(num_classes+1, num_classes+1, bias=False)
        # self.refine_cls = CosineSimFCLayer(cfg, num_classes+1, num_classes+1)

        invert = {}

        for sup_id in range(len(super_class[:-1])):
            fine_grained_ids = super_class[sup_id]
            for index, id_ in enumerate(fine_grained_ids[1:]):

                list_id = invert.get(id_, [])
                list_id.append((sup_id, index+1))
                invert[id_] = list_id


        self.invert_super_class = invert
        self.super_class = super_class

        invert_all = {}
        for sup_id in range(len(super_class)):
            fine_grained_ids = super_class[sup_id]
            for index, id_ in enumerate(fine_grained_ids):

                list_id = invert.get(id_, [])
                list_id.append((sup_id, index))
                invert_all[id_] = list_id

        self.invert_all_class = invert_all

        self.num_classes = num_classes
        self.pooler = torch.nn.AdaptiveAvgPool2d(1)
        self.margin_loss_weight = cfg.MODEL.ROI_HEADS.MARGIN_LOSS_WEIGHT
        self.similar_base_loss_weight = cfg.MODEL.ROI_HEADS.SIMILAR_BASE_LOSS_WEIGHT
        mapper = {i:[] for i in range(81)}
        for super_class, child_classes in enumerate(self.super_class):
            for i in child_classes:
                mapper[i].append(super_class)
        mapper = {key: torch.tensor(val).reshape(1,-1) for key, val in mapper.items()}
        self.mapper= mapper
        # self.init_class_weight(cfg)
        print('-'*64)
        print('super_class: ', super_class)
        print('-'*64)
        print('invert_super_class: ',self.invert_super_class)
        print(len(self.invert_super_class))
        self.centroids = []
        # for sup_id in range(len(self.super_class)):
        #     fine_grained_ids = super_class[sup_id]

        #     self.centroids.append(nn.Parameter(torch.rand(len(fine_grained_ids), 1024)))

        
        # self.super_head = Super_head()
    
    # def similar_base_loss(self):
    #     loss = None
    #     n = len(self.super_class)
    #     for i in range(n):
    #         w = list(self.child_cls_score[i][0].parameters())[0]
    #         # base - novel weights
    #         dis = (w[0].unsqueeze(0) - w[1:])**2
    #         dis = dis.mean(axis=(-1))
    #         dis = dis.max()
    #         if i ==0:
    #             loss = dis
    #         else: loss+= dis
    #     return loss

    def loss_similar_base(self):
        loss = None
        n = len(self.super_class) - 1 # not apply for base classifier 
        for i in range(n):
            w = list(self.child_cls_score[i][0].parameters())[0]
            # base - novel weights
            dis = (w[0].unsqueeze(0) - w[1:])**2
            dis = dis.mean()
            if i ==0:
                loss = dis
            else: loss+= dis
        return loss

    def loss_similar_novel(self):
        loss = None
        # return {'margin loss': loss}
        n_layer = len(self.child_cls_score)
        for i, (id_fg, list_invert) in enumerate(self.invert_super_class.items()):
            weights = []
            n = len(list_invert)
            for idx in list_invert:
                (index_super, index_fg) = idx
                # w =  self.child_cls_score[index_super][0].weight[index_fg].unsqueeze(0)

                w = (list(self.child_cls_score[index_super][0].parameters())[0])[index_fg].unsqueeze(0)
                weights.append(w)
                # print(self.child_cls_score[index_super][0].weight.shape)
                
            weights = torch.cat(weights)
            # print('weights :', weights.shape)

            weights = (weights.unsqueeze(1) - weights.unsqueeze(0).repeat(n, 1, 1))**2
            # print('dis 1 :', dis.shape)

            weights = weights.mean()

            if i==0:
                loss = weights
            else:
                loss+= weights
        return loss

    def loss_similar_novel2(self, invert_all=True):
        # return self.loss_margin_value

        invert_cls = self.invert_all_class if invert_all else self.invert_super_class

        loss_intra = torch.tensor(0.0).cuda()

        gt_classes = self.gt_classes
        features = self.features
        centroids = []
        margin=5
        # print(self.centroids[0])
        for id_fg in range(self.num_classes):
            list_invert = invert_cls[id_fg]

            for (index_super, index_fg) in list_invert:
                indices = torch.where(gt_classes==id_fg)[0]
                if not (indices.shape[0] > 0) : continue
                # w = (list(self.child_cls_score[index_super].parameters())[0])[index_fg].unsqueeze(0)
                w = self.__get_centroids__(index_super, index_fg, features[indices]).unsqueeze(0)
                # print(w.shape)
                # print(features[indices].shape)
                # print(((w - features[indices])**2).mean(0).sum())
                l =  ((w - features[indices])**2).mean(0)-margin
                
                l = torch.clamp(l, min=0).sum()/features.shape[-1]**0.5
                loss_intra += l

                # l = self.centroids[index_super][index_fg](features[indices])
                # loss_intra += l.sum()/features.shape[-1]**0.5
                centroids.append(w)

        centroids = torch.cat(centroids)
        centroids = (centroids.unsqueeze(1) - centroids.unsqueeze(0).repeat(centroids.shape[0], 1, 1))**2
        centroids = centroids.sum(-1)
        centroids += 1e-2
        centroids[range(centroids.shape[0]), range(centroids.shape[0])] = 1e9
        loss_inter = centroids.min()
        # loss_inter = 1

        return loss_intra*1e-2/loss_inter

    def cal_loss_similar_novel2(self, features, gt_classes, invert_all=True):
        invert_cls = self.invert_all_class if invert_all else self.invert_super_class

        loss_intra = torch.tensor(0.0).cuda()

        # gt_classes = self.gt_classes
        # features = self.features
        centroids = []
        margin = 0.2
        # print(self.centroids[0])
        for id_fg in range(self.num_classes):
            list_invert = invert_cls[id_fg]
            indices = torch.where(gt_classes==id_fg)[0]
            if not (indices.shape[0] > 0) : continue
            # print(indices.shape)
            for (index_super, index_fg) in list_invert:
                # w = (list(self.child_cls_score[index_super].parameters())[0])[index_fg].unsqueeze(0)
                w = self.__get_centroids__(index_super, index_fg, features[indices]).unsqueeze(0)
                # print(w.shape)
                # print(features[indices].shape)
                # print(((w - features[indices])**2).mean(0).sum())
                l =  ((w - features[indices])**2)
                
                l = torch.clamp(l-margin, min=0).mean(0).sum()/features.shape[-1]**0.5
                loss_intra += l

                # l = self.centroids[index_super][index_fg](features[indices])
                # loss_intra += l.sum()/features.shape[-1]**0.5
                centroids.append(w)
        n = len(centroids)
        centroids = torch.cat(centroids)
        # print(centroids.shape)
        # loss_inter = torch.tensor(1e9).cuda()
        # for i in range(n-1):
        #     for j in range(i+1, n, 1):
        #         d = ((centroids[i] - centroids[j])**2).sum(-1)
        #         if d < loss_inter:
        #             loss_inter = d

        # print(n)
        centroids = (centroids.unsqueeze(1) - centroids.unsqueeze(0).repeat(n, 1, 1))**2
        centroids = centroids.sum(-1)
        centroids += 1e-2
        centroids[range(centroids.shape[0]), range(centroids.shape[0])] = 1e9
        loss_inter = centroids.min()
        del centroids
        # loss_inter = 1

        self.loss_margin_value = loss_intra/loss_inter
        # return loss_intra/loss_inter

        # return loss


    def loss_similar_novel22(self, invert_all=True):
        invert_cls = self.invert_all_class if invert_all else self.invert_super_class

        loss_intra = torch.tensor(0.0).cuda()

        gt_classes = self.gt_classes
        features = self.features
        centroids = []
        for id_fg in range(self.num_classes):
            list_invert = invert_cls[id_fg]

            for (index_super, index_fg) in list_invert:
                indices = torch.where(gt_classes==id_fg)[0]
                if not (indices.shape[0] > 0) : continue
                w = (list(self.child_cls_score[index_super].parameters())[0])[index_fg].unsqueeze(0)

                # print(w.shape)
                # print(features[indices].shape)
                # print(((w - features[indices])**2).mean(0).sum())
                loss_intra += ((w - features[indices])**2).mean(0).sum()/features.shape[-1]**0.5
                centroids.append(w)

        centroids = torch.cat(centroids)
        centroids = (centroids.unsqueeze(1) - centroids.unsqueeze(0).repeat(centroids.shape[0], 1, 1))**2
        centroids = centroids.sum(-1)
        centroids += 1e-2
        # print('centroids before: ', centroids)
        centroids[range(centroids.shape[0]), range(centroids.shape[0])] = 1e9
        # print(centroids)
        # print('centroids after: ', centroids)
        loss_inter = centroids.min()
        # print('loss_inter', loss_inter)
        return loss_intra*0.01/(loss_inter)

        # return loss


    def loss_similar_novel3(self, invert_all=True):
        invert_cls = self.invert_all_class if invert_all else self.invert_super_class

        loss = torch.tensor(0.0).cuda()

        gt_classes = self.gt_classes
        features = self.features

        for id_fg in range(self.num_classes):
            list_invert = invert_cls[id_fg]

            for (index_super, index_fg) in list_invert:
                indices = torch.where(gt_classes==id_fg)[0]
                if not (indices.shape[0] > 0) : continue
                w = (list(self.child_cls_score[index_super].parameters())[0])[index_fg].unsqueeze(0)

                # print(w.shape)
                # print(features[indices].shape)
                # print(((w - features[indices])**2).mean(0).sum())
                loss += ((w - features[indices])**2).mean(0).sum()/features.shape[-1]**0.5
        return loss

        # # return {'margin loss': loss}
        # n_layer = len(self.child_cls_score)
        # for i, (id_fg, list_invert) in enumerate(self.invert_super_class.items()):
        #     weights = []
        #     n = len(list_invert)
        #     for idx in list_invert:
        #         (index_super, index_fg) = idx
        #         # w =  self.child_cls_score[index_super][0].weight[index_fg].unsqueeze(0)
        #         w = (list(self.child_cls_score[index_super][0].parameters())[0])[index_fg].unsqueeze(0)
        #         weights.append(w)
        #         # print(self.child_cls_score[index_super][0].weight.shape)
                
        #     weights = torch.cat(weights)
        #     # print('weights :', weights.shape)

        #     weights = (weights.unsqueeze(1) - weights.unsqueeze(0).repeat(n, 1, 1))**2
        #     # print('dis 1 :', dis.shape)

        #     weights = weights.mean()

        #     if i==0:
        #         loss = weights
        #     else:
        #         loss+= weights
        return loss

    def loss_similar_base_via_score(self, gt_class):
        loss = None
        n = len(self.super_class) - 1
        for i in range(n):
            w = self.scores[i]
            # base - novel weights
            dis = ((w[0] - w[1:])**2).mean()
            if i ==0:
                loss = dis
            else: loss+= dis
        return loss/n

    # def loss_similar_novel_via_score(self, gt_class):
    #     loss = None
    #     n = len(self.super_class) - 1
    #     for i in range(n):
    #         w = self.scores[i]
    #         # base - novel weights
    #         dis = ((w[0] - w[1:])**2).mean()
    #         if i ==0:
    #             loss = dis
    #         else: loss+= dis
    #     return loss/n

    def loss_fine_grained(self, invert_all=True):
        # print(self.gt_classes)

        invert_cls = self.invert_all_class if invert_all else self.invert_super_class
        if isinstance(self.gt_classes, tuple):
            self.gt_classes = torch.cat(self.gt_classes, dim=0)

        loss = torch.tensor(0.0).cuda()
        for gt_cls in self.gt_classes.unique().cpu().numpy().tolist():
            if gt_cls not in invert_cls.keys(): continue
            super_cls_contains_gt_cls = invert_cls[gt_cls]
            indices = torch.where(self.gt_classes==gt_cls)[0]
            for super_id, fg_id in super_cls_contains_gt_cls:
                # create label for softmax_cross entropy
                labels = torch.zeros(len(indices), dtype=int) + fg_id
                pred_class_logits = self.scores[super_id][indices]

                # caculate loss via cross entropy
                loss += F.cross_entropy(pred_class_logits, labels.cuda(), reduction="mean")/len(indices)


            # loss /= len(indices)
        # print('final_loss: ', loss)
        return loss
    
    def get_indices_via_list(self, listid, id):
        return [i for i, sub_list in enumerate(listid) if id in sub_list]

    def loss_margin(self, classid_list, interest_classid, fetures):
        loss_intra = torch.tensor(0.0).cuda()
        centroids =  []
        for cls_id in range(len(self.budget['fetures'])):
            if cls_id not in interest_classid: continue
            indices = torch.where(classid_list == cls_id)[0] \
                if isinstance(classid_list, torch.Tensor) else self.get_indices_via_list(classid_list, cls_id)

            # cen = fetures[indices].mean(0, keepdim=True)
            current_feat = self.budget['fetures'][cls_id]
            current_feat.append(fetures[indices])
            cen = torch.cat(current_feat,dim=0).mean(0, keepdim=True)

            loss_intra += ((cen - fetures[indices])**2).mean()
            centroids.append(cen)
        
        centroids = torch.cat(centroids, dim=0)
        centroids = (centroids.unsqueeze(1) - centroids.unsqueeze(0).repeat(centroids.shape[0], 1, 1))**2
        centroids = centroids.mean(-1)
        centroids[range(centroids.shape[0]), range(centroids.shape[0])] = 1e9
        loss_inter = centroids.min()
        return loss_intra/loss_inter

    # def loss_margin(self, features, alpha=1):
    #     # list contains torch features
    #     loss_intra = torch.tensor(0.0).cuda()
    #     centroids = []
    #     for i in range(len(features)):
    #         feats = torch.cat(features[i], dim=0)
    #         cen = feats.mean(0, keepdim=True)
    #         loss_intra += ((cen - feats)**2).mean()
    #         centroids.append(cen)
    #     centroids = torch.cat(centroids, dim=0)
    #     centroids = (centroids.unsqueeze(1) - centroids.unsqueeze(0).repeat(centroids.shape[0], 1, 1))**2
    #     centroids = centroids.mean(-1)
    #     centroids[range(centroids.shape[0]), range(centroids.shape[0])] = 1e9
    #     loss_inter = centroids.min()
        # return loss_intra/loss_inter



    def __update_budget__(self, features, gt_classes, k=10):
        classid = gt_classes.unique()
        for i in classid:
            if i >= len(self.budget['fetures']): continue
            indices = torch.where(gt_classes == i)[0]
            self.budget['fetures'][i].append(features[indices].clone())

            if len(self.budget['fetures'][i]) > k:
                self.budget['fetures'][i] = self.budget['fetures'][i][-k:]
    
    def check_loss_margin(self, k=10):
        for feat in self.budget['fetures']:
            if len(feat) == 0: return False
            if torch.cat(feat).shape[0] < k: return False
        return True
        
    def get_loss_margin(self, k=2):
        l = {
            'loss_margin_fine_grained': 0.0,
            'loss_margin_superclass': 0.0,

        }

        if not self.check_loss_margin(k): 
            with torch.no_grad():
                self.__update_budget__(self.features, self.gt_classes, k)
            return l

        import itertools
        gt_classes = self.gt_classes
        gt_classes_unique = gt_classes.unique()
        features = self.features

        # gt_super_class = [self.mapper[int(i)] for i in gt_classes]
        # super_features=[[] for _ in range(len(self.super_class))]
        # for i in range(len(features)):
        #     indices = torch.where(gt_classes == i)[0]
        #     for j in self.mapper[int(i)]:
        #         super_features[j].append(features[indices])

        l = {
            'loss_margin_fine_grained': self.loss_margin(gt_classes, gt_classes_unique, features),
            # 'loss_margin_superclass': self.loss_margin(super_features),

        }
        print(l)
        with torch.no_grad():
            self.__update_budget__(self.features, self.gt_classes, k=10)
        return l


    def margin_loss(self):
        loss = None
        # return {'margin loss': loss}
        n_layer = len(self.child_cls_score)
        for i, (id_fg, list_invert) in enumerate(self.invert_super_class.items()):
            weights = []
            n = len(list_invert)
            for idx in list_invert:
                (index_super, index_fg) = idx

                # w =  self.child_cls_score[index_super][0].weight[index_fg].unsqueeze(0)
                w = (list(self.child_cls_score[index_super][0].parameters())[0])[index_fg].unsqueeze(0)
                weights.append(w)
                # print(self.child_cls_score[index_super][0].weight.shape)
                
            weights = torch.cat(weights)
            # print('weights :', weights.shape)

            weights = (weights.unsqueeze(1) - weights.unsqueeze(0).repeat(n, 1, 1))**2
            # print('dis 1 :', dis.shape)

            weights = weights.mean(axis=(-1))
            weights[torch.arange(n), torch.arange(n)] = 1e3
            # print('dis 2 :', dis.shape)
            # print(dis)
            weights = 1/weights.min()
            if i==0:
                loss = weights
            else:
                loss+= weights
        return loss

    def losses(self):
        losses = {}
        if self.similar_base_loss_weight > 0:
            losses.update({
                # 'loss_similar_base_via_score': self.similar_base_loss()*self.similar_base_loss_weight
                'loss_similar_base': self.loss_similar_base()*self.similar_base_loss_weight
                })
        if self.margin_loss_weight > 0:
            losses.update({
                'loss_similar_novel': self.loss_similar_novel2()*self.margin_loss_weight
                })
        t = 1
        t = 0
        if t >0:
            losses.update(self.get_loss_margin())
            # losses.update({
            #     'loss_fine_grained_novel': self.loss_fine_grained(invert_all=False)*t*10,
            #     'loss_fine_grained_all': self.loss_fine_grained(invert_all=True)*t
            #     })
        return losses

    def forward(self, x):
        # print('FastRCNNOutputLayers shape: ', x.shape)
        # print([t.shape for t in x])
        # assert 2==1
        features_att = None
        is_att = False
        roi_feat = x[0]
        if len(x) == 3:
            features_att = x[1]
            is_att = True
            result = []
            for f in features_att.values():
                f = self.pooler(f)
                f = torch.flatten(f, start_dim=1)
                # print(f.shape)
                result.append(f)
            features_att = torch.cat(result, dim=1)
        x = x[-1]
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1) # bz, 1024

        # roi_feat = self.super_head(roi_feat)
        if roi_feat.dim() > 2:
            roi_feat = torch.flatten(roi_feat, start_dim=1) # bz, 1024

        new_feat = x
        if is_att:
            features_att = features_att.mean(dim=0,keepdim=True).repeat(x.shape[0], 1)
            new_feat = torch.cat((new_feat, features_att), dim=-1)
            # print(new_feat.shape)

        super_cls_scores = self.cls_score(x) # bz x 60 # 1024 x 60
        proposal_deltas = self.bbox_pred(x)
        # del x

        dim  = super_cls_scores.shape

        fine_grained_cls_scores = torch.zeros((dim[0], self.num_classes + 1), device='cuda')
        
        prob_super_cls_scores = torch.nn.Sigmoid()(super_cls_scores)


        scores = []
        result = {key:[] for key in range(self.num_classes + 1)}

        for i in range(len(self.child_cls_score)):
            layer = self.child_cls_score[i]

            mask_x = prob_super_cls_scores[...,i, None]
    
            mask_x = mask_x.repeat((1, len(self.super_class[i])))

            # print(x.shape)
            # assert 2==1
            score = layer(new_feat)

   
            scores.append(score)
            score = torch.nn.Softmax(dim=-1)(score)
            weighted_score = score * mask_x

            for index, j in enumerate(self.super_class[i]):
                result[j].append(weighted_score[..., index, None])

        for i in range(len(self.child_cls_score)):
            for index, j in enumerate(self.super_class[i]):

                score = torch.cat(tuple(result[j]), dim=-1)
                fine_grained_cls_scores[..., j] = score.sum(dim=-1)

        # final_score = self.refine_cls(fine_grained_cls_scores)
        final_score = 0
        # if self.training:
            # self.top_grad()
        self.scores = scores
        return (super_cls_scores, fine_grained_cls_scores, scores, final_score), proposal_deltas

    def top_grad(self):
        pass
        def func(x, ratio=0.15):
            num = int(x.numel()*ratio)

            index = x.argsort(descending=True)[:num]
            x[index] *= 0
            return  x

        # for p in self.parameters():
            # p.grad =  
                    
@ROI_HEADS_OUTPUT_REGISTRY.register()
class SMS_LR(nn.Module):
    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(SMS_LR, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)


        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        # previous_model = torch.load(cfg.MODEL.WEIGHTS)
        # super-class from 1-shot using cosin bwt p6 features
 
        super_class = cfg.MODEL.ROI_HEADS.SUPER_CLASS
        self.is_multi_super_cls = cfg.MODEL.ROI_HEADS.IS_MULTI_SUPER_CLS
        self.topk = cfg.MODEL.ROI_HEADS.TOP_K

        self.cls_score = nn.Linear(input_size, len(super_class))

        # self.mapping = mapping(num_fc=0, fc_dim=1024)
        self.child_cls_score = torch.nn.ModuleList([])

        for i in range(len(super_class)):
            layer = nn.Linear(input_size, len(super_class[i]))
            self.child_cls_score.append(layer)
        
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        invert = {}
        for sup_id in range(len(super_class[:-1])):
            fine_grained_ids = super_class[sup_id]
            for index, id_ in enumerate(fine_grained_ids[1:]):
                list_id = invert.get(id_, [])
                list_id.append((sup_id, index+1))
                invert[id_] = list_id

        self.invert_super_class = invert
        self.super_class = super_class

        invert_all = {}
        for sup_id in range(len(super_class)):
            fine_grained_ids = super_class[sup_id]
            for index, id_ in enumerate(fine_grained_ids):

                list_id = invert_all.get(id_, [])
                list_id.append((sup_id, index))
                invert_all[id_] = list_id

        self.invert_all_class = invert_all
        self.num_classes = num_classes
        mapper = {i:[] for i in range(81)}
        for super_cls, child_classes in enumerate(self.super_class):
            for i in child_classes:
                mapper[i].append(super_cls)
        mapper = {key: torch.tensor(val).reshape(1,-1) for key, val in mapper.items()}
        self.mapper= mapper


    def loss_fine_grained(self, invert_all=True):

        invert_cls = self.invert_all_class if invert_all else self.invert_super_class

        if isinstance(self.gt_classes, tuple):
            self.gt_classes = torch.cat(self.gt_classes, dim=0)

        loss = torch.tensor(0.0).cuda()
        for gt_cls in self.gt_classes.unique().cpu().numpy().tolist():
            if gt_cls not in invert_cls.keys(): continue
            super_cls_contains_gt_cls = invert_cls[gt_cls]
            indices = torch.where(self.gt_classes==gt_cls)[0]

            if indices.shape[0]<1: continue
            for super_id, fg_id in super_cls_contains_gt_cls:
                # create label for softmax_cross entropy
                labels = torch.zeros(len(indices), dtype=int) + fg_id


                pred_class_logits = self.scores[super_id][indices]
                smoothing = 0.5

                # caculate loss via cross entropy
                loss += F.cross_entropy(pred_class_logits, labels.cuda(), reduction="sum")

                if super_id != len(self.scores) - 1:
                    los_func = FGCrossEntropyLoss(weight=None, reduction='sum', smoothing=smoothing)
                    loss += los_func(pred_class_logits, labels.cuda())
 
        return loss/self.gt_classes.shape[0]

    def losses(self):
        losses = {}
        losses.update({'loss_fine_grained': self.loss_fine_grained()*3e1})
        return losses

    def forward(self, x, proposals=None):
        if self.training and proposals:
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            self.gt_classes = gt_classes

        
        proposal_deltas = self.bbox_pred(x)

        super_cls_scores = self.cls_score(x) 
        dim  = super_cls_scores.shape
        fine_grained_cls_scores = torch.zeros((dim[0], self.num_classes + 1), device='cuda')
        prob_super_cls_scores = torch.nn.Sigmoid()(super_cls_scores)
        scores = []
        result = {key:[] for key in range(self.num_classes + 1)}
        for i in range(len(self.child_cls_score)):
            layer = self.child_cls_score[i]

            mask_x = prob_super_cls_scores[...,i, None]
            mask_x = mask_x.repeat((1, len(self.super_class[i])))
            score = layer(x)
            scores.append(score)
            score = torch.nn.Softmax(dim=-1)(score)
            weighted_score = score * mask_x
            for index, j in enumerate(self.super_class[i]):
                result[j].append(weighted_score[..., index, None])

        for i in range(len(self.child_cls_score)):
            for index, j in enumerate(self.super_class[i]):

                score = torch.cat(tuple(result[j]), dim=-1)
                fine_grained_cls_scores[..., j] = score.sum(dim=-1)

        self.scores = scores

        return (super_cls_scores, fine_grained_cls_scores), proposal_deltas


@ROI_HEADS_OUTPUT_REGISTRY.register()
class SoftSuperClassOutputLayers_normal21_Correlation2_clean(SoftSuperClassOutputLayers):
    
    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(SoftSuperClassOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # self.__init_modules__(cfg)
        # self.__init_param__(cfg)


        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        BASE_CLS = 60
        NOVEL_CLS = 20
        NUM_CHILD_CLS = 5
        # previous_model = torch.load(cfg.MODEL.WEIGHTS)
        # super-class from 1-shot using cosin bwt p6 features
 
        super_class = cfg.MODEL.ROI_HEADS.SUPERCLASS
        self.is_multi_super_cls = cfg.MODEL.ROI_HEADS.IS_MULTI_SUPER_CLS
        

        self.cls_score = nn.Linear(input_size, len(super_class))

        # self.mapping = mapping(num_fc=0, fc_dim=1024)
        self.child_cls_score = torch.nn.ModuleList([])

        for i in range(len(super_class)):
            # layer = CosineSimFCLayer(cfg, input_size, len(super_class[i]))
            # layer = CosineSimFCLayer2(cfg, input_size, len(super_class[i]), ratio=0.0, dropout_rate=0.0)
            layer = nn.Linear(input_size, len(super_class[i]))
            self.child_cls_score.append(layer)
        
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        invert = {}
        for sup_id in range(len(super_class[:-1])):
            fine_grained_ids = super_class[sup_id]
            for index, id_ in enumerate(fine_grained_ids[1:]):
                list_id = invert.get(id_, [])
                list_id.append((sup_id, index+1))
                invert[id_] = list_id

        self.invert_super_class = invert
        self.super_class = super_class

        invert_all = {}
        for sup_id in range(len(super_class)):
            fine_grained_ids = super_class[sup_id]
            for index, id_ in enumerate(fine_grained_ids):

                list_id = invert_all.get(id_, [])
                list_id.append((sup_id, index))
                invert_all[id_] = list_id

        self.invert_all_class = invert_all
        self.num_classes = num_classes
        self.pooler = torch.nn.AdaptiveAvgPool2d(1)
        self.margin_loss_weight = 0
        self.similar_base_loss_weight = 0
        mapper = {i:[] for i in range(81)}
        for super_cls, child_classes in enumerate(self.super_class):
            for i in child_classes:
                mapper[i].append(super_cls)
        mapper = {key: torch.tensor(val).reshape(1,-1) for key, val in mapper.items()}
        self.mapper= mapper
        self.budget ={
            'fetures': [[]for i in range(num_classes)],
        }

        logger.info(f'self.mapper: {self.mapper}')
        logger.info('-'*64)
        logger.info(f'super_class: {super_class}')
        logger.info('-'*64)
        logger.info(f'invert_super_class: {self.invert_super_class}')
        logger.info(f'len(self.invert_super_class): {len(self.invert_super_class)}')
        logger.info(f'self.invert_all_class: {self.invert_all_class}')
        logger.info(f'len(self.invert_all_class): {len(self.invert_all_class)}')

    def loss_fine_grained(self, invert_all=True):
        # print(self.gt_classes)

        invert_cls = self.invert_all_class if invert_all else self.invert_super_class

        if isinstance(self.gt_classes, tuple):
            self.gt_classes = torch.cat(self.gt_classes, dim=0)

        loss = torch.tensor(0.0).cuda()
        # N = 0
        for gt_cls in self.gt_classes.unique().cpu().numpy().tolist():
            if gt_cls not in invert_cls.keys(): continue
            super_cls_contains_gt_cls = invert_cls[gt_cls]
            indices = torch.where(self.gt_classes==gt_cls)[0]
            # if gt_cls == self.num_classes: continue
            if indices.shape[0]<1: continue
            for super_id, fg_id in super_cls_contains_gt_cls:
                # create label for softmax_cross entropy
                labels = torch.zeros(len(indices), dtype=int)
                # import random
                # if fg_id == 0 and super_id != len(self.super_class) -1:
                    # labels += random.randint(0, len(self.super_class[super_id])-1)
                # else:
                labels += fg_id
                # print(labels)
                pred_class_logits = self.scores[super_id][indices]
                smoothing = 0.5
                # if fg_id == 0 and super_id != len(self.super_class) -1:
                    # smoothing = 0
                
 

                # caculate loss via cross entropy
                # loss += F.cross_entropy(pred_class_logits, labels.cuda(), reduction="sum", label_smoothing=1/num_classes)
                # print(smoothing)
                # assert 2==1
                loss += F.cross_entropy(pred_class_logits, labels.cuda(), reduction="sum")
                # los_func = CorrelationLabelCrossEntropyLoss(weight=None, reduction='sum', smoothing=smoothing)
                if super_id != len(self.scores) - 1:
                    los_func = FGCrossEntropyLoss(weight=None, reduction='sum', smoothing=smoothing)
                    # los_func2 = SmoothCrossEntropyLoss(weight=None, reduction='sum', smoothing=smoothing)
                    loss += los_func(pred_class_logits, labels.cuda()) #+  los_func2(pred_class_logits, labels.cuda())
                # else:
 
                # loss += los_func(pred_class_logits, labels.cuda())
 
        return loss/self.gt_classes.shape[0]

    def losses(self):
        losses = {}
        # if self.similar_base_loss_weight > 0:
        #     losses.update({
        #         # 'loss_similar_base_via_score': self.similar_base_loss()*self.similar_base_loss_weight
        #         'loss_similar_base': self.loss_similar_base()*self.similar_base_loss_weight
        #         })
        if self.similar_base_loss_weight > 0:
            loss_intra, loss_inter = self.loss_margin_intra_super()
            losses.update({
                'loss_intra_super': loss_intra*self.similar_base_loss_weight,
                'loss_inter_super': loss_inter*self.similar_base_loss_weight
                })
                
        if self.margin_loss_weight > 0:
            loss_intra, loss_inter, loss_intra_weights = self.loss_margin_intra_fg2()
            losses.update({
                'loss_intra_fg': loss_intra*self.margin_loss_weight,
                'loss_inter_fg': loss_inter*self.margin_loss_weight,
                # 'loss_intra_weights': loss_intra_weights*self.margin_loss_weight,
                })
        t = 1
        # t = 0
        if t >0:
            # losses.update(self.get_loss_margin())
            losses.update({'loss_fine_grained': self.loss_fine_grained()*3e1})
            # losses.update({
            #     'loss_fine_grained_novel': self.loss_fine_grained(invert_all=False)*t*10,
            #     'loss_fine_grained_all': self.loss_fine_grained(invert_all=True)*t
            #     })
        del self.features, self.gt_classes
        return losses

    def forward(self, x, proposals=None):
        if self.training and proposals:
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            self.gt_classes = gt_classes


        # features_att = None
        # is_att = False
        # roi_feat = x[0]
        # if len(x) == 3:
        #     features_att = x[1]
        #     is_att = True
        #     result = []
        #     for f in features_att.values():
        #         f = self.pooler(f)
        #         f = torch.flatten(f, start_dim=1)
        #         # print(f.shape)
        #         result.append(f)
        #     features_att = torch.cat(result, dim=1)
        # # x = x[-1]
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1) # bz, 1024

        # roi_feat = self.super_head(roi_feat)
        # if is_att:
        #     features_att = features_att.mean(dim=0,keepdim=True).repeat(x.shape[0], 1)
        #     new_feat = torch.cat((new_feat, features_att), dim=-1)
            # print(new_feat.shape)

        # new_feat = self.mapping(x)
        new_feat = x
        proposal_deltas = self.bbox_pred(x)
        del x
        # if self.training and proposals:
        # self.cal_loss_similar_novel2(new_feat, self.gt_classes)


        super_cls_scores = self.cls_score(new_feat) # bz x 60 # 1024 x 60

        dim  = super_cls_scores.shape
        fine_grained_cls_scores = torch.zeros((dim[0], self.num_classes + 1), device='cuda')
        prob_super_cls_scores = torch.nn.Sigmoid()(super_cls_scores)


        scores = []
        result = {key:[] for key in range(self.num_classes + 1)}
        for i in range(len(self.child_cls_score)):
            layer = self.child_cls_score[i]

            mask_x = prob_super_cls_scores[...,i, None]
            mask_x = mask_x.repeat((1, len(self.super_class[i])))
            score = layer(new_feat)
            scores.append(score)
            score = torch.nn.Softmax(dim=-1)(score)
            weighted_score = score * mask_x
            for index, j in enumerate(self.super_class[i]):
                result[j].append(weighted_score[..., index, None])

        for i in range(len(self.child_cls_score)):
            for index, j in enumerate(self.super_class[i]):

                score = torch.cat(tuple(result[j]), dim=-1)
                fine_grained_cls_scores[..., j] = score.sum(dim=-1)

        self.scores = scores
        self.features = new_feat
        final_score = 0

        return (super_cls_scores, fine_grained_cls_scores, scores, final_score), proposal_deltas

def create_multi_hot(label, num_classes):
    label = torch.tensor(label).reshape(-1)
    label = label.cpu()
    label = label.unsqueeze(0)
    target = torch.zeros(label.size(0), num_classes).scatter_(1, label, 1.)
    return target.cuda()

def super_cls_loss(predicted, inputs):
    num_classes = predicted.shape[1]
    inputs = [create_multi_hot(i, num_classes) for i in inputs]
    inputs = torch.cat(inputs, dim=0)
    return F.binary_cross_entropy_with_logits(predicted, inputs, reduction="mean")

class FGCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():       
            label = torch.zeros(targets.size(0), n_classes).cuda()
            novel_index = torch.where(targets != 0)
            base_index = torch.where(targets == 0)

            if novel_index[0].shape[0] > 0:

                label[novel_index[0]] = label[novel_index[0]].scatter_(1, targets.data[novel_index[0]].unsqueeze(1), 1.0 - smoothing)
                label[novel_index[0], 0] = smoothing

            if base_index[0].shape[0] > 0:

                label[base_index[0]] = label[base_index[0]].fill_(smoothing /(n_classes-1)).scatter_(1, targets.data[base_index[0]].unsqueeze(1), 1.0 - smoothing)


        return label

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))








