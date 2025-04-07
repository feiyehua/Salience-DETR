import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops.boxes import _box_cxcywh_to_xyxy, generalized_box_iou
from torchvision.ops.boxes import box_area


class HungarianMatcher(nn.Module):
    """This class implements the Hungarian matching algorithm for bipartite graphs. It matches predicted bounding 
    boxes to ground truth boxes based on the minimum cost assignment. The cost is computed as a weighted sum of 
    classification, bounding box, and generalized intersection over union (IoU) costs. The focal loss is used to 
    weigh the classification cost. The HungarianMatcher class can be used in single or mixed assignment modes.
    The mixed assignment modes is introduced in `Align-DETR <https://arxiv.org/abs/2304.07527>`_.

    :param cost_class: The weight of the classification cost, defaults to 1
    :param cost_bbox: The weight of the bounding box cost, defaults to 1
    :param cost_giou: The weight of the generalized IoU cost, defaults to 1
    :param focal_alpha: The alpha parameter of the focal loss, defaults to 0.25
    :param focal_gamma: The gamma parameter of the focal loss, defaults to 2.0
    :param mixed_match: If True, mixed assignment is used, defaults to False
    :param small_object_size: Size threshold below which objects are considered small (as ratio of image area)
    :param small_object_iou_scale: Scale factor for IoU threshold for small objects
    """
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mixed_match: bool = False,
        small_object_size: float = 0.04,  # 4% of image area
        small_object_iou_scale: float = 0.9,  # Scale factor for small objects (e.g., 0.9)
    ):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.mixed_match = mixed_match
        
        # Parameters for small object refinement
        self.small_object_size = small_object_size
        self.small_object_iou_scale = small_object_iou_scale

    def calculate_class_cost(self, pred_logits, gt_labels, **kwargs):
        out_prob = pred_logits.sigmoid()

        # Compute the classification cost.
        neg_cost_class = -(1 - self.focal_alpha) * out_prob**self.focal_gamma * (1 - out_prob + 1e-6).log()
        pos_cost_class = -self.focal_alpha * (1 - out_prob)**self.focal_gamma * (out_prob + 1e-6).log()
        cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]

        return cost_class

    def calculate_bbox_cost(self, pred_boxes, gt_boxes, **kwargs):
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)
        return cost_bbox

    def calculate_giou_cost(self, pred_boxes, gt_boxes, **kwargs):
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(_box_cxcywh_to_xyxy(pred_boxes), _box_cxcywh_to_xyxy(gt_boxes))
        return cost_giou

    @torch.no_grad()
    def calculate_cost(self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor):
        # Calculate class, bbox and giou cost
        cost_class = self.calculate_class_cost(pred_logits, gt_labels)
        cost_bbox = self.calculate_bbox_cost(pred_boxes, gt_boxes)
        cost_giou = self.calculate_giou_cost(pred_boxes, gt_boxes)

        # Final cost matrix
        c = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        return c

    def scale_aware_iou_thresholds(self, boxes, image_size=(1.0, 1.0)):
        """Compute scale-aware IoU thresholds based on box size
        
        Args:
            boxes: Boxes in cxcywh format
            image_size: Image size (height, width) for relative area calculation
            
        Returns:
            Tensor of IoU thresholds for each box
        """
        # Calculate the area of each box relative to image size
        areas = boxes[:, 2] * boxes[:, 3] / (image_size[0] * image_size[1])
        
        # Determine if boxes are small
        is_small = areas < self.small_object_size
        
        # Default IoU threshold (0.5, 0.6, 0.7 for cascade stages)
        base_iou_threshold = torch.ones_like(areas) * 0.5
        
        # Apply scaling factor to IoU threshold for small objects
        iou_thresholds = torch.where(
            is_small,
            base_iou_threshold * self.small_object_iou_scale,  # For small objects
            base_iou_threshold  # For normal objects
        )
        
        return iou_thresholds

    @torch.no_grad()
    def forward(
        self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor, gt_copy: int = 1
    ):
        c = self.calculate_cost(pred_boxes, pred_logits, gt_boxes, gt_labels)

        # single assignment
        if not self.mixed_match:
            indices = linear_sum_assignment(c.cpu())
            return torch.as_tensor(indices[0]), torch.as_tensor(indices[1])

        # mixed assignment, used in AlignDETR
        gt_size = c.size(-1)
        num_queries = len(c)
        gt_copy = min(int(num_queries * 0.5 / gt_size), gt_copy) if gt_size > 0 else gt_copy
        src_ind, tgt_ind = linear_sum_assignment(c.cpu().repeat(1, gt_copy))
        tgt_ind = tgt_ind % gt_size
        tgt_ind, ind = torch.as_tensor(tgt_ind, dtype=torch.int64).sort()
        src_ind = torch.as_tensor(src_ind, dtype=torch.int64)[ind].view(-1)
        return src_ind, tgt_ind
