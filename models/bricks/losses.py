from torch.nn import functional as F
import torch
from torchvision.ops.boxes import box_area


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def vari_sigmoid_focal_loss(inputs, targets, weight, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Variational sigmoid focal loss.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        weight: A float tensor with the same shape as inputs.
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    targets = targets.type_as(inputs)

    # calculate probability
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    targets = targets * weight
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def ia_bce_loss(inputs, targets, gt_score, num_boxes, k: float = 0.25, alpha: float = 0, gamma: float = 2):
    prob = inputs.sigmoid().detach()
    # calculate iou_aware_score and constrain the value following original implementation
    iou_aware_score = prob**k * gt_score.unsqueeze(-1)**(1 - k)
    iou_aware_score = iou_aware_score.clamp(min=0.01)
    target_score = targets * iou_aware_score
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + targets
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def complete_iou_loss(pred_boxes, target_boxes, eps=1e-7, reduction='none'):
    """
    CIoU损失：完整交并比损失
    
    参数:
        pred_boxes: 预测框，格式为(x1, y1, x2, y2)或(center_x, center_y, w, h)
        target_boxes: 目标框，格式应与pred_boxes一致
        eps: 数值稳定性的小值
        reduction: 'none', 'mean', 'sum' 中的一个
        
    返回:
        CIoU损失
    """
    # 确保boxes是xyxy格式
    if pred_boxes.size(-1) == 4 and target_boxes.size(-1) == 4:
        # 检测输入是否为cxcywh格式
        if (pred_boxes[..., 2:] >= 0).all() and (pred_boxes[..., 2:] <= 1).all():
            # 如果是cxcywh格式，转换为xyxy
            pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
            pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
            pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
            pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
            pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
            
            target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
            target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
            target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
            target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
            target_boxes = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=-1)
    
    # 计算各框的面积
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    
    # 计算IoU
    # 计算交集区域
    left_top = torch.max(pred_boxes[..., :2], target_boxes[..., :2])
    right_bottom = torch.min(pred_boxes[..., 2:], target_boxes[..., 2:])
    wh = (right_bottom - left_top).clamp(min=0)  # [N, 2]
    intersection = wh[..., 0] * wh[..., 1]
    
    # 计算并集区域
    union = pred_area + target_area - intersection
    
    # IoU
    iou = intersection / (union + eps)
    
    # 找出包含两个框的最小外接矩形
    enclose_left_top = torch.min(pred_boxes[..., :2], target_boxes[..., :2])
    enclose_right_bottom = torch.max(pred_boxes[..., 2:], target_boxes[..., 2:])
    
    # 计算对角线距离平方(c^2)
    enclose_wh = (enclose_right_bottom - enclose_left_top).clamp(min=0)
    enclose_c2 = torch.sum(enclose_wh**2, dim=-1)
    
    # 计算中心点距离平方(ρ^2)
    pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
    target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
    center_distance2 = torch.sum((pred_center - target_center)**2, dim=-1)
    
    # 计算长宽比一致性
    pred_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    target_wh = target_boxes[..., 2:] - target_boxes[..., :2]
    
    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(target_wh[..., 0] / (target_wh[..., 1] + eps)) - 
        torch.atan(pred_wh[..., 0] / (pred_wh[..., 1] + eps)), 2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    # 计算CIoU
    ciou = iou - center_distance2 / (enclose_c2 + eps) - alpha * v
    
    # 计算CIoU损失
    ciou_loss = 1 - ciou
    
    # 根据reduction参数返回相应结果
    if reduction == 'mean':
        return ciou_loss.mean()
    elif reduction == 'sum':
        return ciou_loss.sum()
    else:  # reduction == 'none'
        return ciou_loss
