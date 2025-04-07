import torch
from torchvision.ops.boxes import box_area


def _box_cxcywh_to_xyxy(x):
    """
    转换边界框格式，从(中心x, 中心y, 宽, 高)转换为(左上x, 左上y, 右下x, 右下y)

    Parameters:
        x: 形状为[N, 4]的张量，格式为[cx, cy, w, h]

    Returns:
        形状为[N, 4]的张量，格式为[x1, y1, x2, y2]
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def _box_xyxy_to_cxcywh(x):
    """
    转换边界框格式，从(左上x, 左上y, 右下x, 右下y)转换为(中心x, 中心y, 宽, 高)

    Parameters:
        x: 形状为[N, 4]的张量，格式为[x1, y1, x2, y2]

    Returns:
        形状为[N, 4]的张量，格式为[cx, cy, w, h]
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    计算两组框之间的IoU

    Parameters:
        boxes1: 形状为[N, 4]的张量，格式为xyxy
        boxes2: 形状为[M, 4]的张量，格式为xyxy

    Returns:
        形状为[N, M]的张量，包含boxes1和boxes2中各个框之间的IoU
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def generalized_box_iou(boxes1, boxes2):
    """
    计算两组框之间的广义IoU

    Parameters:
        boxes1: 形状为[N, 4]的张量，格式为xyxy
        boxes2: 形状为[M, 4]的张量，格式为xyxy

    Returns:
        形状为[N, M]的张量，包含boxes1和boxes2中各个框之间的广义IoU
    """
    # 得到两组框之间的IoU
    iou = box_iou(boxes1, boxes2)

    # 计算最小外接矩形的左上和右下坐标
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算最小外接矩形的面积
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # 计算GIoU = IoU - (area - union) / area
    return iou - (area - iou * (box_area(boxes1).unsqueeze(1) + box_area(boxes2) - iou * (box_area(boxes1).unsqueeze(1) + box_area(boxes2)))) / area 