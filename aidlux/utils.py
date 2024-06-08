import cv2
import numpy as np

target_class = ['fire', 'person']


def xywh2xyxy(x):
    '''
    Box (center x, center y, width, height) to (x1, y1, x2, y2)
    中心点xy+宽和高 to 左上右下格式
    '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(box):
    '''
    Box (left_top x, left_top y, right_bottom x, right_bottom y) to (left_top x, left_top y, width, height)
    左上右下格式 to 中心点xy+宽和高
    '''
    box[:, 2:] = box[:, 2:] - box[:, :2]
    return box


def NMS(dets, thresh):
    '''
    单类NMS算法
    dets.shape = (N, 5), (left_top x, left_top y, right_bottom x, right_bottom y, Scores)
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return dets[keep]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_img(img, target_shape: tuple = None, div_num=255, means: list = [0.485, 0.456, 0.406],stds: list = [0.229, 0.224, 0.225]):
    '''
    图像预处理:
    target_shape: 目标shape
    div_num: 归一化除数
    means: len(means)==图像通道数，通道均值, None不进行zscore
    stds: len(stds)==图像通道数，通道方差, None不进行zscore
    '''
    img_processed = np.copy(img)
    # resize
    if target_shape:
        # img_processed = cv2.resize(img_processed, target_shape)
        img_processed = letterbox(img_processed, target_shape, stride=None, auto=False)[0]

    img_processed = img_processed.astype(np.float32)
    img_processed = img_processed / div_num

    # z-score
    if means is not None and stds is not None:
        means = np.array(means).reshape(1, 1, -1)
        stds = np.array(stds).reshape(1, 1, -1)
        img_processed = (img_processed - means) / stds

    # unsqueeze
    img_processed = img_processed[None, :]

    return img_processed.astype(np.float32)


def convert_shape(shapes: tuple or list, int8=False):
    '''
    转化为aidlite需要的格式
    '''
    if isinstance(shapes, tuple):
        shapes = [shapes]
    out = []
    for shape in shapes:
        nums = 1 if int8 else 4
        for n in shape:
            nums *= n
        out.append(nums)
    return out


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def detect_postprocess(prediction, img0shape, img1shape, conf_thres=0.25, iou_thres=0.45):
    '''
    检测输出后处理
    prediction: aidlite模型预测输出 [x,y,width,height,conf,fire,person]
    img0shape: 原始图片shape
    img1shape: 输入图片shape
    conf_thres: 置信度阈值
    iou_thres: IOU阈值
    return: list[np.ndarray(N, 5)], 对应类别的坐标框信息, xywh、conf
    '''
    # 获取输入图片的shape
    h, w, _ = img1shape
    # 预测类别的数目(减去前五列)
    cls_num = prediction.shape[-1] - 5
    # 筛出满足置信度的预测
    valid_condidates = prediction[prediction[..., 4] > conf_thres]
    # 相对坐标转化为绝对坐标
    valid_condidates[:, 0] *= w  # x
    valid_condidates[:, 1] *= h  # y
    valid_condidates[:, 2] *= w  # width
    valid_condidates[:, 3] *= h  # height
    # 将边界框的表示从（x、y、宽度、高度）转换为（左上角x、左上角y、右下角x、右下角y）的形式。
    valid_condidates[:, :4] = xywh2xyxy(valid_condidates[:, :4])
    # 边界框过滤
    valid_condidates = valid_condidates[
        (valid_condidates[:, 0] > 0) & (valid_condidates[:, 1] > 0) & (valid_condidates[:, 2] > 0) & (
                valid_condidates[:, 3] > 0)]
    # 根据置信度，选择最高置信度的类别作为每个目标候选框的类别
    box_cls = valid_condidates[:, 5:].argmax(1)
    cls_box = []
    # 遍历每个类别(fire person)
    for i in range(cls_num):
        temp_boxes = valid_condidates[box_cls == i]   # 广播判等 保留该类的框
        # 该类没有框
        if len(temp_boxes) == 0:
            cls_box.append([])
            continue
        # 对于每个类别，将相应的目标候选框们进行非最大值抑制
        temp_boxes = NMS(temp_boxes, iou_thres)
        # 使用scale_coords函数将坐标从输入图片的形状转换为原始图片的形状
        temp_boxes[:, :4] = scale_coords([h, w], temp_boxes[:, :4], img0shape).round()
        # 将目标候选框的坐标表示从(x1, y1, x2, y2)转换为(x, y, width, height)的形式
        temp_boxes[:, :4] = xyxy2xywh(temp_boxes[:, :4])
        cls_box.append(temp_boxes[:, :5])
    return cls_box


def draw_detect_res(img, all_boxes, person, fire):
    '''
    检测结果绘制
    '''
    img = img.astype(np.uint8)
    color_step = int(255 / len(all_boxes))
    for bi in range(len(all_boxes)):
        if len(all_boxes[bi]) == 0:
            continue
        for box in all_boxes[bi]:
            x, y, w, h = [int(t) for t in box[:4]]
            # 非person或fire
            if bi != 0 and bi != 1:
                continue
            if bi == 0:
                fire = 1
            if bi == 1:
                person = 1
            cv2.putText(img, f'{target_class[bi]}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, bi * color_step, 255 - bi * color_step), thickness=2)
    return img, person, fire
