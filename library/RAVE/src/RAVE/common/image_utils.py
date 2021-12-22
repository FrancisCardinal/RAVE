import torch
from torch.nn import functional as F

import numpy as np

import random

random.seed(42)


def tensor_to_opencv_image(tensor):
    """
    Converts a pytorch tensor that represents an image into an numpy array

    Args:
        tensor (pytorch tensor): The image to convert

    Returns:
        Numpy array: The converted image
    """
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor *= 255.0
    image = np.ascontiguousarray(tensor, dtype=np.uint8)
    return image


def opencv_image_to_tensor(image, DEVICE):
    """
    OpenCV BGR image to tensor RGB image

    Args:
        Image (ndarray): Image with shape (width, height, 3).
        DEVICE (string): Pytorch device.
    """
    tensor = torch.from_numpy(image).to(DEVICE)
    tensor = tensor.permute(2, 0, 1).float()
    tensor /= 255
    return tensor


def inverse_normalize(tensor, mean, std):
    """
    Undo the normalization operation that was performed on an image when
    it was passed to a network

    Args:
        tensor (pytorch tensor): The image on which to perform the operation
        mean (float): The mean that was used for the normalization
        std (float): The std that was used for the normalization

    Returns:
        pytorch tensor: The unormalized image
    """

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def apply_image_translation(
    image_tensor, x_extremums=[-0.2, 0.2], y_extremums=[-0.2, 0.2]
):
    """
    A data augmentation operation, translates the frame randomly, by
    selecting an x and y value in the x_extremums and y_extremums ranges,
    respectively.This should be used instead of the pytorch transform if
    you wish to know the amplitude of the translation.

    Args:
        image_tensor (pytorch tensor):
            The frame on which to apply the translation
        x_extremums (list, optional):
            The min and max value of the x translation. Defaults to [-0.2, 0.2]
        y_extremums (list, optional):
            The min and max value of the y translation. Defaults to [-0.2, 0.2]

    Returns:
        Tuple:
            The translated frame (pytorch tensor), the x offset (float) and the
            y offset (float)
    """

    x_offset = random.uniform(x_extremums[0], x_extremums[1])
    y_offset = random.uniform(y_extremums[0], y_extremums[1])

    output_image_tensor = do_affine_grid_operation(
        image_tensor, translation=(x_offset, y_offset)
    )

    return output_image_tensor, x_offset, y_offset


def apply_image_rotation(image_tensor, rotation_angle_extremums=[-0.1, 0.1]):
    """
    A data augmentation operation, rotates the frame randomly, by selecting
    an angle in the 'rotation_angle_extremums'range.
    This should be used instead of the pytorch transform if you wish to know
    the amplitude of the rotation.

    Args:
        image_tensor (pytorch tensor): The frame on which to apply the rotation
        rotation_angle_extremums (list, optional):
            The min and max value of the rotation angle.
            Defaults to [-0.1, 0.1]

    Returns:
        Tuple:The rotated frame (pytorch tensor) and the rotation angle (float)
    """
    phi = random.uniform(
        rotation_angle_extremums[0], rotation_angle_extremums[1]
    )

    output_image_tensor = do_affine_grid_operation(image_tensor, phi=phi)

    return output_image_tensor, phi


def apply_image_translation_and_rotation(
    image_tensor,
    x_extremums=[-0.2, 0.2],
    y_extremums=[-0.2, 0.2],
    rotation_angle_extremums=[-0.1, 0.1],
):
    """
    A data augmentation operation,translates and rotates the frame randomly,
    by selecting an x and y value in the x_extremums and y_extremums ranges,
    respectively, and by selecting an angle in the
    'rotation_angle_extremums'range. This should be used instead of the
    pytorch transform if you wish to know the amplitude of the translation
    and the rotation.

    Args:
        image_tensor (pytorch tensor):
            The frame on which to apply the translation
        x_extremums (list, optional):
            The min and max value of the x translation. Defaults to [-0.2, 0.2]
        y_extremums (list, optional):
            The min and max value of the y translation. Defaults to [-0.2, 0.2]
        rotation_angle_extremums (list, optional):
            The min and max value of therotation angle. Defaults to [-0.1, 0.1]

    Returns:
        Tuple:
            The translated frame (pytorch tensor), the x offset (float),the y
            offset (float) and the rotation angle (float)
    """

    x_offset = random.uniform(x_extremums[0], x_extremums[1])
    y_offset = random.uniform(y_extremums[0], y_extremums[1])
    phi = random.uniform(
        rotation_angle_extremums[0], rotation_angle_extremums[1]
    )

    output_image_tensor = do_affine_grid_operation(
        image_tensor, (x_offset, y_offset), phi
    )

    return output_image_tensor, x_offset, y_offset, phi


def do_affine_grid_operation(image_tensor, translation=(0, 0), phi=0):
    """
    Executes the affine operation

    Args:
        image_tensor (pytorch tensor):
            The image on which to apply the operation
        translation (Tuple):
            The magnitude of the translation operation
        phi (float):
            The angle of rotation

    Returns:
        pytorch tensor: The processed image
    """
    x_offset, y_offset = translation
    IMAGE_HEIGHT, IMAGE_WIDTH = image_tensor.shape[1], image_tensor.shape[2]

    transformation_matrix = torch.tensor(
        [
            [
                np.cos(phi),
                np.sin(phi) * IMAGE_HEIGHT / IMAGE_WIDTH,
                -x_offset * 2,
            ],
            [
                -np.sin(phi) * IMAGE_WIDTH / IMAGE_HEIGHT,
                np.cos(phi),
                -y_offset * 2,
            ],
        ],
        dtype=torch.float,
    )  # We need the'*2' here because affine_grid considers the top left corner
    # as [-1, -1] and the bottom right one as [1, 1] (as opposed to the
    # convention of this module where the top left corner is [0, 0])

    grid = F.affine_grid(
        transformation_matrix.unsqueeze(0), image_tensor.unsqueeze(0).size()
    )
    image_tensor = F.grid_sample(image_tensor.unsqueeze(0), grid)
    image_tensor = image_tensor.squeeze(0)

    return image_tensor


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    where xy1=top-left, xy2=bottom-right
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    where xy1=top-left, xy2=bottom-right
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        """
        box = 4xn
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords
