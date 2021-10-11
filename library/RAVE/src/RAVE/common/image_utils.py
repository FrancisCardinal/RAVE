import torch
from torch.nn import functional as F

import numpy as np

import random

random.seed(42)


def tensor_to_opencv_image(tensor):
    """Converts a pytorch tensor that represents an image into an numpy array

    Args:
        tensor (pytorch tensor): The image to convert

    Returns:
        Numpy array: The converted image
    """
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor *= 255.0
    image = np.ascontiguousarray(tensor, dtype=np.uint8)
    return image


def inverse_normalize(tensor, mean, std):
    """Undo the normalization operation that was performed on an image when it was
       passed to a network 
       https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/17 

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
    """A data augmentation operation, translates the frame randomly, by selecting an x and y 
       value in the x_extremums and y_extremums ranges, respectively.
       This should be used instead of the pytorch transform if you wish to know the amplitude 
       of the translation. 

    Args:
        image_tensor (pytorch tensor): The frame on which to apply the translation
        x_extremums (list, optional): The min and max value of the x translation. Defaults to [-0.2, 0.2].
        y_extremums (list, optional): The min and max value of the y translation. Defaults to [-0.2, 0.2].

    Returns:
        Tuple: The translated frame (pytorch tensor), the x offset (float) and the y offset (float)
    """

    x_offset = random.uniform(x_extremums[0], x_extremums[1])
    y_offset = random.uniform(y_extremums[0], y_extremums[1])

    output_image_tensor = do_affine_grid_operation(
        image_tensor, translation=(x_offset, y_offset)
    )

    return output_image_tensor, x_offset, y_offset


def apply_image_rotation(image_tensor, rotation_angle_extremums=[-0.1, 0.1]):
    """A data augmentation operation, rotates the frame randomly, by selecting an 
       angle in the 'rotation_angle_extremums'range.
       This should be used instead of the pytorch transform if you wish to know the amplitude 
       of the rotation. 

    Args:
        image_tensor (pytorch tensor): The frame on which to apply the rotation
        rotation_angle_extremums (list, optional): The min and max value of the rotation angle. Defaults to [-0.1, 0.1].

    Returns:
        Tuple: The rotated frame (pytorch tensor) and the rotation angle (float)
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
    """A data augmentation operation, translates and rotates the frame randomly, by selecting an x and y
       value in the x_extremums and y_extremums ranges, respectively, and by selecting an
       angle in the 'rotation_angle_extremums'range.
       This should be used instead of the pytorch transform if you wish to know the amplitude
       of the translation and the rotation.

    Args:
        image_tensor (pytorch tensor): The frame on which to apply the translation
        x_extremums (list, optional): The min and max value of the x translation. Defaults to [-0.2, 0.2].
        y_extremums (list, optional): The min and max value of the y translation. Defaults to [-0.2, 0.2].
        rotation_angle_extremums (list, optional): The min and max value of the rotation angle. Defaults to [-0.1, 0.1].

    Returns:
        Tuple: The translated frame (pytorch tensor), the x offset (float), the y offset (float) and the rotation angle (float)
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
    """Executes the affine operation

    Args:
        image_tensor (pytorch tensor): The image on which to apply the operation
        translation (Tuple): The magnitude of the translation operation
        phi (float): The angle of rotation

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
    )  # We need the'*2' here because affine_grid considers the top left corner as [-1, -1] and the bottom right one as [1, 1] (as opposed to the convention of this module where the top left corner is [0, 0])

    grid = F.affine_grid(
        transformation_matrix.unsqueeze(0), image_tensor.unsqueeze(0).size()
    )
    image_tensor = F.grid_sample(image_tensor.unsqueeze(0), grid)
    image_tensor = image_tensor.squeeze(0)

    return image_tensor
