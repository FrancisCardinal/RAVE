import torch
import numpy as np 

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


def inverse_normalize(tensor, 
                      mean, 
                      std):
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