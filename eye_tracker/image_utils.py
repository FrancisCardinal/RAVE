import torch
import numpy as np 

def tensor_to_opencv_image(tensor):
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor *= 255.0
    image = np.ascontiguousarray(tensor, dtype=np.uint8)
    return image


def inverse_normalize(tensor, mean, std):
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/17 
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor