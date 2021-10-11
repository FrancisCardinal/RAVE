import torch
from torch import sin, cos
import cv2
import numpy as np
from numpy import pi
from math import isclose


def ellipse_loss_function(predictions, targets):
    """Custom loss function for ellipse. This was developped because applying
       MSELoss directly on the parameters of ellipses was not giving good
       results in training sessions. This loss function generates points that
       lie on the target and predicted ellipse (using its parameters) and then
       computes the euclidean distance between each pair of points. The mean
       distance is then computed and used as the loss metric.

    Args:
        predictions (pytorch tensor): The predicted ellipses
        targets (pytorch tensor): The target ellipses

    Returns:
        float:
            The mean distance between the generated points of the predicted
            and target ellipses
    """
    NUMBER_OF_POINTS = 720

    x_predicted, y_predicted = get_points_of_ellipses(
        predictions, NUMBER_OF_POINTS
    )
    x_target, y_target = get_points_of_ellipses(targets, NUMBER_OF_POINTS)

    distance = torch.sqrt(
        (x_predicted - x_target) ** 2 + (y_predicted - y_target) ** 2
    )
    mean_distance = distance.sum() / NUMBER_OF_POINTS

    return mean_distance


def get_points_of_ellipses(ellipses, NUMBER_OF_POINTS):
    """generates points that lie on the ellipse (using its parameters).
       Points are generated using polar coordinates
       https://math.stackexchange.com/questions/2645689/what-is-the-parametric
       -equation-of-a-rotated-ellipse-given-the-angle-of-rotatio

    Args:
        ellipses (pytorch tensor): The parameters of the ellipse
        NUMBER_OF_POINTS (int): Number of points to generate

    Returns:
        tuple of pytorch tensor: The x and y coordinates of the points
    """
    DEVICE = ellipses.device
    NUMBER_OF_ELLIPSES = ellipses.size()[0]

    h, k, a, b, theta = (
        ellipses[:, 0],
        ellipses[:, 1],
        ellipses[:, 2],
        ellipses[:, 3],
        ellipses[:, 4],
    )
    theta = theta * 2 * pi
    h, k, a, b, theta = (
        h.unsqueeze(1),
        k.unsqueeze(1),
        a.unsqueeze(1),
        b.unsqueeze(1),
        theta.unsqueeze(1),
    )
    h, k, a, b, theta = (
        h.expand(-1, NUMBER_OF_POINTS),
        k.expand(-1, NUMBER_OF_POINTS),
        a.expand(-1, NUMBER_OF_POINTS),
        b.expand(-1, NUMBER_OF_POINTS),
        theta.expand(-1, NUMBER_OF_POINTS),
    )

    alpha = torch.linspace(
        0, 2 * pi, NUMBER_OF_POINTS, device=DEVICE, requires_grad=True
    )
    alpha = alpha.unsqueeze(0)
    alpha = alpha.expand(NUMBER_OF_ELLIPSES, -1)

    x = a * cos(alpha) * cos(theta) - b * sin(alpha) * sin(theta) + h
    y = a * cos(alpha) * sin(theta) + b * sin(alpha) * cos(theta) + k

    return x, y


def draw_ellipse_on_image(image, ellipse, color=(255, 0, 0), thickness=1):
    """Draw an ellipse on an image using its parameters

    Args:
        image (numpy array):
            The image on which to draw the ellipse
        ellipse (pytorch tensor):
            The parameters of the ellipse to draw
        color (tuple, optional):
            The color of the ellipse. Defaults to (255, 0, 0).
        thickness (int, optional):
            The thickness of the ellipse. Defaults to 1.

    Returns:
        numpy array: The image with the ellipse drawn on it
    """
    HEIGHT, WIDTH, _ = image.shape
    h, k, a, b, theta = ellipse
    h, k, a, b = h * WIDTH, k * HEIGHT, a * WIDTH, b * HEIGHT
    theta = theta * 2 * pi
    points = get_points_of_an_ellipse(h, k, a, b, theta, ellipse.device, 360)
    points = points.cpu().numpy()

    image = cv2.polylines(image, np.int32([points]), False, color, thickness)

    return image


def get_points_of_an_ellipse(h, k, a, b, theta, device, NUMBER_OF_POINTS):
    """generates points that lie on the ellipse (using its parameters)
       TODO : Change calls to this function to the 'get_points_of_ellipses'
       function instead, which uses paralellism to compute points faster

    Args:
        h (float): x coordinate of the center of the ellipse
        k (float): y coordinate of the center of the ellipse
        a (float): length of the horizontal axis
        b (float): length of the vertical axis
        theta (float): rotation angle of the ellipse relative to the x axis
        device (String): device on which to run the computations
        NUMBER_OF_POINTS (int): Number of points to generate

    Returns:
        tuple of pytorch tensor: The x and y coordinates of the points
    """
    output_points = torch.empty(
        (NUMBER_OF_POINTS, 2), device=device, requires_grad=True
    )

    alphas = torch.linspace(
        0, 2 * pi, NUMBER_OF_POINTS, device=device, requires_grad=True
    )

    current_point_index = 0
    for alpha in alphas:
        x = a * cos(alpha) * cos(theta) - b * sin(alpha) * sin(theta) + h
        y = a * cos(alpha) * sin(theta) + b * sin(alpha) * cos(theta) + k

        output_points.data[current_point_index, :] = torch.tensor(
            (x, y), device=device, requires_grad=True
        )
        current_point_index += 1

    return output_points


if __name__ == "__main__":
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    with torch.no_grad():
        x = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            device=DEVICE,
        )
        y = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            device=DEVICE,
        )
        assert isclose(ellipse_loss_function(x, y).cpu(), 0, rel_tol=1e-5)

        x = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            device=DEVICE,
        )
        y = torch.tensor(
            [[0.2, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            device=DEVICE,
        )
        assert isclose(ellipse_loss_function(x, y).cpu(), 0.1, rel_tol=1e-5)

        x = torch.tensor(
            [[0.1, 0.3, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            device=DEVICE,
        )
        y = torch.tensor(
            [[0.2, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            device=DEVICE,
        )
        assert isclose(
            ellipse_loss_function(x, y).cpu(),
            np.sqrt(0.1 ** 2 + 0.1 ** 2),
            rel_tol=1e-5,
        )

        x = torch.tensor(
            [[0.1, 0.3, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            device=DEVICE,
        )
        y = torch.tensor(
            [[0.2, 0.2, 0.3, 0.4, 0.5], [0.6, 0.6, 0.8, 0.9, 1.0]],
            device=DEVICE,
        )
        assert isclose(
            ellipse_loss_function(x, y).cpu(),
            np.sqrt(0.1 ** 2 + 0.1 ** 2) + 0.1,
            rel_tol=1e-5,
        )
