import torch
from torch import sin, cos
import cv2
import numpy as np
from numpy import pi 
from numpy import testing


def ellipse_loss_function(predictions, targets):
    NUMBER_OF_POINTS = 100

    predicted_points = get_points_of_ellipses(predictions, NUMBER_OF_POINTS)
    target_points    = get_points_of_ellipses(targets, NUMBER_OF_POINTS)

    distance = (torch.sqrt( (predicted_points-target_points)**2) ).sum()
    mean_distance = distance / NUMBER_OF_POINTS

    return mean_distance

def get_points_of_ellipses(ellipses, NUMBER_OF_POINTS): 
    DEVICE = ellipses.device

    NUMBER_OF_ELLIPSES = ellipses.size()[0]
    output_points = torch.empty((NUMBER_OF_ELLIPSES, NUMBER_OF_POINTS, 2), device=DEVICE)

    current_point_index = 0 
    for ellipse in ellipses: 
        h, k, a, b, theta = ellipse
        theta = theta*2*pi 
        points = get_points_of_an_ellipse(h, k, a, b, theta, DEVICE, NUMBER_OF_POINTS)

        output_points[current_point_index] = points
        current_point_index += 1 

    return output_points


def get_points_of_an_ellipse(h, k, a, b, theta, device, NUMBER_OF_POINTS):
    # J'utilise les coordonnées polaires pour générer les points qui appartiennent à l'ellipse ; voir https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio 
    output_points = torch.empty((NUMBER_OF_POINTS, 2), device=device)

    alphas = torch.linspace(0, 2*pi, NUMBER_OF_POINTS, device=device)
    current_point_index = 0 

    for alpha in alphas : 
        x = a*cos(alpha)*cos(theta) - b*sin(alpha)*sin(theta) + h
        y = a*cos(alpha)*sin(theta) + b*sin(alpha)*cos(theta) + k

        output_points[current_point_index] = torch.tensor( (x, y), device=device, requires_grad=True )
        current_point_index += 1 
    
    return output_points


def draw_ellipse_on_image(image, ellipse, color=(255, 0, 0), thickness=1):
    HEIGHT, WIDTH, _ = image.shape
    h, k, a, b, theta = ellipse 
    h, k, a, b = h*WIDTH, k*HEIGHT, a*WIDTH, b*HEIGHT
    theta = theta*2*pi 
    points = get_points_of_an_ellipse(h, k, a, b, theta, ellipse.device, 360)
    points = points.cpu().numpy()

    image = cv2.polylines(image, np.int32([points]), False, color, thickness)

    return image


if __name__=='__main__':
    DEVICE = 'cpu'
    if( torch.cuda.is_available() ): 
        DEVICE = 'cuda'

    with torch.no_grad():
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], device=DEVICE)
        y = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], device=DEVICE)
        testing.assert_allclose ( ellipse_loss_function(x, y).cpu(), 0 )

        x = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], device=DEVICE)
        y = torch.tensor([[0.2, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], device=DEVICE)
        testing.assert_allclose ( ellipse_loss_function(x, y).cpu(), 0.1  )

        x = torch.tensor([[0.1, 0.3, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], device=DEVICE)
        y = torch.tensor([[0.2, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], device=DEVICE)
        testing.assert_allclose ( ellipse_loss_function(x, y).cpu(), 0.2  )