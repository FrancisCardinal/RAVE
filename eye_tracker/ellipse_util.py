import torch
from torch import sin, cos
from numpy import pi 
from tqdm import tqdm

def ellipse_loss_function(predictions, targets):
    NUMBER_OF_POINTS = 100
    DEVICE = predictions.device

    predicted_points = get_points_of_ellipses(predictions, DEVICE, NUMBER_OF_POINTS)
    target_points    = get_points_of_ellipses(targets, DEVICE, NUMBER_OF_POINTS)

    distance = ((predicted_points-target_points)**2).sum()
    mean_distance = distance / NUMBER_OF_POINTS

    return mean_distance

def get_points_of_ellipses(ellipses, device, NUMBER_OF_POINTS): 
    NUMBER_OF_ELLIPSES = ellipses.size()[0]
    output_points = torch.empty((NUMBER_OF_ELLIPSES, NUMBER_OF_POINTS, 2), device=device)

    current_point_index = 0 
    for ellipse in ellipses: 
        h, k, a, b, theta = ellipse
        theta = theta*2*pi 
        points = get_points_of_an_ellipse(h, k, a, b, theta, device, NUMBER_OF_POINTS)

        output_points[current_point_index] = points
        current_point_index += 1 

    return output_points


def get_points_of_an_ellipse(h, k, a, b, theta, device, NUMBER_OF_POINTS = 300):
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