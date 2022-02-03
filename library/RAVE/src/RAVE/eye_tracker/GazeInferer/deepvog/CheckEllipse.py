import numpy as np
def checkEllipse(x,y, c1,c2,w,h,radian):
    x_can = (x-c1) * np.cos(radian) + (y - c2) * np.sin(radian)
    y_can = -(x-c1) * np.sin(radian) + (y - c2) * np.cos(radian)
    output = np.power(x_can/w,2) + np.power(y_can/h,2)
    return output

def computeEllipseConfidence(prediction, center, ellipse_w, ellipse_h, radian):
    return 0.99