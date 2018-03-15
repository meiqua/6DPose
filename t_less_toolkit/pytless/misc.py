# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import numpy as np
from PIL import Image, ImageDraw

def ensure_dir(path):
    """
    Ensures that the specified directory exists (it is created if it does not
    exist).

    :param path: Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def draw_rect(im, rect, color=(255, 255, 255)):
    """
    Draws a rectangle to the given image.

    :param im: Image in which the rectangle will be drawn (uint8, from 0 to 255).
    :param rect: Rectangle specified by [x, y, width, height], where [x, y]
                 is the top-left corner of the rectangle.
    :param color: Color of the rectangle.
    :return: Image with the drawn rectangle.
    """
    vis_pil = Image.fromarray(im)
    draw = ImageDraw.Draw(vis_pil)
    draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
                   outline=color, fill=None)
    del draw
    return np.array(vis_pil)
