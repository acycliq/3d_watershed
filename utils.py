import os
import pciSeq
import numpy as np
from PIL import Image
from skimage.color import label2rgb


def rgb2hex(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def pallete(id):
    if id % 24 == 0:
        return (2, 63, 165)
    elif id % 24 == 1:
        return (125, 135, 185)
    elif id % 24 == 2:
        return (190, 193, 212)
    elif id % 24 == 3:
        return (214, 188, 192)
    elif id % 24 == 4:
        return (187, 119, 132)
    elif id % 24 == 5:
        return (142, 6, 59)
    elif id % 24 == 6:
        return (74, 111, 227)
    elif id % 24 == 7:
        return (133, 149, 225)
    elif id % 24 == 8:
        return (181, 187, 227)
    elif id % 24 == 9:
        return (230, 175, 185)
    elif id % 24 == 10:
        return (224, 123, 145)
    elif id % 24 == 11:
        return (211, 63, 106)
    elif id % 24 == 12:
        return (17, 198, 56)
    elif id % 24 == 13:
        return (141, 213, 147)
    elif id % 24 == 14:
        return (198, 222, 199)
    elif id % 24 == 15:
        return (234, 211, 198)
    elif id % 24 == 16:
        return (240, 185, 141)
    elif id % 24 == 17:
        return (239, 151, 8)
    elif id % 24 == 18:
        return (15, 207, 192)
    elif id % 24 == 19:
        return (156, 222, 214)
    elif id % 24 == 20:
        return (213, 234, 231)
    elif id % 24 == 21:
        return (243, 225, 235)
    elif id % 24 == 22:
        return (246, 196, 225)
    elif id % 24 == 23:
        return (247, 156, 212)


def get_colour(labels=None):
    if labels is None:
        labels = np.arange(24)

    rgb = [[c/255 for c in pallete(d)] for d in labels]
    return rgb


def colourise(cell_labels, img=None):
    overlay = label2rgb(cell_labels, image=img, colors=get_colour(), bg_label=0)
    overlay = 255 * overlay
    return overlay.astype(np.uint8)


def overlay(cell_labels, background):
    out = label2rgb(cell_labels, image=background, colors=get_colour(), bg_label=0)
    out = 255 * out
    out = out.astype(np.uint8)
    return out


def unpack(stack, out_dir, mode=None, make_tiles=False):
    '''
    reads a 3d tiff image and unpacks it
    :param stack:
    :param out_dir:
    :return:
    '''
    img_depth = stack.shape[0]
    print("image has %d pages" %  img_depth)
    for n in range(img_depth):
        img = stack[n].astype(np.uint8)
        fName = os.path.join(out_dir, "page_%03d.jpg" % n)
        Image.fromarray(img, mode=mode).save(fName)
        print("Image was saved at %s" % fName)
        tiles_dir = os.path.join(out_dir, "tiles", "page_%03d" % n)
        if (img.max() > 0) and make_tiles:
            pciSeq.tile_maker(fName, z_depth=7, out_dir=tiles_dir)
            print("tiles created at was saved at %s" % tiles_dir)
        else:
            print("Image %s is totally empty. Skipping the tile maker..." % fName)
