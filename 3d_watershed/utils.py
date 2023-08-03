import os
import cv2
import pciSeq
import numpy as np
from PIL import Image
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from skimage import morphology, restoration
from base_logger import logger


def rgb2hex(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def pallete(id):
    if id % 24 == 0:
        return 2, 63, 165
    elif id % 24 == 1:
        return 125, 135, 185
    elif id % 24 == 2:
        return 190, 193, 212
    elif id % 24 == 3:
        return 214, 188, 192
    elif id % 24 == 4:
        return 187, 119, 132
    elif id % 24 == 5:
        return 142, 6, 59
    elif id % 24 == 6:
        return 74, 111, 227
    elif id % 24 == 7:
        return 133, 149, 225
    elif id % 24 == 8:
        return 181, 187, 227
    elif id % 24 == 9:
        return 230, 175, 185
    elif id % 24 == 10:
        return 224, 123, 145
    elif id % 24 == 11:
        return 211, 63, 106
    elif id % 24 == 12:
        return 17, 198, 56
    elif id % 24 == 13:
        return 141, 213, 147
    elif id % 24 == 14:
        return 198, 222, 199
    elif id % 24 == 15:
        return 234, 211, 198
    elif id % 24 == 16:
        return 240, 185, 141
    elif id % 24 == 17:
        return 239, 151, 8
    elif id % 24 == 18:
        return 15, 207, 192
    elif id % 24 == 19:
        return 156, 222, 214
    elif id % 24 == 20:
        return 213, 234, 231
    elif id % 24 == 21:
        return 243, 225, 235
    elif id % 24 == 22:
        return 246, 196, 225
    elif id % 24 == 23:
        return 247, 156, 212


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


def unpack(stack, out_dir, mode=None, make_tiles=False, page_ids=None):
    '''
    reads a 3d tiff image and unpacks it
    :param stack:
    :param out_dir:
    :return:
    '''
    img_depth = stack.shape[0]
    logger.info("image has %d pages" %  img_depth)
    for n in range(img_depth):
        img = stack[n].astype(np.uint8)
        if page_ids is not None:
            page_num = page_ids[n]
        else:
            page_num = n
        fName = os.path.join(out_dir, "page_%03d.jpg" % page_num)
        Image.fromarray(img, mode=mode).save(fName)
        logger.info("Image was saved at %s" % fName)
        tiles_dir = os.path.join(out_dir, "tiles", "page_%03d" % page_num)
        if (img.max() > 0) and make_tiles:
            pciSeq.tile_maker(fName, z_depth=7, out_dir=tiles_dir)
            logger.info("tiles created at was saved at %s" % tiles_dir)
        else:
            logger.info("Image %s is totally empty or make_tiles is set to False. Skipping the tile maker..." % fName)


def adjust_image(img, n, cfg):
    '''
    takes in an image (grayscale, uint8) and adds some contrast.
    Returns an array uint8
    :param img_url:
    :return:
    '''
    img = normalize_img(img)
    lims = _stretchlim(img)
    dapi_adj = _imadjust2(img, lims)

    if cfg['do_rolling_ball']:
        logger.info("rolling_ball")
        background = restoration.rolling_ball(dapi_adj)
        Image.fromarray(background.astype(np.uint8)).save("background_%d.jpg" % n)
        dapi_restored = dapi_adj - background
        # Image.fromarray(dapi_restored.astype(np.uint8)).save("dapi_restored.tif")
    else:
        dapi_restored = dapi_adj

    # logger.info("adaptiveThreshold")
    mask = cv2.adaptiveThreshold(dapi_restored.astype(np.uint8),
                                 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 101,
                                 0)
    return mask, dapi_adj


def normalize_img(img, normalize=[0, 99.9]):
    perc1, perc2 = np.percentile(img, list(normalize))
    img = img - perc1
    img /= (perc2-perc1)
    img = img * 255.0
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def _stretchlim(img):
    # adjust contrast
    nbins = 255
    tol_low = 0.01
    tol_high = 0.99
    sz = np.shape(img)
    if len(sz) == 2:
        img = img[:, :, None]
        sz = np.shape(img)

    p = sz[2]
    ilowhigh = np.zeros([2, p])
    for i in range(0, p):
        hist, bins = np.histogram(img[:, :, i].ravel(), nbins + 1, [0, nbins])
        cdf = np.cumsum(hist) / sum(hist)
        ilow = np.argmax(cdf > tol_low)
        ihigh = np.argmax(cdf >= tol_high)
        if ilow == ihigh:
            ilowhigh[:, i] = np.array([1, nbins])
        else:
            ilowhigh[:, i] = np.array([ilow, ihigh])

    lims = ilowhigh / nbins
    return lims


def _imadjust2(img, lims):
    # adjusts intensity
    lims = lims.flatten()
    img2 = np.copy(img)
    lowIn = lims[0]
    highIn = lims[1]
    lowOut = 0
    highOut = 1
    gamma = 1
    lut = _adjustWithLUT(lowIn, highIn, lowOut, highOut, gamma)
    return lut[img2].astype(np.uint8)

def _adjustWithLUT(lowIn, highIn, lowOut, highOut, gamma):
    lutLength = 256  # assumes uint8
    lut = np.linspace(0, 1, lutLength)
    lut = _adjustArray(lut, lowIn, highIn, lowOut, highOut, gamma)
    lut = _img_as_ubyte(lut)
    return lut

def _adjustArray(img, lIn, hIn, lOut, hOut, g):
    # %make sure img is in the range [lIn;hIn]
    img = np.maximum(lIn, np.minimum(hIn, img));

    out = ((img - lIn) / (hIn - lIn)) ** g
    out = out ** (hOut - lOut) + lOut
    return out


def _img_as_ubyte(x):
    out = np.zeros(x.shape)
    out[x == 0.3] = 77
    out[x != 0.3] = img_as_ubyte(x)[x != 0.3]
    return out

