from scipy.ndimage import label, generate_binary_structure
from numba import jit
import os
import pandas as pd
import pciSeq
import numpy as np
from PIL import Image
from skimage import morphology
from skimage.measure import regionprops_table
import skimage.io
import fastremap
from scipy import ndimage
from skimage.color import label2rgb
import diplib as dip


def intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def stitch3D(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    for i in range(len(masks)-1):
        print('stitching mask %d ' % i)
        iou = intersection_over_union(masks[i+1], masks[i])[1:,1:]
        if iou.size > 0:
            iou[iou < stitch_threshold] = 0.0
            # iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1)==0.0)[0]
            istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i+1] = istitch[masks[i+1]]
    return masks

def remove_small_cells(cell_labels, min_size=5):
    # p_cut = 2
    regions = pd.DataFrame(regionprops_table(cell_labels, properties=['label', 'area']))
    # min_size = np.percentile(regions.area, p_cut)
    # logger.info("the %dth percentile is: %d" % (p_cut, min_size))
    small_cells = regions[regions.area <= min_size]
    print("found %d cells with area less than %d" % (small_cells.shape[0], min_size))

    # remove small cells
    cell_labels = fastremap.mask(cell_labels, small_cells.label.values)
    cell_labels, label_map = fastremap.renumber(cell_labels, in_place=False)
    return cell_labels.astype(np.uint64), label_map

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


def main(bw_img):
    Image.fromarray(bw_img).save(r'.\debug\bw_img.jpg')

    # shrink the shapes by a few pixels
    s = generate_binary_structure(2, 1)
    bw_eroded = morphology.erosion(bw_img == 255, s)
    bw_eroded = bw_eroded.astype(np.uint8)
    Image.fromarray(bw_img).save(r'.\debug\bw_eroded.jpg')

    # for each shape get the distance from the closest zero-valued pixel
    distance = ndimage.distance_transform_edt(bw_eroded)

    # expand back the shapes..
    dilated_img = morphology.dilation(distance, s)

    # seeds: the brightest pixels of the shapes, typically close to the center of each shape.
    seeds = dip.Maxima(distance)
    cell_labels = dip.SeededWatershed(-distance, seeds, dilated_img > 0,
                                      maxDepth=distance.max(),
                                      flags={'labels'}
                                      )
    print('max distance %f' % distance.max())
    cell_labels = np.array(cell_labels)
    cell_labels, _ = remove_small_cells(cell_labels)
    # overlay = label2rgb(cell_labels, image=background, colors=get_colour(), bg_label=0)
    # overlay = 255 * overlay
    # overlay = overlay.astype(np.uint8)
    # Image.fromarray(overlay, mode='RGB').save(r'.\debug\overlay.jpg')
    # Image.fromarray(overlay).show()
    # print('ok')
    return cell_labels


def colourise(cell_labels, img=None):
    overlay = label2rgb(cell_labels, image=img, colors=get_colour(), bg_label=0)
    overlay = 255 * overlay
    # img_labels = Image.fromarray(overlay.astype(np.uint8))
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

def app(masks_url, background_url):
    background = skimage.io.imread(background_url)
    background = background[20:40]

    array_3d = skimage.io.imread(masks_url)
    array_3d = array_3d[20:40]

    ## for debugging use
    # background = background[21:23, 4024:4247, 1952:2226]
    # array_3d = array_3d[21:23, 4024:4247, 1952:2226]

    labels_list = []
    for i, img in enumerate(array_3d):
        lbl = main(img)
        labels_list.append(lbl)
    labels = np.stack(labels_list)

    stitched_labels = stitch3D(labels.astype(np.uint64), stitch_threshold=0.009)
    rgb_masks = colourise(stitched_labels, background)
    unpack(rgb_masks, r"microglia_backround_images\mask_tiles_stitched_fixed_v4", mode="RGB", make_tiles=True)

    out_npy = r'microglia_backround_images\mask_tiles_stitched_fixed_v4\stitched_masks.npy'
    np.save(out_npy, stitched_labels)
    print('stitched_masks saved at %s' % out_npy)
    # Image.fromarray(rgb_masks[10].astype(np.uint8), "RGB")
    print("Done!")


    # array_3d = array_3d.transpose(2, 0, 1)

if __name__ == "__main__":
    # page_list = [46, 47, 48]
    masks_tif = "all_pages.tiff"  # black and white image
    all_img_adj_tiff = 'all_img_adj.tiff'  # the background after all the adjustments
    app(masks_tif, all_img_adj_tiff)

