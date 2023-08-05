from scipy.ndimage import generate_binary_structure
from numba import jit
import pandas as pd
import os
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import morphology
from skimage.measure import regionprops_table
import skimage.io
import fastremap
from scipy import ndimage
import diplib as dip
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from watershed_3d.utils import colourise, unpack
from watershed_3d.base_logger import logger


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


def intersection_over_union_2(masks_true, masks_pred):
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
    overlap = _label_overlap2(masks_true, masks_pred)

    # print('x.max is %d' % masks_true.max())
    # print('y.max is %d' % masks_pred.max())
    n_pixels_pred = overlap.sum(axis=0)
    n_pixels_true = overlap.sum(axis=1)

    x = np.array((n_pixels_true)).flatten()
    y = np.array((n_pixels_pred)).flatten()

    pred_plus_true = x[overlap.row] + y[overlap.col]
    iou_data = overlap.data /(pred_plus_true - overlap.data)
    iou = coo_matrix((iou_data, (overlap.row, overlap.col)), shape=overlap.shape)

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


# @jit(nopython=True)
def _label_overlap2(x, y):
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
    data = np.ones(len(x))
    n_rows = 1 + x.max().astype(np.int32)
    n_cols = 1 + y.max().astype(np.int32)
    overlap = coo_matrix((data, (x, y)), shape = (n_rows, n_cols), dtype=np.int32)
    # overlap = coo_matrix((1 + x.max().astype(np.int32), 1 + y.max().astype(np.int32)), dtype=np.int32 )
    # overlap.data[x, y] += 1
    overlap = overlap.tocsc().tocoo()
    return overlap


def stitch3D(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    for i in range(len(masks)-1):
        # logger.info('stitching mask %d ' % i)
        iou = intersection_over_union(masks[i + 1], masks[i])[1:, 1:]
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


def stitch3D_coo(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
    mmax = masks[0].max()
    for i in range(len(masks)-1):
        # logger.info('stitching mask %d ' % i)
        iou_coo = intersection_over_union_2(masks[i+1], masks[i])
        # iou = iou_coo.toarray()[1:, 1:]

        # remove now first column and first row from the coo matrix
        is_coord_zero = iou_coo.row * iou_coo.col
        row = iou_coo.row[is_coord_zero != 0] - 1
        col = iou_coo.col[is_coord_zero != 0] - 1
        data = iou_coo.data[is_coord_zero != 0]
        m,n = iou_coo.shape
        iou_coo = coo_matrix((data, (row, col)), shape=(m-1, n-1))
        if iou_coo.data.size > 0:
            # remove elements smaller than the threshold
            idx = iou_coo.data >= stitch_threshold
            iou_coo.data = iou_coo.data[idx]
            iou_coo.col = iou_coo.col[idx]
            iou_coo.row = iou_coo.row[idx]

            # iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou_coo.argmax(axis=1) + 1
            istitch = np.asarray(istitch).flatten()
            max_axis1 = iou_coo.max(axis=1)
            max_axis1_arr = max_axis1.toarray()
            ino = np.nonzero(max_axis1_arr == 0.0)[0]
            # ino_2 = np.nonzero(iou_coo.max(axis=1)==0.0)[0]
            # assert np.all(ino==ino_2)
            istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i+1] = istitch[masks[i+1]]
    return masks


def remove_small_cells(i, cell_labels, min_size=5):
    regions = pd.DataFrame(regionprops_table(cell_labels, properties=['label', 'area']))
    small_cells = regions[regions.area <= min_size]
    # logger.info("image: %d: Removing small shapes" % i)
    # logger.info("image: %d: Found %d shapes with area less than %d" % (i, small_cells.shape[0], min_size))

    # remove small cells
    cell_labels = fastremap.mask(cell_labels, small_cells.label.values)
    cell_labels, label_map = fastremap.renumber(cell_labels, in_place=False)


    p_cut = 2
    regions = pd.DataFrame(regionprops_table(cell_labels, properties=['label', 'area']))
    min_size = np.percentile(regions.area, p_cut)
    # logger.info("the %dth percentile is: %d" % (p_cut, min_size))
    small_cells = regions[regions.area <= min_size]

    cell_labels = fastremap.mask(cell_labels, small_cells.label.values)
    cell_labels, label_map = fastremap.renumber(cell_labels, in_place=False)

    return cell_labels.astype(np.uint64), label_map


def watershed(i, bw_img, opts):

    # target_dir = os.path.join(Path(opts['microglia_image']).parent, 'debug', 'bw_images')
    # Path(target_dir).mkdir(parents=True, exist_ok=True)
    # Image.fromarray(bw_img).save(os.path.join(target_dir, 'bw_img_%03d.jpg' % i))

    # shrink the shapes by a few pixels
    s = generate_binary_structure(2, 1)
    bw_eroded = morphology.erosion(bw_img == 255, s)
    bw_eroded = bw_eroded.astype(np.uint8)

    # for each shape get the distance from the closest zero-valued pixel
    distance = ndimage.distance_transform_edt(bw_eroded)

    # expand back the shapes..
    dilated_img = morphology.dilation(distance, s)

    # seeds: the brightest pixels of the shapes, typically close to the center of each shape.
    seeds = dip.Maxima(distance)
    cell_labels = dip.SeededWatershed(dip.Gauss(-distance), seeds, dilated_img > 0,
                                      maxDepth=distance.max(),
                                      flags={'labels'}
                                      )
    # logger.info('max distance %f' % distance.max())
    cell_labels = np.array(cell_labels)

    cell_labels, _ = remove_small_cells(i, cell_labels)
    return cell_labels


# def main(masks_url, background_url):
def main(bw_masks, image_3d, opts):
    # image_3d = skimage.io.imread(opts['background_url'])
    # image_3d = image_3d[20:30]

    # bw_masks = skimage.io.imread(opts['masks_url'])
    # bw_masks = bw_masks[20:30]

    ## for debugging use
    # background = background[21:23, 4024:4247, 1952:2226]
    # array_3d = array_3d[21:23, 4024:4247, 1952:2226]

    labels_list = []
    good_pages = []
    logger.info('Started watershed')
    for i, img in enumerate(bw_masks):
        if i in opts['exclude_pages']:
            # logger.info('Skipping watershed on page % d' % i)
            lbl = np.zeros(img.shape)
            labels_list.append(lbl)
        else:
            # logger.info('Doing watershed on page % d' % i)
            good_pages.append(i)
            lbl = watershed(i, img, opts)
            labels_list.append(lbl)
    labels = np.stack(labels_list)
    logger.info('Finished watershed')

    # logger.info('stitch3D starts')
    # stitched_labels = stitch3D(labels.astype(np.uint64), stitch_threshold=0.009)
    # logger.info('stitch3D finishes')

    logger.info('stitching the 2d masks')
    stitched_labels_2 = stitch3D_coo(labels.astype(np.uint64), stitch_threshold=0.009)
    logger.info('stitching finished')

    target_dir = os.path.join(Path(opts['microglia_image']).parent, 'debug')
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    out_npy = os.path.join(target_dir, 'stitched_masks.npy')
    np.save(out_npy, stitched_labels_2)
    logger.info('stitched_masks saved at %s' % out_npy)

    good_pages = good_pages[:20]
    rgb_masks = colourise(stitched_labels_2[good_pages], image_3d[good_pages])

    dir_name = os.path.join(target_dir, 'segmentation_samples')
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    unpack(rgb_masks, dir_name, mode="RGB", make_tiles=False, page_ids=good_pages)
    # # Image.fromarray(rgb_masks[10].astype(np.uint8), "RGB")
    logger.info("Saved some segmented images at %s" % dir_name)

    return stitched_labels_2


if __name__ == "__main__":
    # page_list = [46, 47, 48]
    bw_masks_tif = r".\microglia\bw_image.tiff"  # black and white image
    image_3d_tiff = r'.\microglia\adj_img.tiff'  # the background after all the adjustments
    main(bw_masks_tif, image_3d_tiff)

