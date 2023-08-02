# reads the dapi and microglia image stacks, adjust the image for
# brightness and contrast and then saves the idividual pages on the disk

import skimage.io
import numpy as np
from pathlib import Path
import os
import tifffile as tif
from utils import adjust_image
from base_logger import logger


def stack_to_images(cfg):
    filename = cfg['microglia_image']
    image_3d = skimage.io.imread(filename)
    n = image_3d.shape[0]

    mask_list = []
    img_list = []
    for i in range(n):
        if i in cfg['exclude_pages']:
            logger.info('Skipping page %d. It is in the exclude list' % i)
            bw_mask = np.zeros(image_3d.shape[1:])
            adj_img = np.zeros(image_3d.shape[1:])
        else:
            logger.info('adjusting page %d out of %d' % (i + 1, n))
            bw_mask, adj_img = adjust_image(image_3d[i], i, cfg)
        mask_list.append(bw_mask)
        img_list.append(adj_img)

    target_dir = os.path.join(Path(filename).parent, 'debug')
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    bw_arr = np.stack(mask_list).astype(np.uint8)
    tif.imwrite(os.path.join(target_dir, 'bw_image.tiff'), bw_arr, bigtiff=True)
    logger.info('black and white image saved at %s' % os.path.join(target_dir, 'bw_image.tiff'))

    adj_arr = np.stack(img_list).astype(np.uint8)
    tif.imwrite(os.path.join(target_dir, 'adj_img.tiff'), adj_arr, bigtiff=True)
    logger.info('curve-adjusted image saved at %s' % os.path.join(target_dir, 'adj_img.tiff'))

    return bw_arr, adj_arr


if __name__ == "__main__":
    dic = {
        # 'dapi': r"F:\data\Christina\3D_watershed\DAPI_retiled_image.tif",
        'microglia': r"D:\microglia_ WT997_icvAB_Iba1_retiled.tif"
    }
    for di in dic.items():
        stack_to_images(di)

    logger.info("Done!")
