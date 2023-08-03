import numpy as np
import os
from watershed_3d.preprocess import stack_to_images
import watershed_3d.segment as segment
from watershed_3d.base_logger import logger

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def main(opts):
    """
    Main entry point for 3d watershed segmentation
    :param opts: Dictionary with the following keys:
        'do_rolling_ball': Boolean, if trye then the image will be filtered by the rolling-ball
                            algorithm to correcct for uneven illumination/exposure. Use that on
                            extreme cases as it increases execution time massively
        'microglia_image': Path to your 3d image to be segmented with watershed. Format must be ZYX
        'exclude_pages':    List of integers denoting the pages to be excluded from the segmentation.

    :return: An array of the same size as your image with the segmentation masks. Pages that
    have been excluded are totally black, iw all values are zero
    """

    bw_arr, adj_arr = stack_to_images(opts)
    out = segment.main(bw_arr, adj_arr, opts)
    return out


if __name__ == "__main__":
    opts = {
        # 'do_3d': False,
        'do_rolling_ball': False,
        'microglia_image':  os.path.join(ROOT_DIR, '..', 'data', 'microglia_ WT997_icvAB_Iba1_retiled.tif'),
        # 'exclude_pages': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65],
        'exclude_pages': np.hstack([np.arange(20), 25+np.arange(66-25)])
        # 'masks_url': r".\microglia\bw_image.tiff", # black and white image
        # 'background_url': r'.\microglia\adj_img.tiff',
    }
    main(opts)
    logger.info('Done')
