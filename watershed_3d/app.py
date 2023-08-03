import numpy as np
import os
from watershed_3d.preprocess import stack_to_images
import watershed_3d.segment as segment
from watershed_3d.base_logger import logger

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def main(opts):
    bw_arr, adj_arr = stack_to_images(opts)
    segment.main(bw_arr, adj_arr, opts)


if __name__ == "__main__":
    opts = {
        # 'do_3d': False,
        'do_rolling_ball': False,
        'microglia_image':  os.path.join(ROOT_DIR, '../data', 'microglia_ WT997_icvAB_Iba1_retiled.tif'),
        # 'exclude_pages': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65],
        'exclude_pages': np.hstack([np.arange(20), 25+np.arange(66-25)])
        # 'masks_url': r".\microglia\bw_image.tiff", # black and white image
        # 'background_url': r'.\microglia\adj_img.tiff',
    }
    main(opts)
    logger.info('Done')
