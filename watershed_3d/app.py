import numpy as np
import os
from watershed_3d.preprocess import stack_to_images
import watershed_3d.segment as segment
from watershed_3d.base_logger import logger

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def app(image_url=None, exclude_pages=None, do_rolling_ball=False):
    """
    Main entry point for 3d watershed segmentation
    image_url: Path to your 3d image to be segmented with watershed. Format must be ZYX
    do_rolling_ball: Boolean, if true then the image will be processed with the rolling-ball
                        algorithm to correct for uneven illumination/exposure. Use that on
                        extreme cases as it increases execution time massively
    exclude_pages:    List of integers denoting the pages to be excluded from the segmentation.

    returns an array with the segmentation masks. It has same shape as your 3d image. Pages that
    have been excluded are totally black, ie all values are zero
    """
    assert image_url is not None, "Need to pass the path to your image when you call app()"

    opts = {'microglia_image': image_url,
            'exclude_pages': exclude_pages,
            'do_rolling_ball': do_rolling_ball}

    bw_arr, adj_arr = stack_to_images(opts)
    out = segment.main(bw_arr, adj_arr, opts)
    return out


if __name__ == "__main__":
    do_rolling_ball = False
    image_url = r"/home/dimitris/Desktop/christina/microglia_ WT997_icvAB_Iba1_retiled.tif"
    bad_pages = list(range(0, 16)) + [65, 66]
    # image_url = os.path.join(ROOT_DIR, '..', 'data', 'microglia_ WT997_icvAB_Iba1_retiled.tif')
    # exclude_pages = np.hstack([np.arange(20), 25+np.arange(66-25)])

    app(image_url=image_url,
         exclude_pages=bad_pages,
         do_rolling_ball=False)
    logger.info('Done')
