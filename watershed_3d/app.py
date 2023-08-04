import numpy as np
import os
from watershed_3d.preprocess import stack_to_images
import watershed_3d.segment as segment
from watershed_3d.base_logger import logger

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def main(image_url=None, exclude_pages=None, do_rolling_ball=False):
    """
    Main entry point for 3d watershed segmentation
    image_url: Path to your 3d image to be segmented with watershed. Format must be ZYX
    do_rolling_ball: Boolean, if trye then the image will be filtered by the rolling-ball
                        algorithm to correct for uneven illumination/exposure. Use that on
                        extreme cases as it increases execution time massively
    exclude_pages:    List of integers denoting the pages to be excluded from the segmentation.

    returns an array of the same size as your image with the segmentation masks. Pages that
    have been excluded are totally black, iw all values are zero
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
    image_url = os.path.join(ROOT_DIR, '..', 'data', 'microglia_ WT997_icvAB_Iba1_retiled.tif')
    exclude_pages = np.hstack([np.arange(20), 25+np.arange(66-25)])

    main(image_url=image_url,
         exclude_pages=exclude_pages,
         do_rolling_ball=do_rolling_ball)
    logger.info('Done')
