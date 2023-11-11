import numpy as np
import os
from watershed_3d.preprocess import preprocess
import watershed_3d.segment as segment
import watershed_3d.segment_3d as segment_3d
from watershed_3d.base_logger import logger

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def app(image_url=None, exclude_pages=[], do_rolling_ball=False, mode='2d_stitch', maxDepth=1.0):
    """
    Main entry point for 3d watershed segmentation
    image_url: Path to your 3d image to be segmented with watershed. Format must be ZYX
    do_rolling_ball: Boolean, if true then the image will be processed with the rolling-ball
                        algorithm to correct for uneven illumination/exposure. Use that on
                        extreme cases as it increases execution time massively
    exclude_pages:    List of integers denoting the pages to be excluded from the segmentation.
    mode: Should be either '2d_stitch' or '3d'
    'maxDepth': parameter that controls whether two shapes when their boundaries meet, will be
                merged or not. Is it is set too high then you will end up with an undersegmented
                image (few and very large shapes). If it is too low, then you will have an
                oversegmented image (too many and probably small shapes). maxDepth is only
                relevant if mode='3d', otherwise it is ignored.


    returns an array with the segmentation masks. It has same shape as your 3d image. Pages that
    have been excluded are totally black, ie all values are zero
    """
    assert image_url is not None, "Need to pass the path to your image when you call app()"

    opts = {'microglia_image': image_url,
            'exclude_pages': exclude_pages,
            'do_rolling_ball': do_rolling_ball,
            'mode': mode,
            'maxDepth': maxDepth
            }

    bw_arr, adj_arr = preprocess(opts)
    if mode == '2d_stitch':
        out = segment.main(bw_arr, adj_arr, opts)
    elif mode == '3d':
        out = segment_3d.main(bw_arr, adj_arr, opts)
    else:
        raise Exception

    return out


if __name__ == "__main__":
    do_rolling_ball = False
    image_url = r"E:\data\Christina\MG_segmentation _test\microglia.tiff"
    mode = '3d'
    maxDepth = 1.5

    app(image_url=image_url, mode=mode, maxDepth=maxDepth)
    logger.info('Done')
