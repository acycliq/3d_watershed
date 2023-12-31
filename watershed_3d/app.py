import numpy as np
import os
from watershed_3d.preprocess import preprocess
import watershed_3d.segment as segment
import watershed_3d.segment_3d as segment_3d
from watershed_3d.base_logger import logger

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def app(image_url=None, opts=None):
    """
    Main entry point for 3d watershed segmentation
    image_url: Path to your 3d image to be segmented with watershed. Format must be ZYX
    cfg: dictionary wth user-defined options. The keys of the dict can be:
        'exclude_pages             A list with the planes to ignore. Default is an empty list
        'mode'                     Either '2d_stitch' or '3d'
        'stitch_threshold'         If the overlap is bigger than 'stitch_threshold' then the label
                                   will be passed to the overlapping shape in the following plane.
                                   Default value is 0.009.
                                   This option is relevant only if 'mode' is '2d_stitch' otherwise
                                   it is ignored.
        'min_size'                 cells on any given 2d plane with area less than min_size are removed.
                                   Default value = 5px. Relevant only if 'mode' is '2d_stitch' otherwise
                                   it is ignored. If you do not want to use it in any case, set it to None
        'p_cut'                    Similar to the above. Cells on any given 2d plane with area less than
                                   p_cut percentile are removed. Default value is 2 which means the bottom
                                   0.02 percentile. If you do not want to use it in any case, set it to None
        'do_rolling_ball'          If True then the rolling ball filter will be applied on the image.
                                   Useful only in cases where locally there is very high brightness
                                   and the rest of the image or too dark. Default value is False
        'maxDepth'                 parameter that controls whether two shapes when their boundaries meet,
                                   will be merged or not. Is it is set too high then you will end up with an
                                   undersegmented image (few and very large shapes). If it is too low, then
                                   you will have an oversegmented image (too many and probably small
                                   shapes). maxDepth is only relevant if mode='3d', otherwise it is ignored.
                                   Default value is 1.0

    Note that the p_cut is applied after the min_size. Ie if mode: '2d_stitch' and both min_size, p_cut are
    not None, then we first filter on min_size and then we apply the p_cut

    Returns an array with the segmentation masks. It has same shape as your 3d image. Pages that
    have been excluded are totally black, ie all values are zero
    """

    if opts is None:
        opts = {
            'exclude_pages': [], 'do_rolling_ball': False, 'mode': '2d_stitch', 'stitch_threshold': 0.009,
            'maxDepth': 1.0, 'min_size': 5, 'p_cut':2
        }
    assert image_url is not None, "Need to pass the path to your image when you call app()"

    opts = {} if opts is None else opts

    # dict with default options
    cfg = {
        'microglia_image': image_url,
        'exclude_pages': [],
        'do_rolling_ball': False,
        'mode': '2d_stitch',
        'stitch_threshold': 0.009,
        'min_size': 5,
        'p_cut': 2,
        'maxDepth': 1.0
    }

    # update with user defined options
    cfg.update(opts)

    bw_arr, adj_arr = preprocess(cfg)
    if cfg['mode'] == '2d_stitch':
        out = segment.main(bw_arr, adj_arr, cfg)
    elif cfg['mode'] == '3d':
        out = segment_3d.main(bw_arr, adj_arr, cfg)
    else:
        raise Exception

    return out


if __name__ == "__main__":
    do_rolling_ball = False
    image_url = r"/media/dimitris/New Volume/data/Christina/MG_segmentation _test/microglia.tiff"
    mode = '3d'
    maxDepth = 1.5

    app(image_url=image_url, cfg={mode:mode, maxDepth:maxDepth})
    logger.info('Done')
