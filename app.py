from preprocess import stack_to_images
import segment
from base_logger import logger


def main(opts):
    if opts['do_preprocess']:
        key = 'microglia'
        for val in opts.key(key):
            stack_to_images({key: val})

    segment.main(opts)


if __name__ == "__main__":
    opts = {
        'do_3d': False,
        'do_preprocess': False,
        'microglia':  r"D:\microglia_ WT997_icvAB_Iba1_retiled.tif",
        'exclude_pages': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 65],
        'masks_url': r".\microglia\bw_image.tiff", # black and white image
        'background_url': r'.\microglia\adj_img.tiff',
    }
    main(opts)
    logger.info('Done')
