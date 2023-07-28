from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from base_logger import logger
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool


def label_spots(spots, masks):
    z_stacks = np.unique(spots.z_stack)

    pool = ThreadPool(cpu_count())
    out = pool.map(loop_body(spots, masks), z_stacks)
    out = pd.concat(out)
    logger.info(out.shape)
    return out


def loop_body(spots, masks):
    def inner_fun(z):
        df = spots[spots.z_stack == z].copy()
        mask = masks[int(np.floor(z))]
        mask_csr = csr_matrix(mask)
        logger.info('doing %d' % z)
        label = mask_csr[df.y, df.x]
        label = np.asarray(label)[0]
        df['label'] = label
        return df
    return inner_fun


if __name__ == "__main__":
    spots_url = r"F:\data\Christina\3D_watershed\spots_WT997_icvAB_OMP_restitched.csv"
    spots_df = pd.read_csv(spots_url)
    masks = np.load(r".\microglia\stitched_masks.npy")
    out = label_spots(spots_df, masks)
    print(out.head())