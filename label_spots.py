from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import pandas as pd
from base_logger import logger
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool

def app():
    spots_url = r"F:\data\Christina\3D_watershed\spots_WT997_icvAB_OMP_restitched.csv"
    spots = pd.read_csv(spots_url)
    print(spots.shape)

    masks = np.load(r".\microglia\stitched_masks.npy")

    z_stacks = np.unique(spots.z_stack)
    temp = []
    for z in z_stacks:
        df = spots[spots.z_stack == z].copy()
        mask = masks[int(np.floor(z))]
        mask_csr = csr_matrix(mask)
        logger.info('doing %d' % z)
        label = mask_csr[df.y, df.x]
        label = np.asarray(label)[0]
        df['label'] = label
        temp.append(df)

    out = np.vstack(temp)
    logger.info(out.shape)

    logger.info('Start par')
    pool = ThreadPool(cpu_count())
    out = pool.map(loop_body(spots, masks), z_stacks)
    out = pd.concat(out)
    logger.info('end par')
    logger.info('ok')
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
    app()