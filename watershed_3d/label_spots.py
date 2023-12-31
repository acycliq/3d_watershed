from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from watershed_3d.base_logger import logger
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool


def label_spots(spots, masks):
    """

    Parameters
    ----------
    spots: Dataframe with yor spots. Columns must include the labels 'x', 'y', 'z_stack'
    masks: numpy array with the 3d segmentation masks

    Returns: A dataframe, same as the spots dataframe that was passed-in but with an extra
            column named 'label'. The values in this new column correspond to the label of
            the object (aka microglia) in the 'masks' numpy array that contains the
            corresponding spot. If label=0 then the spot is on the background.
    -------

    """
    z_stacks = np.unique(spots.z_stack)

    pool = ThreadPool(cpu_count())
    out = pool.map(wrapper(spots, masks), z_stacks)
    out = pd.concat(out)
    logger.info(out.shape)
    return out


# todo: decorate this!
def wrapper(spots, masks):
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
    masks_arr = np.load(r"../microglia/stitched_masks.npy")
    res = label_spots(spots_df, masks_arr)
    print(res.head())