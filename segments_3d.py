from scipy.ndimage import label, generate_binary_structure
import edt
import numpy as np
from skimage import morphology
import skimage.io
import diplib as dip
from utils import colourise, unpack


def main(bw_img):
    # Image.fromarray(bw_img).save(r'.\debug\bw_img.jpg')

    # shrink the shapes by a few pixels
    s = generate_binary_structure(3, 1)
    bw_eroded = morphology.erosion(bw_img == 255, s)
    bw_eroded = bw_eroded.astype(np.uint8)
    # Image.fromarray(bw_img).save(r'.\debug\bw_eroded.jpg')

    # for each shape get the distance from the closest zero-valued pixel
    distance = edt.edt(
        bw_eroded, anisotropy=(0.23, 0.23, 0.9),
        black_border=True, order='C',
        parallel=-1  # number of threads, <= 0 sets to num cpu
    )

    # expand back the shapes..
    dilated_img = morphology.dilation(distance, s)

    # seeds: the brightest pixels of the shapes, typically close to the center of each shape.
    seeds = dip.Maxima(distance)
    cell_labels = dip.SeededWatershed(-distance, seeds, dilated_img > 0,
                                      maxDepth=1.5,
                                      flags={'labels'}
                                      )
    print('max distance %f' % distance.max())
    cell_labels = np.array(cell_labels)
    return cell_labels


def app(masks_url, background_url):
    background = skimage.io.imread(background_url)
    background = background[20:40]

    array_3d = skimage.io.imread(masks_url)
    array_3d = array_3d[20:40]

    ## for debugging use
    # background = background[21:23, 4024:4247, 1952:2226]
    # array_3d = array_3d[21:23, 4024:4247, 1952:2226]

    labels = main(array_3d)

    rgb_masks = colourise(labels, background)
    unpack(rgb_masks, r"microglia_backround_images\masks3d_maxDepth1.5", mode="RGB", make_tiles=True)

    out_npy = r'microglia_backround_images\masks3d_maxDepth1.5\masks.npy'
    np.save(out_npy, labels)
    print('masks saved at %s' % out_npy)
    # Image.fromarray(rgb_masks[10].astype(np.uint8), "RGB")
    print("Done!")


    # array_3d = array_3d.transpose(2, 0, 1)

if __name__ == "__main__":
    # page_list = [46, 47, 48]
    masks_tif = "all_pages.tiff"  # black and white image
    all_img_adj_tiff = 'all_img_adj.tiff'  # the background after all the adjustments
    app(masks_tif, all_img_adj_tiff)

