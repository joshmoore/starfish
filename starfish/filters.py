import numpy as np
import cv2
from skimage import restoration


def gaussian_low_pass(img, sigma, ksize=None, border=None):
    img_swap = swap(img)
    if ksize is None:
        ksize = int(2 * np.ceil(2 * sigma) + 1)

    if border is None:
        border = cv2.BORDER_REPLICATE

    blurred = cv2.GaussianBlur(img_swap,
                               (ksize, ksize),
                               sigma,
                               borderType=border
                               )

    blurred = blurred.astype(np.uint16)
    return swap(blurred)


def gaussian_high_pass(img, sigma, ksize=None, border=None):
    blurred = gaussian_low_pass(img, sigma, ksize, border)

    over_flow_ind = img < blurred
    res = img - blurred
    res[over_flow_ind] = 0

    return res


def richardson_lucy_deconv(img, psf, num_iter, clip=False):
    img_swap = swap(img)
    img_deconv = restoration.richardson_lucy(img_swap, psf, iterations=num_iter, clip=clip)

    # here be dragons. img_deconv is a float. this should not work, but the result looks nice
    # modulo boundary values? wtf indeed.
    img_deconv = img_deconv.astype(np.uint16)
    return swap(img_deconv)


def swap(img):
    img_swap = img.swapaxes(0, img.ndim - 1)
    return img_swap