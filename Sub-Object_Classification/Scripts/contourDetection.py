from astropy.io import fits

import cv2 as cv
import numpy as np

import argparse
import random
from os import path as os_path
from tqdm import tqdm

from _helpers import make_directory, prepareFits

import warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings(action='ignore', category=AstropyUserWarning)
warnings.filterwarnings(action='ignore', category=FITSFixedWarning)


def getContourEdges(contour, src_width):
    min_vert = 71
    min_hor = 10
    align_l = 5
    align_r = 6
    min_height = 20
    min_width = 3
    max_width = 20
    max_height = 140

    x1, x2, mx, y1, y2 = contour[:, 0, 0].min(), contour[:, 0, 0].max() + 1,\
        contour[:, 0, 0].mean(), contour[:, 0, 1].min(), contour[:, 0, 1].max() + 1

    height = y2 - y1
    width = x2 - x1
    if (min_height <= height <= max_height) and (min_width <= width <= max_width) and height > width:
        if height < min_vert:
            y1 = max(0, y2 - min_vert)
        if width < min_hor:
            if mx - x1 < align_l:
                # print("Left alignment:", align_l - (cx - x1))
                x1 = max(0, mx - align_l)
            if x2 - mx < align_r:
                # print("Right alignment:", align_r - (x2 - cx))
                x2 = min(src_width, mx + align_r)
        return np.round(np.array([y1, x1, y2, x2], dtype='int'))
    else:
        return None, None, None, None


def main():
    ## blockSize: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7,
    # and so on.
    ## C: Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may
    # be zero or negative as well.

    block_size = 11
    constant = 2

    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', default='train', choices=('train', 'test'))
    # parser.add_argument('--input_csv', default='Subtypes', choices=('Subtypes', 'Combined'))
    # parser.add_argument('--class_column', default='Sp type', choices=('Sp type', 'Cl'))
    parser.add_argument('--data_root', default='/media/sargis/Data/Stepan_Home/Datasets/Stepan_Datasets/DFBS')
    parser.add_argument('--fits_path', default=None)
    parser.add_argument('--headers_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--num_samples', default=4 * 1e6)

    args = parser.parse_args()

    num_samples = int(args.num_samples)
    data_root = args.data_root
    fits_path = args.fits_path or os_path.join(data_root, 'fits_files')
    headers_path = args.headers_path or os_path.join(data_root, 'fits_headers')
    output_dir = args.headers_path or os_path.join(data_root, 'extractedContours_4m')

    print('Number of samples to be extracted:', num_samples)

    fits_headers, fits_set = prepareFits(
        headers_path=headers_path,
        fits_path=fits_path,
        headers_pattern="*.fits.hdr",
        fits_pattern="*.fits")

    # fits_count = 0
    extracted_count = 0
    n_headers = len(fits_headers)
    random.shuffle(fits_headers)

    make_directory(output_dir)

    for i in tqdm(range(n_headers)):
        plate = fits_headers[i].split('/')[-1].split('.')[0]

        if plate in fits_set:
            fbs_plate = fits.open(f'{fits_path}/{plate}.fits')

            plate_img = np.array(fbs_plate[0].data, dtype=np.uint16)
            height, width = plate_img.shape
            del fbs_plate

            scaled_img = ((plate_img - plate_img.min()) / (plate_img.max() - plate_img.min()) * 255).astype(np.uint8)

            if np.mean(scaled_img) < 127.5:
                scaled_img = np.invert(scaled_img)

            gblur = cv.GaussianBlur(scaled_img, (3, 3), 2, 2)
            del scaled_img

            g_th = cv.adaptiveThreshold(gblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY_INV, blockSize=block_size, C=constant)
            del gblur

            contours, hierarchy = cv.findContours(g_th, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            for index, contour in enumerate(tqdm(contours)):
                y1, x1, y2, x2 = getContourEdges(contour, width)

                if all([y1, x1, y2, x2]):
                    result = plate_img[y1:y2, x1:x2].copy()
                    # result_sized = cv.resize(result, (20, 140))

                    image_path = f'{output_dir}/{plate}__{index}.tiff'

                    cv.imwrite(image_path, result)

                    extracted_count += 1
                    if extracted_count > num_samples:
                        break

            # fits_count += 1
        # if fits_count > 5:
        if extracted_count > num_samples:
            break

    print('\nExtraction Completed...\n')


if __name__ == '__main__':
    main()
