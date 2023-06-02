import cv2 as cv
from astropy.io import fits
import numpy as np
from os import listdir
from fnmatch import fnmatch
import random
from _helpers import make_directory

from tqdm import tqdm


def prepareFits(headers_path, fits_path, headers_pattern, fits_pattern):
    headers_folder = listdir(headers_path)
    fits_folder = listdir(fits_path)

    fits_headers = []
    fits_files = []

    headers_pattern = headers_pattern
    fits_pattern = fits_pattern

    for entry in headers_folder:
        if fnmatch(entry, headers_pattern):
            fits_headers.append('./data/fits_headers/' + entry)

    for entry in fits_folder:
        if fnmatch(entry, fits_pattern):
            fits_files.append('./data/fits_files/' + entry)

    fits_headers = np.array(fits_headers)
    fits_files = np.array(fits_files)
    fits_set = set(map(lambda x: x.split('/')[-1].split('.')[0], fits_files))

    return fits_headers, fits_files, fits_set


def getContourEdges(contour, shape_x):
    min_vert = 71
    min_hor = 10
    align_l = 5
    align_r = 6
    min_height = 20
    min_width = 3
    max_width = 20
    max_height = 140

    x1, x2, mx, y1, y2 = contour[:, 0, 0].min(), contour[:, 0, 0].max() + 1, contour[:, 0, 0].mean(), contour[:, 0, 1].min(), contour[:, 0, 1].max() + 1
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
                x2 = min(shape_x, mx + align_r)
        return np.round(np.array([y1, x1, y2, x2], dtype='int'))
    else:
        return None, None, None, None


fits_headers, fits_files, fits_set = prepareFits(
    headers_path='data/fits_headers',
    fits_path='data/fits_files',
    headers_pattern="*.hdr",
    fits_pattern="*.fits")

fits_count = 0


random.shuffle(fits_headers)

make_directory('data/extractedContours/')
make_directory('data/extractedContours_2/')

for i in tqdm(range(len(fits_headers))):
    plate = fits_headers[i].split('/')[-1].split('.')[0]
    if plate in fits_set:
        fbs_plate = fits.open('./data/fits_files/' + plate + '.fits')

        plate_img = np.array(fbs_plate[0].data, dtype=np.uint16)
        shape_y, shape_x = plate_img.shape
        del fbs_plate

        scaled_img = ((plate_img - plate_img.min()) / (plate_img.max() - plate_img.min()) * 255).astype(np.uint8)

        if np.mean(scaled_img) < 127.5:
            scaled_img = np.invert(scaled_img)

        gblur = cv.GaussianBlur(scaled_img, (3, 3), 2, 2)
        del scaled_img

        g_th = cv.adaptiveThreshold(gblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY_INV, 11, 2)
        del gblur

        contours, hierarchy = cv.findContours(g_th, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        for index, contour in tqdm(enumerate(contours)):
            y1, x1, y2, x2 = getContourEdges(contour, shape_x)

            if all([y1, x1, y2, x2]):
                result = plate_img[y1:y2, x1:x2].copy()
                result_sized = cv.resize(result, (20, 140))

                image_path = f'data/extractedContours/{plate}__{index}.png'
                image_2_path = f'data/extractedContours_2/{plate}__{index}.png'

                cv.imwrite(image_path, result)
                cv.imwrite(image_2_path, result_sized)

        fits_count += 1
    if fits_count > 5:
        break
