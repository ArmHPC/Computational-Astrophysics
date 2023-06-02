# Imports
import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv

from astropy.io import fits
from astropy import wcs

from pavlidis import pavlidis

from os import mkdir
from shutil import rmtree
from fnmatch import fnmatch
from os import listdir

from tqdm import tqdm


def loadImages(X):
    images_list = []

    for i, path in enumerate(X):
        im = Image.open(path)
        arr = np.array(im)

        arr = (arr-arr.min())/(arr.max()-arr.min())

        images_list.append(arr)

    return np.array(images_list)


def prepareData(path):
    data = pd.read_csv(path)
    if 'Unnamed: 0' in data:
        data.set_index('Unnamed: 0', inplace=True)
    return data


def make_directory(path):
    folders = path.split('/')
    current_path = ''
    for folder in folders[:-1]:
        current_path += folder + '/'
        try:
            mkdir(current_path)
        except OSError:
            pass

    current_path += folders[-1]
    rmtree(current_path, ignore_errors=True)
    mkdir(current_path)


def can_go_down(img, x, y):
    if img[y, x] == 255 and img[y + 1, x] == 255:
        return True
    return False


def alignEdges(edges, attr, min_height=120, min_width=15, align_b=20):
    align_l = min_width // 2
    align_r = min_width - align_l
    align_t = min_height - align_b

    y1, x1, y2, x2 = edges

    height = y2 - y1
    width = x2 - x1

    if height < min_height:
        if attr['cy'] - y1 < align_t:
            y1 = max(0, attr['cy'] - align_t)
        if y2 - attr['cy'] < align_b:
            y2 = min(attr['height'], attr['cy'] + align_b)
    if width < min_width:
        if attr['cx'] - x1 < align_l:
            x1 = max(0, attr['cx'] - align_l)
        if x2 - attr['cx'] < align_r:
            x2 = min(attr['width'], attr['cx'] + align_r)

    return y1, x1, y2, x2


def getEdgeCoordinates(array_2d, attr, min_acceptable_height=20, max_acceptable_height=140, max_acceptable_width=20):
    y1, x1, y2, x2 = array_2d[:, 0].min(), array_2d[:, 1].min(), array_2d[:, 0].max() + 1, array_2d[:, 1].max() + 1

    height = y2 - y1
    width = x2 - x1

    if (min_acceptable_height <= height <= max_acceptable_height) and (width <= max_acceptable_width):
        y1, x1, y2, x2 = alignEdges([y1, x1, y2, x2], attr)
        return y1, x1, y2, x2
    else:
        return None, None, None, None


def prepareFits(headers_path, fits_path, headers_pattern="*.fits.hdr", fits_pattern="*.fits"):
    headers_folder = listdir(headers_path)
    fits_folder = listdir(fits_path)

    fits_headers = list([])
    headers_set = set({})
    files_set = set({})

    for entry in headers_folder:
        if fnmatch(entry, headers_pattern):
            fits_headers.append(f'{headers_path}/{entry}')

    fits_headers = sorted(fits_headers)

    headers_set.update(map(lambda x: x.split('/')[-1].split(headers_pattern[1:])[0], fits_headers))

    for entry in fits_folder:
        f_name = entry.split(fits_pattern[1:])[0]
        if fnmatch(entry, fits_pattern) and f_name in headers_set:
            files_set.add(f_name)

    fits_headers = np.array(fits_headers)

    return fits_headers, files_set


def getCoordinates(fits_headers, ra_dec):
    coordinates = np.ones((len(fits_headers), ra_dec.shape[0], 2)) * (-1)
    plates_containing_objects = set({})

    print('Calculating coordinates ...')
    for i in tqdm(range(len(fits_headers))):
        hdu_list = fits.open(fits_headers[i])
        w = wcs.WCS(hdu_list[0].header)
        xy = w.all_world2pix(ra_dec, 1, quiet=True)
        matching_indices = np.where((xy[:, 0] >= 0) & (xy[:, 0] <= 9601) & (xy[:, 1] >= 0) & (xy[:, 1] <= 9601))[0]
        coordinates[i][matching_indices] = xy[matching_indices]

        if len(matching_indices):
            plate = fits_headers[i].split('/')[-1].split('.')[0]
            plates_containing_objects.add(plate)
    return coordinates, plates_containing_objects


def getPlateCoordinates(xy, mx, my):
    return np.where((xy[:, 0] >= 0) & (xy[:, 0] < mx) & (xy[:, 1] >= 0) & (xy[:, 1] < my))[0]


def getFoundContoursEdges(contour) -> np.ndarray:
    min_height = 20
    min_width = 3
    max_height = 140
    max_width = 20

    x1, x2, y1, y2 = contour[:, 0, 0].min(), contour[:, 0, 0].max() + 1, \
        contour[:, 0, 1].min(), contour[:, 0, 1].max() + 1

    height = y2 - y1
    width = x2 - x1
    if (min_height <= height <= max_height) and (min_width <= width <= max_width) and height > width:
        return np.round(np.array([y1, x1, y2, x2], dtype='int'))
    else:
        return np.zeros(4)


def getContourEdges(src, dx, dy):
    height = src.shape[0]
    width = src.shape[1]

    top = 150
    bottom = 25
    left = 25
    right = 25

    top_bound = max(0, dy - top)
    bottom_bound = min(dy + bottom, height)
    left_bound = max(0, dx - left)
    right_bound = min(dx + right, width)

    region = src[
             top_bound:bottom_bound,
             left_bound:right_bound
             ].copy()

    attributes = {'cx': dx, 'cy': dy, 'height': height, 'width': width}

    bounds = pavlidis(region, min(dy, top), min(dx, left))
    contour_edges = getEdgeCoordinates((bounds + [top_bound, left_bound]).astype('int'), attributes)

    if all(contour_edges):
        # Theo-Pavlidis algorithm
        return contour_edges
    else:
        # OpenCV contour detection algorithm
        contours, _ = cv.findContours(region, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        biggest_contour = [None, None, None, None]
        max_height = 0

        if len(contours):
            for i, contour in enumerate(contours):
                y1, x1, y2, x2 = getFoundContoursEdges(contour)
                if all([y1, x1, y2, x2]):
                    h = y2 - y1
                    if (h > max_height) and (x1 <= min(dx, left) <= x2) and (y1 <= min(dy, top) <= y2):
                        biggest_contour = [y1, x1, y2, x2]
                        max_height = h

        if all(biggest_contour):
            biggest_contour = np.array(biggest_contour) + np.array([top_bound, left_bound, top_bound, left_bound])
            biggest_contour = alignEdges(biggest_contour, attributes)

        return biggest_contour
