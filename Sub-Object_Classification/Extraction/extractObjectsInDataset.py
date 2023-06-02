# Imports
import numpy as np
import cv2 as cv

from astropy.io import fits
from astropy import wcs

from pavlidis import pavlidis
from _helpers import make_directory, prepareData

from fnmatch import fnmatch
from os import listdir, path as os_path
import argparse
from tqdm import tqdm
from time import perf_counter

import warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings(action='ignore', category=AstropyUserWarning)
warnings.filterwarnings(action='ignore', category=FITSFixedWarning)


# Function definitions
def can_go_down(img, x, y):
    if img[y, x] == 255 and img[y + 1, x] == 255:
        return True
    return False


def alignEdges(edges, attr):
    min_height = 120
    min_width = 15

    align_l = 7
    align_r = min_width - align_l
    align_b = 20
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


def getEdgeCoordinates(array_2d, attr):
    min_acceptable_height = 20
    max_acceptable_height = 140
    max_acceptable_width = 20

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
        return contour_edges
    else:
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


def main():
    ## blockSize: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7,
    # and so on.
    ## C: # Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may
    # be zero or negative as well.

    block_size = 15
    constant = 2
    calculate_coordinates = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='Subtypes')  # Combined
    parser.add_argument('--output_path')
    parser.add_argument('--class_column', default='Sp type')  # Cl
    parser.add_argument('--data_root', default='/home/stepan/Data/DFBS')
    parser.add_argument('--fits_path', default=None)
    parser.add_argument('--headers_path', default=None)

    args = parser.parse_args()

    data_root = args.data_root
    csv_name = args.input_csv
    output_path = args.output_path or f'{data_root}/Extracted/{csv_name}/{block_size}_{constant}'
    class_column = args.class_column
    fits_path = args.fits_path or os_path.join(data_root, 'fits_files')
    headers_path = args.headers_path or os_path.join(data_root, 'fits_headers')

    raw_folder = 'images'
    classified_folder = 'images_classified'

    # Process
    data = prepareData(
        path=f'{data_root}/Datasets/{csv_name}.csv')
    print(data.head(), '\n')

    make_directory(f'{output_path}/{raw_folder}')
    for class_name in np.unique(data[class_column]):
        make_directory(f'{output_path}/{classified_folder}/{class_name}')

    fits_headers, fits_set = prepareFits(
        headers_path=headers_path,
        fits_path=fits_path,
        headers_pattern="*.fits.hdr",
        fits_pattern="*.fits")
    
    if calculate_coordinates:
        coordinates, plates_containing_objects = getCoordinates(
            fits_headers=fits_headers,
            ra_dec=data[['_RAJ2000', '_DEJ2000']])
        np.save(f'{data_root}/Coordinates/{csv_name}', coordinates + 1)
    else:
        print('Loading coordinates ...')
        coordinates = np.load(f'{data_root}/Coordinates/{csv_name}.npy') - 1
        plates_containing_objects = fits_set

    print('Done.\n')

    datapoint_plates = dict({})
    all_datapoints = set({})  # Just for statistics

    incorrect_datapoints = dict({})
    short_images_extracted_count = 0
    short_20_images = set({})

    t = perf_counter()
    N = len(fits_headers)

    print('Extracting objects by plate ...')
    for i in tqdm(range(N)):
        it_t = perf_counter() - t
        t = perf_counter()
        print(f"{(i / N) * 100:.1f}% | {i}/{N} | {it_t:.2f} s/it | Average estimated time: "
              f"{(N-i)*it_t:.2f} seconds", end="\r")
        plate = fits_headers[i].split('/')[-1].split('.')[0]

        if plate in fits_set and plate in plates_containing_objects:
            fbs_plate = fits.open(f'{fits_path}/{plate}.fits')

            plate_img = np.array(fbs_plate[0].data, dtype=np.uint16)
            shape_y, shape_x = plate_img.shape
            del fbs_plate

            scaled_img = ((plate_img - plate_img.min()) / (plate_img.max() - plate_img.min()) * 255).astype(np.uint8)

            if np.mean(scaled_img) < 127.5:
                scaled_img = np.invert(scaled_img)

            gblur = cv.GaussianBlur(scaled_img, (3, 3), 2, 2)
            del scaled_img

            g_th = cv.adaptiveThreshold(gblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY_INV, blockSize=block_size, C=constant)
            del gblur
            plate_datapoints = getPlateCoordinates(coordinates[i], shape_x, shape_y)
            for pd_i in plate_datapoints:
                if pd_i not in all_datapoints:
                    all_datapoints.add(pd_i)
                if pd_i not in datapoint_plates:
                    dx, dy = np.round(coordinates[i, pd_i]).astype(int)
                    if g_th[dy, dx] == 255:
                        while (dy < shape_y - 1) and can_go_down(g_th, dx, dy):
                            dy += 1
    
                        try:
                            y1, x1, y2, x2 = getContourEdges(g_th, dx, dy)
                            if all([y1, x1, y2, x2]):
                                result = plate_img[y1:y2, x1:x2]
                                result_sized = cv.resize(result, (20, 140))
    
                                datapoint_plates[pd_i] = dict({  ##########################
                                    'plate': plate,
                                    'dx': dx,
                                    'dy': dy,
                                })
    
                                image_path = f'{output_path}/{raw_folder}/{pd_i}__{data.loc[pd_i, "Name"]}.tiff'
                                classes_path = f'{output_path}/{classified_folder}/{data.loc[pd_i, class_column]}' \
                                               f'/{pd_i}__{data.loc[pd_i, "Name"]}.tiff'
    
                                data.loc[pd_i, 'dx'] = dx
                                data.loc[pd_i, 'dy'] = dy
                                data.loc[pd_i, 'plate'] = plate
                                data.loc[pd_i, 'path'] = image_path
    
                                cv.imwrite(image_path, result)
                                cv.imwrite(classes_path, result)
    
                                incorrect_datapoints.pop(pd_i, None)
                            else:
                                if pd_i not in short_20_images:
                                    short_20_images.add(pd_i)
                        except AssertionError as err:
                            print(err, '20', pd_i)
                        except Exception as err2:
                            print(err2, '20-2', pd_i)
                    else:
                        extracted = False
                        for i_x in range(max(0, dx - 2), min(shape_x, dx + 3)):
                            if extracted:
                                break
                            for i_y in range(dy, max(-1, dy - 3), -1):
                                if extracted:
                                    break
                                if i_x == dx and i_y == dy:
                                    continue
                                if g_th[i_y, i_x] == 255:
                                    y = int(i_y)
                                    while (y < shape_y - 1) and can_go_down(g_th, i_x, y):
                                        y += 1
                                    try:
                                        y1, x1, y2, x2 = getContourEdges(g_th, i_x, y)
                                        if all([y1, x1, y2, x2]):
                                            result = plate_img[y1:y2, x1:x2]
                                            result_sized = cv.resize(result, (20, 140))
    
                                            datapoint_plates[pd_i] = dict({  ##########################
                                                'plate': plate,
                                                'dx': i_x,
                                                'dy': y,
                                            })
    
                                            image_path = f'{output_path}/{raw_folder}/{pd_i}__{data.iloc[pd_i]["Name"]}.tiff'
                                            classes_path = f'{output_path}/{classified_folder}/{data.iloc[pd_i][class_column]}' \
                                                           f'/{pd_i}__{data.loc[pd_i, "Name"]}.tiff'
    
                                            data.loc[pd_i, 'dx'] = i_x
                                            data.loc[pd_i, 'dy'] = y
                                            data.loc[pd_i, 'plate'] = plate
                                            data.loc[pd_i, 'path'] = image_path
    
                                            cv.imwrite(image_path, result)
                                            cv.imwrite(classes_path, result)
    
                                            incorrect_datapoints.pop(pd_i, None)
    
                                            extracted = True
                                            short_images_extracted_count += 1
                                    except AssertionError as err:
                                        print("Internal error", err, '30', pd_i)
                                    except Exception as err2:
                                        print("Internal error2", err2, '30-2', pd_i)
                        if not extracted:
                            if pd_i not in incorrect_datapoints:
                                incorrect_datapoints[pd_i] = [plate]
                            else:
                                incorrect_datapoints[pd_i].append(plate)
                else:
                    continue
    
    data = data[data['plate'].notna()]
    data.to_csv(f'data/Datasets/{csv_name}_{block_size}_{constant}_extracted.csv')

    print()
    print('all_datapoints:', len(all_datapoints))
    print('incorrect_datapoints:', len(incorrect_datapoints))
    print('short_20_images:', len(short_20_images.difference(incorrect_datapoints, datapoint_plates)))
    print('extracted images count:',
          len(all_datapoints)
          - len(short_20_images.difference(incorrect_datapoints, datapoint_plates))
          - len(incorrect_datapoints))
    print('short_images_extracted_count:', short_images_extracted_count)


if __name__ == '__main__':
    main()
