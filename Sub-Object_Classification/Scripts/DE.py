### TO BE DELETED
from time import perf_counter
WATCHLIST = [2480, 4497, 6329, 726, 4933, 3626, 5359, 5373, 6001]
OPT_LIST = [5084, 6251, 6276, 6289, 5992]
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


######################### Imports
import pandas as pd
from os import listdir
from fnmatch import fnmatch
import numpy as np
from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt
import cv2 as cv
import sys
sys.path.insert(1, 'TPA/pavlidis/build/lib.win-amd64-3.9')
from pavlidis import pavlidis


######################### Function definitions
def insideCoordinate(img, x, y):
    if img[y,x-1] == 255 and img[y,x+1] == 255 and img[y-1,x] == 255 and img[y+1,x] == 255:
        return True
    return False

def getEdgeCoordinates(array_2d, cx, cy):
    min_vert = 71
    min_hor = 10
    align_l = 5
    align_r = 6
    vertical_diff = 0
    y1, x1, y2, x2 = array_2d[:,0].min(), array_2d[:,1].min(), array_2d[:,0].max()+1, array_2d[:,1].max()+1

    if y2 - y1 < min_vert:
        vertical_diff = min_vert - (y2 - y1)
        # print("Top alignment:", )
        y1 = max(0, y2 - min_vert)
    if x2 - x1 < min_hor:
        if cx - x1 < align_l:
            # print("Left alignment:", align_l - (cx - x1))
            x1 = max(0, cx - align_l)
        if x2 - cx < align_r:
            # print("Right alignment:", align_r - (x2 - cx))
            x2 = min(9600, cx + align_r)
    return y1, x1, y2, x2, vertical_diff

def prepareData(path):
    data = pd.read_csv(path)
    data.drop(0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data["plate"] = np.nan
    data["path"] = np.nan
    data["dx"] = np.zeros(data.shape[0])
    data["dy"] = np.zeros(data.shape[0])
    data[['_RAJ2000', '_DEJ2000']] = data[['_RAJ2000', '_DEJ2000']].astype(float)

    # print(data.head())
    return data

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

    # print(fits_headers[:5])
    # print('Files in headers folder:', len(headers_folder))
    # print('Headers in headers folder:', len(fits_headers))
    # print()
    # print(fits_files[:5])
    # print('Files in fits folder:', len(fits_folder))
    # print('Fits files in fits folder:', len(fits_files))

    fits_headers = np.array(fits_headers)
    fits_files = np.array(fits_files)
    fits_set = set(map(lambda x: x.split('/')[-1].split('.')[0], fits_files))
    
    return fits_headers, fits_files, fits_set

def getCoordinates(fits_headers, data):
    coordinates = np.ones((len(fits_headers), data.shape[0], 2)) * (-1)

    for i in range(len(fits_headers)):
        hdulist = fits.open(fits_headers[i])
        w = wcs.WCS(hdulist[0].header)

        xy = w.all_world2pix(data[['_RAJ2000', '_DEJ2000']], 1, quiet=True)

        matching_indices = np.where((xy[:,0] >= 0) & (xy[:,0] <= 9601) & (xy[:,1] >= 0) & (xy[:,1] <= 9601))[0]
        
        coordinates[i][matching_indices] = xy[matching_indices]
    
    return coordinates

######################### Proces
data = prepareData(
    path='../data/DFBS.csv')

fits_headers, fits_files, fits_set = prepareFits(
    headers_path='../data/fits_headers',
    fits_path='../data/fits_files',
    headers_pattern="*.hdr",
    fits_pattern="*.fits")

coordinates = getCoordinates(
    fits_headers=fits_headers,
    data=data)
np.save('data/coordinates.csv', coordinates+1)
# coordinates = np.load('data/coordinates.csv.npy') - 1


datapoint_plates = dict({})
all_datapoints = set({})

incorrect_coordinate_count = 0

for i in range(len(fits_headers)):
    plate = fits_headers[i].split('/')[-1].split('.')[0]
    if plate in fits_set:
        fbs_plate = fits.open('./data/fits_files/' + plate + '.fits')

        plate_img = fbs_plate[0].data
        del fbs_plate
        
        scaled_img = ((plate_img/plate_img.max())*255).astype(np.uint8)
        del plate_img

        if np.mean(scaled_img) < 127.5:
            scaled_img = np.invert(scaled_img)

        gblur = cv.GaussianBlur(scaled_img, (3, 3), 2, 2)
        # mblur = cv.medianBlur(scaled_img, 3)

        # del scaled_img #########################################################################

        g_th = cv.adaptiveThreshold(gblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY_INV,21,2)
        # m_th = cv.adaptiveThreshold(mblur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #             cv.THRESH_BINARY_INV,11,2)

        g_th_m = cv.adaptiveThreshold(gblur, 255, cv.ADAPTIVE_THRESH_MEAN_C,\
                    cv.THRESH_BINARY_INV,21,2)
        # m_th_m = cv.adaptiveThreshold(mblur, 255, cv.ADAPTIVE_THRESH_MEAN_C,\
        #             cv.THRESH_BINARY_INV,11,2)

        
        del gblur #########################################################################
        # del mblur #########################################################################

        plate_datapoints = np.where(coordinates[i,:,0] >= 0)[0]
        for pd_i in plate_datapoints:
            if pd_i not in all_datapoints: ###############################################################################
                all_datapoints.add(pd_i)
            if pd_i not in datapoint_plates:
                dx, dy = np.round(coordinates[i, pd_i]).astype(int)
                if g_th[dy,dx] == 255:
                    while insideCoordinate(g_th, dx, dy):
                        dy += 1

                    try:
                        pavl_res = pavlidis(g_th, dy, dx)
                        y1, x1, y2, x2, vd, hd = getEdgeCoordinates(pavl_res, dx, dy)
                        if y2 - y1 - vd > 20 and (y2 - y1 - vd)/(x2 - x1 - hd):
                            # print(pd_i)
                            # fig = plt.figure()
                            # plt.gray()
                            # ax1 = fig.add_subplot(221)
                            # ax2 = fig.add_subplot(222)
                            # ax3 = fig.add_subplot(223)
                            # ax4 = fig.add_subplot(224)
                            # ax1.imshow(scaled_img[y1:y2,x1:x2])
                            # ax2.imshow(scaled_img[dy-100:dy+16,dx-15:dx+16])
                            # ax3.imshow(g_th[y1:y2,x1:x2])
                            # ax4.imshow(g_th_m[y1:y2,x1:x2])
                            # plt.show()
                            
                            result = scaled_img[y1:y2,x1:x2]
                            result_sized = cv.resize(scaled_img[y1:y2,x1:x2], (20, 140))
                            
                            datapoint_plates[pd_i] = dict({ ##########################
                                'plate': plate,
                                'dx': dx,
                                'dy': dy,
                            })
                            
                            image_path = f'data/images/{pd_i}__{data.loc[pd_i, "Name"]}.tiff'
                            image_2_path = f'data/images_2/{pd_i}__{data.loc[pd_i, "Name"]}.tiff'

                            data.loc[pd_i, 'dx'] = dx
                            data.loc[pd_i, 'dy'] = dy
                            data.loc[pd_i, 'plate'] = plate
                            data.loc[pd_i, 'path'] = image_path

                            cv.imwrite(image_path, result)
                            cv.imwrite(image_2_path, result_sized)

                            # img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                            # fig = plt.figure()
                            # plt.gray()
                            # ax1 = fig.add_subplot(121)
                            # ax2 = fig.add_subplot(122)
                            # ax1.imshow(result)
                            # ax2.imshow(img)
                            # plt.show()
                    except AssertionError as err:
                        print(err)
                    except Exception as err2:
                        print(err2)
                        
                else:
                    # fig = plt.figure()
                    # plt.gray()
                    # ax1 = fig.add_subplot(131)
                    # ax2 = fig.add_subplot(132)
                    # ax3 = fig.add_subplot(133)
                    # p = np.copy(g_th[dy-30:dy+10,dx-7:dx+8])
                    # if g_th[dy,dx-1] == 0:
                    #     g_th[dy,dx-1] = 50
                    # else:
                    #     g_th[dy,dx-1] = 200
                    # if g_th[dy,dx+1] == 0:
                    #     g_th[dy,dx+1] = 50
                    # else:
                    #     g_th[dy,dx+1] = 200
                    # if g_th[dy-1,dx] == 0:
                    #     g_th[dy-1,dx] = 50
                    # else:
                    #     g_th[dy-1,dx] = 200
                    # if g_th[dy+1,dx] == 0:
                    #     g_th[dy+1,dx] = 50
                    # else:
                    #     g_th[dy+1,dx] = 200
                    print(pd_i)
                    # ax1.imshow(g_th[dy-30:dy+10,dx-7:dx+8])
                    # ax2.imshow(p)
                    # ax3.imshow(scaled_img[dy-50:dy+10,dx-7:dx+8])
                    # plt.show()
                    incorrect_coordinate_count += 1
            else:
                continue


data = data[data['plate'].notna()]
data.to_csv('data/DFBS_extracted.csv')
print(len(all_datapoints))
print(incorrect_coordinate_count)
