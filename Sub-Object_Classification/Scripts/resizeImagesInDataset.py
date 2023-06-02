import argparse
from _helpers import make_directory

import cv2
import numpy as np
from pandas import read_csv


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv_path')
parser.add_argument('--output_path')

args = parser.parse_args()
input_csv_path = args.input_csv_path or 'data/Datasets/Combined_extracted.csv'
output_path = args.output_path or 'data/Extracted/Combined/images_resized/'

make_directory(output_path)

data = read_csv(input_csv_path, index_col='Unnamed: 0')

images_count = 0
max_acceptable_width = 20
max_acceptable_height = 140

for row, value in data.iterrows():
    path = value['path']

    im = cv2.imread(path, -1)

    s = im.shape

    arr = im.astype(np.float32)
    arr_min = arr.min()
    arr_max = arr.max()

    arr = (arr - arr_min) / (arr_max - arr_min)

    if arr.mean() > 0.5:
        arr = 1 - arr

    if s[0] > max_acceptable_height or s[1] > max_acceptable_width:
        continue

    d_width = (max_acceptable_width - s[1])
    d_height = (max_acceptable_height - s[0])

    d_top = int(d_height / 2)
    d_bottom = int(d_height - d_top)

    d_left = int(d_width / 2)
    d_right = int(d_width - d_left)

    for l in range(d_left):
        arr = np.insert(arr, 0, 0, axis=1)

    for r in range(d_right):
        b = np.zeros((s[0], 1))
        arr = np.append(arr, b, axis=1)

    for t in range(d_top):
        arr = np.insert(arr, 0, 0, axis=0)

    for b in range(d_bottom):
        b = np.zeros((1, arr.shape[1],))
        arr = np.append(arr, b, axis=0)

    arr = arr * (arr_max - arr_min) + arr_min

    cv2.imwrite(output_path + f'{row}__{value["Name"]}.tiff', arr.astype(np.uint16))

    images_count += 1

print(f'All done. Out of {data.shape[0]} images {images_count} were resized')

exit(0)
