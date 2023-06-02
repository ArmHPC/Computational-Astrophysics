import argparse
from os import listdir, path as os_path
from _helpers import make_directory

import cv2
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_path')
parser.add_argument('--output_path')

args = parser.parse_args()
input_path = args.input_path or 'data/extractedResizedContours/'
output_path = args.output_path or 'data/extractedResizedScaledContours/'

images_count = 0

images = listdir(input_path)
make_directory(output_path)

for im_name in tqdm(images):
    im_path = os_path.join(input_path, im_name)
    im = cv2.imread(im_path, -1)

    arr = im.astype(np.float32)
    arr_min = arr.min()
    arr_max = arr.max()

    arr = 255 * ((arr - arr_min) / (arr_max - arr_min))

    cv2.imwrite(output_path + im_name, arr.astype(np.uint8))

    images_count += 1

print(f'All done. Out of {len(images)} images {images_count} were scaled')

exit(0)
