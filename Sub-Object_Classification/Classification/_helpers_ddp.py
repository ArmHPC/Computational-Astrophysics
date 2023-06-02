from os import mkdir, makedirs
from shutil import rmtree

import numpy as np
import pandas as pd
from PIL import Image


def print_training_logs(epoch, current_index, data_size, percentage, loss, device_id):
    epoch_message = 'Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
        epoch, current_index, data_size, percentage)
    epoch_message = "{:<40}".format(epoch_message)

    loss_message = 'Overall Loss: {:.6f}'.format(loss)
    loss_message = "{:<30}".format(loss_message)

    device_message = f'GPU ID: {device_id}'

    print(f'{epoch_message}{loss_message}{device_message}')


# def make_epoch_directories(path, epoch):
#     folders = ['test', 'val']
#     sub_folders = ['Glow', 'Res_j', 'Final', 'L', 'R', 'Sum']
#     for f in folders:
#         for sf in sub_folders:
#             try:
#                 makedirs(f'{path}/{f}/{epoch}/{sf}')
#             except FileExistsError:
#                 pass
#             except Exception as err:
#                 print(err)


# def loadImages(X):
#     images_list = []
#
#     for i, path in enumerate(X):
#         im = Image.open(path)
#         arr = np.array(im)
#
#         arr = (arr-arr.min())/(arr.max()-arr.min())
#
#         images_list.append(arr)
#
#     return np.array(images_list)


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
        except OSError as error:
            pass
    current_path += folders[-1]
    rmtree(current_path, ignore_errors=True)
    mkdir(current_path)
