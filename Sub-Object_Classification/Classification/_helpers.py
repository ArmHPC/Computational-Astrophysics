from os import mkdir
from shutil import rmtree

from torch import save as torch_save
import pandas as pd


def print_training_logs(arg1, arg2, arg3, arg4, arg5):
    epoch_message = 'Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
        arg1, arg2, arg3, arg4)
    epoch_message = "{:<40}".format(epoch_message)
    loss_message = 'Overall Loss: {:.6f}'.format(arg5)
    print(f'{epoch_message}{loss_message}')


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


def save_best_parameters(accuracy, best_acc, model_ckpt, optim_ckpt, epoch, model_dir):
    #####################################################
    if accuracy >= best_acc['value']:
        best_acc['value'] = accuracy
        best_acc['epoch'] = epoch
        best_acc['model_ckpt'] = model_ckpt
        best_acc['optim_ckpt'] = optim_ckpt

        torch_save(best_acc, f'{model_dir}/best_acc_{accuracy}_epoch_{epoch}.pth')
    #####################################################
