import os
import argparse
import yaml

import load_data_hvd as load_data
import models
import train_hvd as train

import torch
import horovod.torch as hvd


def main(config):
    # Training Device
    if config['training']['use_gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    # Datasets
    train_dir = config['data']['train']['path']
    val_dir = config['data']['val']['path']
    test_dir = config['data']['test']['path']

    # Batch sizes
    train_batch_size = config['training']['batch_size']['train']
    val_batch_size = config['training']['batch_size']['val']
    test_batch_size = config['training']['batch_size']['test']

    # Network parameters
    num_epochs = config['training']['epochs']
    # num_classes = config['data']['output']['num_classes']
    num_classes = len(os.listdir(train_dir))
    input_shape = tuple(config['data']['input']['shape'])

    # Project
    root_dir = config['project']['root']
    train_id = config['project']['train_id']

    # Models
    model_dir = f'{root_dir}/model/{train_id}'
    checkpoints_dir = f'{root_dir}/Checkpoint/{train_id}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # In case if you want to continue your training from a certain checkpoint
    start_epoch = config['training']['start_epoch']
    load_model_path = config['model']['load_path']
    save_model_path = f"{model_dir}/final.pth"

    # start_epoch = 6
    # load_model_path = f'{checkpoints_dir}/5.pth'

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    train_data, train_classes, train_proportions = load_data.load_images(train_dir, train_batch_size, 'train', hvd)
    val_data, val_classes, _ = load_data.load_images(
        val_dir, val_batch_size, 'val', hvd) if val_dir else (None, None, None)
    test_data, test_classes, _ = load_data.load_images(
        test_dir, test_batch_size, 'test', hvd) if test_dir else (None, None, None)

    print('\nTraining started:')

    net = models.Model(num_classes=num_classes, input_shape=input_shape).to(device)
    print(net)

    if load_model_path:
        net.load_state_dict(torch.load(load_model_path))

    net = train.train_model(
        net,
        train=train_data,
        val=val_data,
        test=test_data,
        epochs=num_epochs,
        start_epoch=start_epoch,
        device=device,
        model_folder=checkpoints_dir,
        train_id=train_id,
        classes=test_classes,
        train_proportions=train_proportions,
        hvd=hvd
    )

    torch.save(net.state_dict(), save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config) as file:
        yaml_data = yaml.safe_load(file)

    main(yaml_data)
