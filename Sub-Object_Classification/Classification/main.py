import os
import argparse
import yaml

import load_data
import models
import train

import torch
import torch.nn as nn


def main(config):
    fine_tune = config['training']['fine_tune'] or False

    # Training Device
    if config['training']['use_gpu']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
    else:
        device = torch.device('cpu')

    print('Training device:', device)

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
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    grad_clip_norm = config['training']['grad_clip_norm']

    train_data, train_classes, train_proportions = load_data.load_images(train_dir, train_batch_size, 'train')
    val_data, val_classes, _ = load_data.load_images(
        val_dir, val_batch_size, 'val', _drop_last=False) if val_dir else (None, None, None)
    test_data, test_classes, _ = load_data.load_images(
        test_dir, test_batch_size, 'test', _drop_last=False) if test_dir else (None, None, None)

    params_train = []
    params_fine_tune = []

    if fine_tune:
        net = models.Model(num_classes=19, input_shape=input_shape, arch='default')

        if load_model_path:
            net.load_state_dict(torch.load(load_model_path))

        net.classifier.fc2 = nn.Linear(
            in_features=net.classifier.fc2.in_features,
            out_features=num_classes
        )

        for name, param in net.named_parameters():
            if 'classifier.fc' in name or 'classifier.conv4' in name or 'classifier.bn4' in name:
                params_fine_tune.append(param)
            else:
                params_train.append(param)

    else:
        net = models.Model(num_classes=num_classes, input_shape=input_shape, arch='default')

        if load_model_path:
            net.load_state_dict(torch.load(load_model_path))

    net.to(device)
    print(net)
    print('\nTraining started:')

    net = train.train_model(
        net,
        train=train_data,
        val=val_data,
        test=test_data,
        epochs=num_epochs,
        start_epoch=start_epoch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        params_train=params_train,
        params_fine_tune=params_fine_tune,
        device=device,
        model_folder=checkpoints_dir,
        train_id=train_id,
        train_classes=train_classes,
        test_classes=test_classes,
        train_proportions=train_proportions
    )

    torch.save(net.state_dict(), save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config) as file:
        yaml_data = yaml.safe_load(file)

    main(yaml_data)
