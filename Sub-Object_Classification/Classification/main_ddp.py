import os
import argparse
import yaml

import load_data_ddp as load_data
import models
import train_ddp as train

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def setup(rank, ws):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=ws)


def cleanup():
    dist.destroy_process_group()


def main(rank, ws, config):
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

    # set up the process groups
    setup(rank, ws)

    train_data, train_classes, train_proportions = load_data.load_images(
         train_dir, train_batch_size, 'train', rank=rank, world_size=ws
    )

    val_data, val_classes, _ = load_data.load_images(
        val_dir, val_batch_size, 'val', rank=rank, world_size=ws
    ) if val_dir else (None, None, None)

    test_data, test_classes, _ = load_data.load_images(
        test_dir, test_batch_size, 'test', rank=rank, world_size=ws
    ) if test_dir else (None, None, None)

    print('\nTraining started:')

    net = models.Model(num_classes=num_classes, input_shape=input_shape).to(rank)
    print(net)

    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)

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
        cleanup=cleanup,
        dist=dist,
        rank=rank
    )

    torch.save(net.state_dict(), save_model_path)


if __name__ == "__main__":
    # suppose we have 3 gpus
    n_gpus = 1
    world_size = n_gpus

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config) as file:
        yaml_data = yaml.safe_load(file)

    mp.spawn(
        main,
        args=(world_size, yaml_data),
        nprocs=n_gpus
    )


# # in case we load a DDP model checkpoint to a non-DDP model
# model_dict = OrderedDict()
# pattern = re.compile('module.')
# for k,v in state_dict.items():
#     if re.search("module", k):
#         model_dict[re.sub(pattern, '', k)] = v
#     else:
#         model_dict = state_dict
# model.load_state_dict(model_dict)
