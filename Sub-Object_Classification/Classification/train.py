import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import time

from models import Model
from test import evaluate
from _helpers import print_training_logs, save_best_parameters
from losses import FocalLoss


def train_model(net: Model, **kwargs):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_id = kwargs['train_id']
    writer = SummaryWriter(comment=f'/{train_id}')
    stats = dict({})

    net.train()

    train_data = kwargs['train']
    val_data = kwargs['val']
    test_data = kwargs['test']

    train_classes = kwargs['train_classes']

    print('\nClasses:', train_classes.keys())
    # train_proportions = kwargs['train_proportions']
    train_data_size = len(train_data.dataset)
    train_batch_size = train_data.batch_size
    train_batches_count = len(train_data)

    num_epochs = kwargs['epochs']
    start_epoch = kwargs['start_epoch']
    device = kwargs['device']

    checkpoints_dir = kwargs['checkpoints_dir']
    model_dir = kwargs['model_dir']

    ## Optimizer
    LR = kwargs['learning_rate']
    # weight_decay = kwargs['weight_decay']
    # grad_clip_norm = kwargs['grad_clip_norm']
    # rho = 0.95
    betas = (0.9, 0.999)
    min_ckpt_acc = kwargs['min_ckpt_acc']

    # optimizer = optim.Adadelta(net.parameters(), lr=LR, rho=rho)
    # optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=LR, betas=betas)
    print('Optimizer: Adam')

    ## Learning Rate Scheduler
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    ## Loss
    loss = kwargs['loss']

    if loss['name'] == 'Focal':
        print('Loss:', loss['name'])
        num_classes = len(train_classes)
        criterion = FocalLoss(num_classes=num_classes, alpha=loss['alpha'], gamma=loss['gamma'], reduction="mean")

    else:
        # class_weights = torch.sqrt(torch.tensor(train_proportions).reciprocal())
        # class_weights /= class_weights.mean()

        # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        criterion = nn.CrossEntropyLoss()

    best_acc = {'epoch': 0, 'value': min_ckpt_acc, 'model_ckpt': None, 'optim_ckpt': None}

    print('\nTraining started:')

    for epoch in range(start_epoch, num_epochs):
        if epoch % 5 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.9
        # elif epoch > 100 and epoch % 10 == 0:
        #     for g in optimizer.param_groups:
        #         g['lr'] = g['lr'] * 0.95

        net.train()
        net.to(device)

        start = time.time()
        epoch_losses = []

        for batch_idx, (x, y) in enumerate(train_data):
            x = x.to(device)
            y = y.to(device)

            # Optimizer zero setting
            optimizer.zero_grad()

            # Feed-Forward
            output = net(x)

            loss = criterion(output, y)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 8 == 0:
                print_training_logs(epoch, batch_idx * train_batch_size, train_data_size,
                                    100. * batch_idx / train_batches_count, loss.item())

        print(f'Epoch {epoch}/{num_epochs - 1}')

        if epoch % 1 == 0:
            with torch.no_grad():
                net.eval()
                # net.to(torch.device("cpu"))

                if train_data:
                    train_acc = evaluate(
                        train_data, net, domain='train', classes=train_classes, device=device
                    )
                    stats['train'] = train_acc
                if test_data:
                    test_acc = evaluate(
                        test_data, net, domain='test', device=device,
                        classes=train_classes, best_acc=best_acc['value']
                    )
                    stats['test'] = test_acc
                if val_data:
                    val_acc = evaluate(
                        val_data, net, domain='val', classes=train_classes, device=torch.device("cpu")
                    )
                    stats['val'] = val_acc

                save_best_parameters(
                    accuracy=test_acc,
                    best_acc=best_acc,
                    model_ckpt=net.state_dict(),
                    optim_ckpt=optimizer.state_dict(),
                    epoch=epoch,
                    model_dir=model_dir
                )

                if epoch % 10 == 0:
                    ckpt_path = f'{checkpoints_dir}/{epoch}.pth'
                    try:
                        torch.save(net.state_dict(), ckpt_path)
                    except Exception as err:
                        print('Error:', err)

                net.train()
                net.to(device)

                writer.add_scalars(main_tag='Accuracy', tag_scalar_dict=stats, global_step=epoch)

        epoch_loss = sum(epoch_losses) / (len(epoch_losses) + 1e-5)
        writer.add_scalars(main_tag='Loss', tag_scalar_dict={'train': epoch_loss}, global_step=epoch)
        # writer.add_scalar("Loss/train", epoch_loss, epoch)

        end = time.time()
        print(f"Runtime for the epoch is {end - start}")
        print('-' * 62)

    print("Training is over.")
    writer.flush()
    writer.close()

    return net
