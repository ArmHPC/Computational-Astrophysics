import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import time

from test import evaluate
from _helpers import print_training_logs


def train_model(net, **kwargs):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_id = kwargs['train_id']
    writer = SummaryWriter(comment=train_id)
    stats = dict({})

    net.train()

    train_data = kwargs['train']
    val_data = kwargs['val']
    test_data = kwargs['test']

    train_classes = kwargs['train_classes']
    test_classes = kwargs['test_classes']

    # train_proportions = kwargs['train_proportions']
    train_data_size = len(train_data.dataset)
    train_batch_size = train_data.batch_size
    train_batches_count = len(train_data)

    num_epochs = kwargs['epochs']
    start_epoch = kwargs['start_epoch']
    device = kwargs['device']

    model_folder = kwargs['model_folder']

    ## Optimizer
    LR = kwargs['learning_rate']
    LR_fine_tune = 0
    weight_decay = kwargs['weight_decay']
    grad_clip_norm = kwargs['grad_clip_norm']
    # rho = 0.95

    params_train = kwargs['params_train']
    params_fine_tune = kwargs['params_fine_tune']

    # optimizer = optim.Adadelta(net.parameters(), lr=LR, rho=rho)
    optimizer = optim.Adam(params_train, lr=LR, weight_decay=weight_decay)
    optimizer_fine_tune = optim.SGD(params_fine_tune, lr=LR_fine_tune, weight_decay=weight_decay)

    ## Learning Rate Scheduler
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    torch.optim.lr_scheduler.StepLR(optimizer_fine_tune, step_size=5, gamma=0.9)

    ## Loss
    # class_weights = torch.sqrt(torch.tensor(train_proportions).reciprocal())
    # class_weights /= class_weights.mean()

    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, num_epochs):
        net.train()
        net.to(device)

        start = time.time()
        epoch_losses = []

        for batch_idx, (x, y) in enumerate(train_data):
            x = x.to(device)
            y = y.to(device)

            # Optimizer zero setting
            optimizer.zero_grad()
            optimizer_fine_tune.zero_grad()

            # Feed-Forward
            output = net(x)

            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm(net.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer_fine_tune.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 8 == 0:
                print_training_logs(epoch, batch_idx * train_batch_size, train_data_size,
                                    100. * batch_idx / train_batches_count, loss.item())

        print(f'Epoch {epoch}/{num_epochs - 1}')

        if epoch % 5 == 0:
            with torch.no_grad():
                net.eval()
                # net.to(torch.device("cpu"))

                if train_data:
                    train_acc = evaluate(
                        train_data, net, domain='train', classes=(train_classes, test_classes), device=device
                    )
                    stats['train'] = train_acc
                if test_data:
                    test_acc = evaluate(
                        test_data, net, domain='test', classes=(train_classes, test_classes), device=device
                    )
                    stats['test'] = test_acc
                if val_data:
                    val_acc = evaluate(
                        val_data, net, domain='val', classes=(train_classes, test_classes), device=torch.device("cpu")
                    )
                    stats['val'] = val_acc

                ckpt_path = f'{model_folder}/{epoch}.pth'
                try:
                    torch.save(net.state_dict(), ckpt_path)
                except Exception as err:
                    print('Error:', err)

                net.train()
                net.to(device)

                writer.add_scalars(main_tag='Accuracy', tag_scalar_dict=stats, global_step=epoch)

        epoch_loss = sum(epoch_losses) / (len(epoch_losses) + 1e-5)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        end = time.time()
        print(f"Runtime for the epoch is {end - start}")
        print('-' * 62)

    print("Training is over.")
    writer.flush()
    writer.close()

    return net
