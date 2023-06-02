import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import time

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
    classes = kwargs['classes']
    # train_proportions = kwargs['train_proportions']

    num_epochs = kwargs['epochs']
    start_epoch = kwargs['start_epoch']
    device = kwargs['device']

    model_folder = kwargs['model_folder']

    evaluate = kwargs['evaluate']

    ## Optimizer
    learning_rate = 0.001
    rho = 0.95
    betas = (0.9, 0.999)

    # optimizer = optim.Adadelta(net.parameters(), lr=learning_rate, rho=rho)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=betas)

    ## Loss
    # class_weights = torch.sqrt(torch.tensor(train_proportions).reciprocal())
    # class_weights /= class_weights.mean()

    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = nn.CrossEntropyLoss()

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

            optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 2 == 0:
                print_training_logs(epoch, batch_idx * len(x), len(train_data.dataset),
                                    100. * batch_idx / len(train_data), loss.item())

        print(f'Epoch {epoch}/{num_epochs - 1}')

        if epoch % 5 == 0:
            with torch.no_grad():
                net.eval()
                # net.to(torch.device("cpu"))

                if train_data:
                    train_acc = evaluate(train_data, net, domain='train', classes=classes, device=device)
                    stats['train'] = train_acc
                if test_data:
                    test_acc = evaluate(test_data, net, domain='test', classes=classes, device=device)
                    stats['test'] = test_acc
                if val_data:
                    val_acc = evaluate(val_data, net, domain='val', classes=classes, device=torch.device("cpu"))
                    stats['val'] = val_acc

                ckpt_path = f'{model_folder}/{epoch}.pth'
                try:
                    torch.save(net.state_dict(), ckpt_path)
                except Exception as err:
                    print('Error:', err)

                net.train()
                net.to(device)

                writer.add_scalars(main_tag='Accuracy', tag_scalar_dict=stats, global_step=epoch)

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        end = time.time()
        print(f"Runtime for the epoch is {end - start}")
        print('-' * 62)

    print("Training is over.")
    writer.flush()
    writer.close()

    return net
