import argparse
import torch.optim

import load_data
import models

from sklearn.metrics import classification_report


# Testing Device
test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# test_device = torch.device("cpu")

input_shape = (160, 50)


def infer_evaluate(dataloader, model, classes=None, device=torch.device(test_device), return_predictions=True):
    with torch.no_grad():
        y_preds = []

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            # calculate outputs by running images through the network
            outputs = model(x)

            if return_predictions:
                predicted = torch.argmax(outputs.data, dim=1)
            else:
                predicted = outputs.data

            y_preds.extend(predicted.cpu().numpy())

    return y_preds


def infer(device=test_device):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/Inference/Subtypes/19_2/images')
    parser.add_argument('--ckpt_path',
                        default='Classification_PyTorch/Checkpoint/Dense_6_High_Focal_25_3_Final/136.pth')
    parser.add_argument('--num_classes', default=10, choices=(5, 6, 10))
    parser.add_argument('--batch_size', default=128)

    # parser.add_argument('--class_column', default='Sp type', choices=('Sp type', 'Cl'))
    # parser.add_argument('--output_path')
    # parser.add_argument('--fits_path', default=None)
    # parser.add_argument('--headers_path', default=None)

    args = parser.parse_args()

    infer_dir = args.data_root
    checkpoint_path = args.ckpt_path
    num_classes = args.num_classes
    batch_size = args.batch_size

    # checkpoint_path = './Checkpoint/ResNet/485_64_63.pth'
    # checkpoint_path = './Checkpoint/ViT_10/225_70_68.5.pth'

    # num_classes = 10
    # checkpoint_path = './Checkpoint/Orig_10/100.pth'
    # data_root = '/home/sargis/Datasets/Stepan/DFBS_Combine'

    # num_classes = 6
    # checkpoint_path = './Checkpoint/Orig_6/105.pth'
    # data_root = '/home/sargis/Datasets/Stepan/DFBS_Combine_6'

    # num_classes = 6
    # checkpoint_path = './Checkpoint/Orig_6_High/105.pth'
    # data_root = '/home/sargis/Datasets/Stepan/DFBS_Combine_6_High'

    infer_data = load_data.load_images(path=infer_dir, batch_size=batch_size, domain='inference', _drop_last=False)

    net = models.Model(num_classes=num_classes, input_shape=input_shape).to(device)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()

    infer_evaluate(dataloader=infer_data, model=net, device=device)


if __name__ == "__main__":
    infer()
