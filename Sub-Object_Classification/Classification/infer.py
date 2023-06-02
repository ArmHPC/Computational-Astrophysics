import argparse
import torch.optim

import load_data
import models

from sklearn.metrics import classification_report


# Testing Device
test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# test_device = torch.device("cpu")

input_shape = (160, 50)


def evaluate(dataloader, model, domain='test', classes=None, device=torch.device(test_device)):
    correct = 0
    total = 0

    with torch.no_grad():
        y_preds = []
        y_gts = []

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            # calculate outputs by running images through the network
            outputs = model(x)

            # _, predicted = torch.max(outputs.data, dim=1)
            predicted = torch.argmax(outputs.data, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            y_preds.extend(predicted.cpu().numpy())
            y_gts.extend(y.cpu().numpy())

    accuracy = round(100 * correct / total, ndigits=2)
    print(f'Accuracy of the network on the {domain} images: {accuracy} %')

    if domain == 'test':
        print(classification_report(y_gts, y_preds, zero_division=0, target_names=classes))
        print(classes)
        # Output the classification results with their quantities for the specified class
        aaa = torch.Tensor(y_preds)[torch.where(torch.Tensor(y_gts) == classes['C Ba'])]
        print(aaa.unique(return_counts=True))

    if domain == 'inference':
        return predicted
    return accuracy


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

    evaluate(dataloader=infer_data, model=net, domain='inference', device=device)


if __name__ == "__main__":
    infer()
