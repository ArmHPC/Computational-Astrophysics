import torch.optim

import load_data
import models

from sklearn.metrics import classification_report

import os.path as os_path


# Testing Device
test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# test_device = torch.device("cpu")

input_shape = (160, 50)


def evaluate(dataloader, model, domain='test', classes=None, device=torch.device(test_device), best_acc=80):
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

    if domain == 'test' and accuracy >= best_acc:
        print(classification_report(y_gts, y_preds, zero_division=0, target_names=classes))
        # print(classes)
        # Output the classification results with their quantities for the specified class
        # aaa = torch.Tensor(y_preds)[torch.where(torch.Tensor(y_gts) == classes['C Ba'])]
        # print(aaa.unique(return_counts=True))

    return accuracy


def run_test(save_inference_ckpt=False, device=test_device):
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

    # num_classes = 5
    # checkpoint_path = './model/Final_Fine_Tune_10_5_Focal_loss_Augment_15/best_acc_90.07_epoch_173.pth'
    # data_root = '/media/sargis/Data/Stepan_Home/Datasets/Stepan_Datasets/DFBS_Combine_5_High'

    num_classes = 3
    checkpoint_path = './model/Final_Fine_Tune_10_3_Focal_loss_Augment_15_WH_25/best_acc_95.8_epoch_126.pth'
    data_root = '/media/sargis/Data/Stepan_Home/Datasets/Stepan_Datasets/DFBS_Combine_3_High'

    train_dir = f'{data_root}/train'
    test_dir = f'{data_root}/test'

    train_data, train_classes, _ = load_data.load_images(train_dir, 1, 'train', _drop_last=False)
    test_data, _, _ = load_data.load_images(test_dir, 1, 'test', _drop_last=False)

    net = models.Model(num_classes=num_classes, input_shape=input_shape).to(device)

    ckpt = torch.load(checkpoint_path)
    net.load_state_dict(ckpt['model_ckpt'])
    net.eval()

    if save_inference_ckpt:
        dir_name = os_path.dirname(checkpoint_path)
        file_name = os_path.basename(checkpoint_path)
        epoch, _ = os_path.splitext(file_name)
        ckpt_path = f'{dir_name}/{epoch}_Infer.pth'

        data = {
            "ckpt": net.state_dict(),
            "classes": train_classes
        }

        try:
            torch.save(data, ckpt_path)
        except Exception as err:
            print('Error:', err)

        print('Saved the checkpoint for inference.')
    else:
        net.eval()

        evaluate(dataloader=train_data, model=net, domain='train', device=device)
        evaluate(dataloader=test_data, model=net, device=device, classes=train_classes)


if __name__ == "__main__":
    save_ckpt = False
    run_test(save_inference_ckpt=save_ckpt)
