import argparse
from tqdm import tqdm
import torch.optim
import numpy as np
from collections import Counter

import load_data
import models


# Testing Device
test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# test_device = torch.device("cpu")

input_shape = (160, 50)


def infer_evaluate(
        dataloader, model,
        device=torch.device(test_device),
        threshold=None,
        classes=None,
        save_dir=None,
        load_saved=False
):
    if load_saved:
        y_indices = np.load(f'{save_dir}/indices.npy')
        y_preds = np.load(f'{save_dir}/preds.npy')

        print('Predictions were loaded from previously saved file...\n')

    else:
        y_preds = []
        y_indices = []

        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tqdm(dataloader)):
                x = x.to(device)

                # calculate outputs by running images through the network
                outputs = model(x)

                predicted, indices = torch.max(torch.softmax(outputs.data, dim=1), dim=1)
                y_preds.extend(predicted.cpu().numpy())
                y_indices.extend(indices.cpu().numpy())

        y_indices = np.array(y_indices)
        y_preds = np.array(y_preds)

        np.save(f'{save_dir}/indices', np.array(y_indices))
        np.save(f'{save_dir}/preds', np.array(y_preds))

    if threshold:
        classes['Other'] = -1
        for i, prob in enumerate(y_preds):
            if prob < threshold:
                y_indices[i] = -1

    label_predictions = [list(classes.keys())[list(classes.values()).index(num)] for num in y_indices]
    class_counts = dict(Counter(label_predictions))

    for label in classes.keys():
        if label not in class_counts:
            class_counts[label] = 0

    return class_counts


def infer(device=test_device):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_root',
        default='/media/sargis/Data/Stepan_Home/Datasets/Stepan_Datasets/DFBS/Inference_data/4m/data'
    )
    parser.add_argument(
        '--ckpt_path',
        default='./model/Final_Fine_Tune_10_3_Focal_loss_Augment_15_WH_25/best_acc_95.8_epoch_126_Infer.pth'
    )
    parser.add_argument('--num_classes', default='3', choices=('3', '5', '6', '10'))
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--threshold', default=None)
    parser.add_argument('--load_saved', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    infer_dir = args.data_root
    checkpoint_path = args.ckpt_path
    num_classes = int(args.num_classes)
    batch_size = args.batch_size
    threshold = float(args.threshold) if args.threshold else None
    load_saved = args.load_saved or False

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

    infer_data = load_data.load_images(
        path=infer_dir, batch_size=batch_size, domain='inference', _drop_last=False, _shuffle=True
    )

    net = models.Model(num_classes=num_classes, input_shape=input_shape).to(device)

    ckpt_data = torch.load(checkpoint_path)
    test_classes = ckpt_data['classes']
    net.load_state_dict(ckpt_data['ckpt'])
    net.eval()

    print('Evaluation started:')
    predictions = infer_evaluate(
        dataloader=infer_data,
        model=net,
        device=device,
        classes=test_classes,
        threshold=threshold,
        save_dir=f'{infer_dir}/..',
        load_saved=load_saved
    )
    print('Threshold:', threshold)
    print(predictions)


if __name__ == "__main__":
    infer()
