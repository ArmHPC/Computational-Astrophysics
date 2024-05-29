import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, num_classes=5, alpha: float = 0.25, gamma: float = 3, reduction: str = "mean"):
        # alpha: float = -1,
        # gamma: float = 2,
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor,):
        inputs = inputs.float()
        targets = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets = targets.float()

        p = torch.sigmoid(inputs)

        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)

        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


# def sigmoid_focal_loss(
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     num_classes=5,
#     # alpha: float = -1,
#     alpha: float = 0.25,
#     # gamma: float = 2,
#     gamma: float = 3,
#     reduction: str = "mean",
# ) -> torch.Tensor:
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         num_classes: Number of classes in training dataset.
#
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples. Default = -1 (no weighting).
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#         reduction: 'none' | 'mean' | 'sum'
#                  'none': No reduction will be applied to the output.
#                  'mean': The output will be averaged.
#                  'sum': The output will be summed.
#     Returns:
#         Loss tensor with the reduction option applied.
#     """
#     inputs = inputs.float()
#     targets = F.one_hot(targets.long(), num_classes=num_classes)
#     targets = targets.float()
#
#     p = torch.sigmoid(inputs)
#
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#
#     p_t = p * targets + (1 - p) * (1 - targets)
#
#     loss = ce_loss * ((1 - p_t) ** gamma)
#
#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss
#
#     if reduction == "mean":
#         loss = loss.mean()
#     elif reduction == "sum":
#         loss = loss.sum()
#
#     return loss
