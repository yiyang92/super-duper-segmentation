import torch

EPSILON = 1e-5


def intesection_over_union(
    gt_masks: torch.Tensor, pred_masks: torch.Tensor
) -> torch.Tensor:
    """Computer mean IOU over classes."""
    assert gt_masks.shape == pred_masks.shape

    num_classes = len(torch.unique(gt_masks))
    # Confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        for j in range(num_classes):
            mask_i = gt_masks == i
            mask_j = pred_masks == j
            confusion_matrix[i, j] = torch.sum(mask_i * mask_j)
    # IOUs
    iou_scores = torch.zeros(num_classes)
    for i in range(num_classes):
        # Compute the IoU score for class i
        true_positives = confusion_matrix[i, i]
        false_positives = torch.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = torch.sum(confusion_matrix[i, :]) - true_positives
        iou_scores[i] = true_positives / (
            true_positives + false_positives + false_negatives + EPSILON
        )

    return torch.mean(iou_scores)
