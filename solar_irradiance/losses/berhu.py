"""
The BerHu penalty and the grouped effect
https://arxiv.org/abs/1207.6868
"""
import torch.nn


class MeanAdaptiveBerHuLoss(torch.nn.Module):
    def __init__(self):
        super(MeanAdaptiveBerHuLoss, self).__init__()

    def forward(self, pred, target):
        """ Computes BerHu loss function

        l1_thresh = 0.2 * maximum difference in a batch

        L1 distance when the error is less than or equal to l1_thresh, otherwise (error^2 + l1_thresh^2) / 2*l1_thresh
        """
        assert pred.dim() == target.dim(), f"inconsistent dimensions, {pred.dim()} is not equal to {target.dim()}"

        error = torch.subtract(pred, target)
        abs_error = torch.abs(error)

        l1_thresh = torch.max(abs_error) * 0.2

        # Case 1: |x| <= l1_thresh
        case1_error = abs_error
        # Case 2: (x^2 + l1_thresh^2) / (2*l1_thresh)
        case2_error = (error**2 + l1_thresh**2) / (2 * l1_thresh)

        condition = torch.less_equal(abs_error, l1_thresh)
        loss = torch.where(condition, case1_error, case2_error)

        return loss.mean()
