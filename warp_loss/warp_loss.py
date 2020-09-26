import torch
from torch.autograd import Variable, Function


def to_one_hot(tensor: torch.Tensor, n_dims=None) -> torch.Tensor:
    """
    Take long `tensor` with `n_dims` and
    convert it to one-hot representation with n + 1 dimensions.
    """
    tensor = tensor.long().view(-1, 1)
    n_dims = n_dims if n_dims else int(torch.max(tensor)) + 1
    one_hot_tensor = torch.zeros(tensor.size(0), n_dims).scatter_(1, tensor, 1)
    return one_hot_tensor.view(tensor.size(0), -1)


class WARPAutograd(Function):
    """
    Autograd function of WARP loss for both Binary and MultiLabel.
    Based on these papers:
    * WSABIE: Scaling Up To Large Vocabulary Image Annotation
    * Deep Convolutional Ranking for Multilabel Image Annotation
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, target: torch.Tensor, max_num_trials: int = None, alpha: int = 0
    ) -> float:
        batch_size = target.size(0)
        label_size = target.size(1)
        # Set max number of trials
        max_num_trials = max_num_trials or label_size - 1
        # Rank Weights
        rank_weights = torch.Tensor([1.0 / (i + 1) for i in range(label_size)])
        rank_weights = torch.cumsum(rank_weights, dim=0)
        # Transform rank into loss with weightening function (L)
        L = torch.zeros_like(target)
        # Positive labels and negative labels
        positive_indices = target.long()
        negative_indices = target.eq(0).long()
        for i in range(batch_size):
            # Get indices of positives and negatives
            pos_target = positive_indices[i].nonzero().view(-1)
            neg_target = negative_indices[i].nonzero().view(-1)
            # Continue if everything is positive or no positives
            if len(pos_target) == 0 or len(pos_target) == label_size:
                continue
            # Uniform weights for sampling negative
            weights = torch.Tensor([1 / neg_target.size(0)]).repeat(neg_target.size(0))
            # Randomly pick negatives with multinomial
            samples = torch.gather(
                neg_target,
                dim=0,
                index=torch.multinomial(weights, max_num_trials, replacement=True),
            ).long()
            # Pick score for each random sample
            samples_scores = torch.gather(input[i].view(-1), dim=0, index=samples)
            # Sample score margin of positive target with each sample
            sample_score_margin = (alpha + samples_scores - input[i, pos_target].view(-1, 1)).gt(0).long()
            # Pick first nonzero
            idx = torch.arange(sample_score_margin.size(-1), 0, -1)
            sample_score_margin_idx = sample_score_margin * idx
            rejection = torch.argmax(sample_score_margin_idx, dim=-1)
            # Select rank for sample (+ 1 because we start from 0)
            label_rank = rank_weights[(label_size - 1) / (rejection + 1)]
            # Check for zero in sample_score_margin
            # if we didn't manage to sample negative with greater rank
            maybe_zero = sample_score_margin[torch.arange(pos_target.size(0)), rejection]
            label_rank[maybe_zero.eq(0)] = 0
            # Update L
            L[i, pos_target] = label_rank
        # Calculate loss
        loss = 0
        for i in range(batch_size):
            pos = positive_indices[i].nonzero().view(-1)
            neg = negative_indices[i].nonzero().view(-1)
            loss += (
                L[i, pos].view(-1, 1) * torch.relu(alpha + input[i, neg] - input[i, pos].view(-1, 1))
            ).sum()
        # Variables for backward
        ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.alpha = alpha
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Gather saved for backward variables
        input, target = ctx.saved_variables
        L = Variable(ctx.L, requires_grad=False)
        alpha = ctx.alpha
        positive_indices = ctx.positive_indices
        negative_indices = ctx.negative_indices
        # Construct grad_input
        grad_input = torch.zeros_like(input)
        for i in range(grad_input.size(0)):
            pos_ind = positive_indices[i].nonzero().view(-1)
            neg_ind = negative_indices[i].nonzero().view(-1)
            # Continue if everything is positive or no positives
            if len(pos_ind) == 0 or len(neg_ind) == 0:
                continue
            # Generate one hot from positive and negative
            one_hot_pos = to_one_hot(pos_ind, grad_input.size(1))
            one_hot_neg = to_one_hot(neg_ind, grad_input.size(1))
            # Samples to zero-out because of ReLU
            samples_to_zero = alpha + input[i, neg_ind] - input[i, pos_ind].view(-1, 1)
            # Proposed gradient
            proposed_gradient = L[i, pos_ind].view(-1, 1, 1) * (one_hot_neg - one_hot_pos.unsqueeze(1))
            # Zero gradients based on samples_to_zero
            proposed_gradient[~samples_to_zero.gt(0)] = 0
            grad_input[i] = proposed_gradient.sum((0, 1))
        grad_input = grad_output * Variable(grad_input)
        return grad_input, None, None, None


class WARPLoss(torch.nn.Module):
    """
    WARP Loss implementation for Binary and MultiLabel classification.
    Based on these papers:
    * WSABIE: Scaling Up To Large Vocabulary Image Annotation
    * Deep Convolutional Ranking for Multilabel Image Annotation

    WARPLoss inside passes input and target to CPU to calculate loss and gradients.

    Parameters
    ----------
    alpha : `int`, optional (default = `0`)
        Margin component for Hinge in WARP.
    max_num_trials : `int`, optional (default = `None`)
        Max number of trials to sample a violating label.
        If None set max_num_trials equal to label size.
    """

    def __init__(self, alpha: int = 0, max_num_trials: int = None) -> None:
        super().__init__()
        self._alpha = alpha
        self._max_num_trials = max_num_trials

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return WARPAutograd.apply(input.cpu(), target.cpu(), self._max_num_trials, self._alpha)
