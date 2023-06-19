import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Callable


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean"
) -> Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    # if torch.jit.is_scripting():
    #     raise NotImplementedError(
    #         "F.triplet_margin_with_distance_loss does not support JIT scripting: "
    #         "functions requiring Callables cannot be scripted."
    #     )

    # if has_torch_function_variadic(anchor, positive, negative):
    #     return handle_torch_function(
    #         triplet_margin_with_distance_loss,
    #         (anchor, positive, negative),
    #         anchor,
    #         positive,
    #         negative,
    #         distance_function=distance_function,
    #         margin=margin,
    #         swap=swap,
    #         reduction=reduction,
    #     )

    # Check validity of reduction mode
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

    # Check dimensions
    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    if not (a_dim == p_dim and p_dim == n_dim):
        raise RuntimeError(
            (f"The anchor, positive, and negative tensors are expected to have "
             f"the same number of dimensions, but got: anchor {a_dim}D, "
             f"positive {p_dim}D, and negative {n_dim}D inputs"))

    # Calculate loss
    if distance_function is None:
        distance_function = torch.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    dis_pos_neg = distance_function(positive, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.  If True, and if the positive example is closer to the
    # negative example than the anchor is, swaps the positive example and the
    # anchor in the loss computation.

    # if swap:
    #     dist_swap = distance_function(positive, negative)
    #     dist_neg = torch.minimum(dist_neg, dist_swap)
    loss = torch.clamp_min(margin + dist_pos - dist_neg - dis_pos_neg, 0)

    # Apply reduction
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss
    



def arcface_triplet_loss(anc, pos, neg, a_label, p_label, n_label, distance_func = None):
    if distance_func is None:
        distance_func = lambda x1,x2: 1 - F.cosine_similarity(x1,x2)

    criterion_arc = torch.nn.CrossEntropyLoss()
    criterion_triplet = triplet_margin_with_distance_loss
    loss = (criterion_arc(anc,a_label) + criterion_arc(pos,p_label) + criterion_arc(neg,n_label))/3 + criterion_triplet(anc, pos, neg, distance_function= distance_func, margin = 1)
    return loss