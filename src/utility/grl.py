# src/utility/grl.py

from torch.autograd import Function

"""
Gradient Reversal Layer implementation.

For use in domain adversarial training.

During the forward pass, it acts as an identity function.
During the backward pass, it multiplies the gradient by -alpha.
"""


class _GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass of the GRL

        Inputs:
            x: input tensor
            alpha: scaling factor for the gradient reversal
        """
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the GRL

        Reverses the gradient by multiplying with -alpha.

        Inputs:
            grad_output: gradient of the loss w.r.t. the output
        """
        # multiply the incoming gradient by -alpha
        return -ctx.alpha * grad_output, None
    
def grad_reverse(x, alpha: float = 1.0):
    """
    Apply Gradient Reversal Layer to input tensor x with scaling factor alpha.

    Inputs:
        x: input tensor
        alpha: scaling factor for the gradient reversal
    """
    return _GRL.apply(x, alpha)