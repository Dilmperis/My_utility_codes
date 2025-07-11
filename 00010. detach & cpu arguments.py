'''
1) .detach() removes the tensor from the computation graph.

It tells PyTorch:
    “I don't want to compute gradients for this tensor anymore.”
    Useful when you just want to evaluate, visualize, or log the outputs.
    If you don't call .detach(), and later try to compute gradients (even by accident), it may:
        -Hold onto extra memory (wasteful)
        -Cause unexpected errors in evaluation or logging.

===============================================================================================
2) If your model is running on GPU (cuda), the tensors are on the GPU too.

    But libraries like:
        a) torchmetrics
        b) matplotlib
        c) NumPy
        d) pandas
        e) Python built-in logging
        f) File I/O
        ... can't handle CUDA tensors.
So .cpu() moves the tensor to the CPU, making it usable in those libraries.
'''

