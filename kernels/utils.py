import triton
import triton.language as tl

@triton.jit
def activation_forward(x, mode: tl.constexpr):
    # 0 => linear, 1 => sigmoid, 2 => tanh
    # This function returns f(x)
    if mode == 0:
        return x
    if mode == 1:
        return 1.0 / (1.0 + tl.exp(-x))  # sigmoid
    return x

@triton.jit
def activation_backward(y, dy, mode: tl.constexpr):
    # y is the "activated" output from activation_forward
    # dy is dL/dy, the gradient wrt the activated output
    # This function returns dL/dx, i.e. dy * f'(x)
    if mode == 0:
        return dy
    if mode == 1:
        return dy * y * (1.0 - y)       # sigmoid derivative
    return dy          # tanh derivative