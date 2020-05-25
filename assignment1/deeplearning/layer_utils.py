from deeplearning.layers import *
from deeplearning.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# Extra Credit
def affine_leaky_relu_forward(x, w, b, alpha):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - alpha: activation parameter

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, leaky_relu_cache = leaky_relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

# Extra Credit
def affine_leaky_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, leaky_relu_cache = cache
    da = leaky_relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# Extra Credit
def affine_elu_forward(x, w, b, alpha):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - alpha: activation parameter

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, elu_cache = elu_forward(a)
    cache = (fc_cache, elu_cache)
    return out, cache

# Extra Credit
def affine_elu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, elu_cache = cache
    da = elu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_params):
    """
    Convenience layer that performs an affine transform + batchnorm + ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: Parameters for batchnom layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a_1, fc_cache = affine_forward(x, w, b)
    a_2, bn_cache = batchnorm_forward(a_1, gamma, beta, bn_params)
    out, relu_cache = relu_forward(a_2)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for an affine transform + batchnorm + ReLU
    """
    fc_cache, bn_cache, relu_cache = cache
    da_2 = relu_backward(dout, relu_cache)
    da_1, dgamma, dbeta = batchnorm_backward(da_2, bn_cache)
    dx, dw, db = affine_backward(da_1, fc_cache)
    return dx, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

# Extra Credit! (added spatial batchnorm within helper functions)

def conv_spabn_relu_forward(x, w, b, conv_param, gamma, beta, bn_params):
    """
    A convenience layer that performs Conv + SpaBN + ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a_1, conv_cache = conv_forward_fast(x, w, b, conv_param)
    a_2, spabn_cache = spatial_batchnorm_forward(a_1, gamma, beta, bn_params)
    out, relu_cache = relu_forward(a_2)
    cache = (conv_cache, spabn_cache, relu_cache)
    return out, cache


def conv_spabn_relu_backward(dout, cache):
    """
    Backward pass for the Conv + SpaBN + ReLU convenience layer.
    """
    conv_cache, spabn_cache, relu_cache = cache
    da_2 = relu_backward(dout, relu_cache)
    da_1, dgamma, dbeta = spatial_batchnorm_backward(da_2, 
                                                     spabn_cache)
    dx, dw, db = conv_backward_fast(da_1, conv_cache)
    return dx, dw, db, dgamma, dbeta

def conv_spabn_relu_pool_forward(x, w, b, conv_param, gamma, beta, 
                                 bn_params, pool_param):
    """
    Convenience layer that performs Conv + SpaBN + ReLU + Maxpool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a_1, conv_cache = conv_forward_fast(x, w, b, conv_param)
    a_2, spabn_cache = spatial_batchnorm_forward(a_1, gamma, beta, bn_params)
    s, relu_cache = relu_forward(a_2)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, spabn_cache, relu_cache, pool_cache)
    return out, cache


def conv_spabn_relu_pool_backward(dout, cache):
    """
    Backward pass for the Conv + SpaBN + ReLU + Maxpool convenience layer
    """
    conv_cache, spabn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da_2 = relu_backward(ds, relu_cache)
    da_1, dgamma, dbeta = spatial_batchnorm_backward(da_2, 
                                                     spabn_cache)
    dx, dw, db = conv_backward_fast(da_1, conv_cache)
    return dx, dw, db, dgamma, dbeta