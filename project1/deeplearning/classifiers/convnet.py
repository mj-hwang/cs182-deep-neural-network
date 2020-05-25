import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class MyConvNet(object):
    """
    A multi-layer convolutional network with the following architecture:

    (conv-relu-conv-relu-max pool) * 2 - (affine-relu) * 3 - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 32, 32, 32], 
                 filter_size=3, hidden_dim=[64, 128, 256], num_classes=10, 
                 weight_scale=1e-3, reg=0.0, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        F1, F2, F3, F4 = num_filters
        H1, H2, H3 = hidden_dim

        # convolutional layers
        self.params["W1"] = np.random.normal(scale=weight_scale, 
                                             size=(F1, C, 
                                                   filter_size, filter_size))
        self.params["b1"] = np.zeros(F1)
        self.params["gamma1"] = np.ones(F1)
        self.params["beta1"] = np.zeros(F1)

        self.params["W2"] = np.random.normal(scale=weight_scale, 
                                             size=(F2, F1, 
                                                   filter_size, filter_size))
        self.params["b2"] = np.zeros(F2)
        self.params["gamma2"] = np.ones(F2)
        self.params["beta2"] = np.zeros(F2)

        self.params["W3"] = np.random.normal(scale=weight_scale, 
                                             size=(F3, F2, 
                                                   filter_size, filter_size))
        self.params["b3"] = np.zeros(F3)
        self.params["gamma3"] = np.ones(F3)
        self.params["beta3"] = np.zeros(F3)

        self.params["W4"] = np.random.normal(scale=weight_scale, 
                                             size=(F4, F3, 
                                                   filter_size, filter_size))
        self.params["b4"] = np.zeros(F4)
        self.params["gamma4"] = np.ones(F4)
        self.params["beta4"] = np.zeros(F4)

        # affine layers
        self.params["W5"] = np.random.normal(scale=weight_scale, 
                                             size=((F4 * 64, H1)))
        self.params["b5"] = np.zeros(H1)
        self.params["gamma5"] = np.ones(H1)
        self.params["beta5"] = np.zeros(H1)

        self.params["W6"] = np.random.normal(scale=weight_scale, 
                                             size=(H1, H2))
        self.params["b6"] = np.zeros(H2)
        self.params["gamma6"] = np.ones(H2)
        self.params["beta6"] = np.zeros(H2)

        self.params["W7"] = np.random.normal(scale=weight_scale, 
                                             size=(H2, H3))
        self.params["b7"] = np.zeros(H3)
        self.params["gamma7"] = np.ones(H3)
        self.params["beta7"] = np.zeros(H3)

        self.params["W8"] = np.random.normal(scale=weight_scale, 
                                             size=(H3, num_classes))
        self.params["b8"] = np.zeros(num_classes)

        self.bn_param1 = {'mode': 'train'}
        self.bn_param1 = {'mode': 'train'}
        self.bn_param2 = {'mode': 'train'}
        self.bn_param3 = {'mode': 'train'}
        self.bn_param4 = {'mode': 'train'}
        self.bn_param5 = {'mode': 'train'}
        self.bn_param6 = {'mode': 'train'}
        self.bn_param7 = {'mode': 'train'}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # if y is None:
        #     self.bn_param1['mode'] = 'test'
        #     self.bn_param2['mode'] = 'test' 
        #     self.bn_param3['mode'] = 'test'
        #     self.bn_param4['mode'] = 'test'
        #     self.bn_param5['mode'] = 'test'
        #     self.bn_param6['mode'] = 'test'
        #     self.bn_param7['mode'] = 'test'

        W1, b1 = self.params['W1'], self.params['b1']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        W2, b2 = self.params['W2'], self.params['b2']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        W3, b3 = self.params['W3'], self.params['b3']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        W4, b4 = self.params['W4'], self.params['b4']
        gamma4, beta4 = self.params['gamma4'], self.params['beta4']
        W5, b5 = self.params['W5'], self.params['b5']
        gamma5, beta5 = self.params['gamma5'], self.params['beta5']
        W6, b6 = self.params['W6'], self.params['b6']
        gamma6, beta6 = self.params['gamma6'], self.params['beta6']
        W7, b7 = self.params['W7'], self.params['b7']
        gamma7, beta7 = self.params['gamma7'], self.params['beta7']
        W8, b8 = self.params['W8'], self.params['b8']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################        
        out1, cache1 = conv_spabn_relu_forward(X, W1, b1, conv_param, 
                                               gamma1, beta1, self.bn_param1)
        out2, cache2 = conv_spabn_relu_pool_forward(out1, W2, b2, conv_param, 
                                                    gamma2, beta2,
                                                    self.bn_param2, pool_param)
        out3, cache3 = conv_spabn_relu_forward(out2, W3, b3, conv_param, 
                                               gamma3, beta3, self.bn_param3)
        out4, cache4 = conv_spabn_relu_pool_forward(out3, W4, b4, conv_param,
                                                    gamma4, beta4,
                                                    self.bn_param4, pool_param)
        out5, cache5 = affine_bn_relu_forward(out4, W5, b5, gamma5, beta5, self.bn_param5)
        out6, cache6 = affine_bn_relu_forward(out5, W6, b6, gamma6, beta6, self.bn_param6)
        out7, cache7 = affine_bn_relu_forward(out6, W7, b7, gamma7, beta7, self.bn_param7)

        out8, cache8 = affine_forward(out7, W8, b8)
        scores = out8
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y) 

        dout, dW8, db8 = affine_backward(dout, cache8)
        dout, dW7, db7, grads["gamma7"], grads["beta7"] = \
        affine_bn_relu_backward(dout, cache7)
        dout, dW6, db6, grads["gamma6"], grads["beta6"] = \
        affine_bn_relu_backward(dout, cache6)
        dout, dW5, db5, grads["gamma5"], grads["beta5"] = \
        affine_bn_relu_backward(dout, cache5)

        dout, dW4, db4, grads["gamma4"], grads["beta4"] = \
        conv_spabn_relu_pool_backward(dout, cache4)
        dout, dW3, db3, grads["gamma3"], grads["beta3"] = \
        conv_spabn_relu_backward(dout, cache3)

        dout, dW2, db2, grads["gamma2"], grads["beta2"] = \
        conv_spabn_relu_pool_backward(dout, cache2)
        dout, dW1, db1, grads["gamma1"], grads["beta1"] = \
        conv_spabn_relu_backward(dout, cache1)

        loss += self.reg * 0.5 * \
                (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2) + \
                 np.sum(W5**2) + np.sum(W6**2) + np.sum(W7**2) + np.sum(W8**2))

        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        dW5 += self.reg * W5
        dW6 += self.reg * W6
        dW7 += self.reg * W7
        dW8 += self.reg * W8

        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W2"] = dW2
        grads["b2"] = db2
        grads["W3"] = dW3
        grads["b3"] = db3
        grads["W4"] = dW4
        grads["b4"] = db4
        grads["W5"] = dW5
        grads["b5"] = db5
        grads["W6"] = dW6
        grads["b6"] = db6
        grads["W7"] = dW7
        grads["b7"] = db7
        grads["W8"] = dW8
        grads["b8"] = db8
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
