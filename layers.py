import numpy
from convolution import Convolution
from pooling import Pooling

class Layer:

    def setup(self, input_shape, rng):
        pass

    def fprop(self, input):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, output_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

    def output_shape(self, input_shape):
        """ Calculate shape of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError()

class Conv(Layer):

    def __init__(self, n_filters, filter_shape,weight_scale,weight_decay):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.weight_scale = weight_scale
        self.weight_decay = weight_decay
        self.conv_ob = Convolution()

    def setup(self, input_shape, rng):

        n_featuremaps = input_shape[1]
        W_shape = (n_featuremaps, self.n_filters) + self.filter_shape # this appends the filter shape eg : (16,4,3,3)
        self.W = rng.normal( size=W_shape, scale=self.weight_scale)  # generate random filters with gaussian(normal) distribution
        self.b = numpy.zeros(self.n_filters) # initialize bias weights equal to zero

    def params(self):
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def set_params(self,W,b):
        self.W = W
        self.b = b

    def output_shape(self, input_shape):
        # Zero padding is considered
        height = input_shape[2] #- self.filter_shape[0] + 1
        width = input_shape[3] #- self.filter_shape[1] + 1
        shape = (input_shape[0],
                 self.n_filters,
                 height,
                 width)
        return shape

    def forward_propogation(self, input): # forward propogation
        print("Forward  Propogation : Convolution")
        self.last_input = input # activation this layer
        self.last_input_shape = input.shape # shape of activation
        conv_shape = self.output_shape(input.shape)
        convout = self.conv_ob.convolve(input,self.W,conv_shape)
        z=self.W
        return convout + self.b[numpy.newaxis, :, numpy.newaxis, numpy.newaxis]


    def backward_propogation(self, output_grad): # backward propogation of gradients
        print("Backward Propogation : Convolution")
        input_grad = numpy.empty(self.last_input_shape)

        self.dW = numpy.zeros(self.W.shape)

        # Convolve
        input_grad, self.dW = self.conv_ob.convolve_backprop(self.last_input, output_grad, self.W, self.dW)
        n_imgs = output_grad.shape[0]
        self.db = numpy.sum(output_grad, axis=(0, 2, 3)) / (n_imgs)
        self.dW =self.dW -  self.weight_decay * self.W

        return input_grad




class Pool(Layer):
    def __init__(self, pool_shape, mode='max'):
        self.mode = mode
        self.pool_h, self.pool_w = pool_shape
        self.pool_ob = Pooling()

    def forward_propogation(self, input):
        print("Forward  Propogation : Pooling")
        self.last_input_shape = input.shape # save for backprop

        self.last_maxpositions = numpy.empty(self.output_shape(input.shape)+(2,), dtype=numpy.int)  # positions which hold the maximum elements

        poolout,self.last_maxpositions = self.pool_ob.pool(input,self.pool_h,self.pool_w,self.last_maxpositions,self.mode)
        return poolout

    def backward_propogation(self, output_grad):
        print("Backward Propogation : Pooling")
        input_grad = numpy.empty(self.last_input_shape)

        input_grad  = self.pool_ob.pool_backprop(output_grad,self.last_input_shape,self.last_maxpositions)

        return input_grad

    def output_shape(self, input_shape):
        shape = (input_shape[0],
                 input_shape[1],
                 input_shape[2]//self.pool_h,
                 input_shape[3]//self.pool_w)
        return shape





class Flatten(Layer):
    def forward_propogation(self, input):
        print("Forward  Propogation : Flatten")
        self.last_input_shape = input.shape
        return numpy.reshape(input, (input.shape[0], -1))

    def backward_propogation(self, output_grad):
        print("Backward Propogation : Flatten")
        return numpy.reshape(output_grad, self.last_input_shape)

    def output_shape(self, input_shape):
        return (input_shape[0], numpy.prod(input_shape[1:]))





class Activation(Layer):
    def __init__(self, type):

        if type == 'sigmoid':
            self.fun = self.sigmoid
            self.fun_d = self.sigmoid_d
        elif type == 'relu':
            self.fun = self.relu
            self.fun_d = self.relu_d
        elif type == 'tanh':
            self.fun = self.tanh
            self.fun_d = self.tanh_d
        else:
            raise ValueError('Invalid activation function.')

    def forward_propogation(self, input):
        print("Forward  Propogation : ReLU")
        self.last_input = input
        return self.fun(input)

    def backward_propogation(self, output_grad):
        print("Backward Propogation : ReLU")
        a=output_grad
        b=output_grad*self.fun_d(self.last_input)
        x=self.last_input
        return output_grad*self.fun_d(self.last_input)

    def output_shape(self, input_shape):
        return input_shape

    def sigmoid(self,x):
        return 1.0 / (1.0 + numpy.exp(-x))

    def sigmoid_d(self,x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self,x):
        return numpy.tanh(x)

    def tanh_d(self,x):
        e = numpy.exp(2 * x)
        return (e - 1) / (e + 1)

    # def relu(self, x):
    #     z = numpy.log(1+numpy.exp(x))#numpy.maximum(0.0, x)
    #     return z
    #
    # def relu_d(self, x):
    #     dx = 1/(1+numpy.exp(-x))#numpy.zeros(x.shape)
    #     #dx[x >= 0] = 1
    #     return dx




    def relu(self,x):
        z=numpy.maximum(0.0, x)
        return z

    def relu_d(self,x):
        dx = numpy.zeros(x.shape)
        dx[x >= 0] = 1
        return dx



class Linear(Layer):
    def __init__(self, n_out, weight_scale, weight_decay=0.0):
        self.n_out = n_out
        self.weight_scale = weight_scale
        self.weight_decay = weight_decay

    def setup(self, input_shape, rng):
        n_input = input_shape[1]
        W_shape = (n_input, self.n_out)
        self.W = rng.normal(size=W_shape, scale=self.weight_scale)
        self.b = numpy.zeros(self.n_out)

    def forward_propogation(self, input):
        print("Forward  Propogation : Linear")
        self.last_input = input
        res = numpy.dot(input, self.W) + self.b
        return numpy.dot(input, self.W) + self.b

    def backward_propogation(self, output_grad):
        print("Backward Propogation : Linear")
        n = output_grad.shape[0]
        self.dW = numpy.dot(self.last_input.T, output_grad)/n - self.weight_decay*self.W
        self.db = numpy.mean(output_grad, axis=0)
        return numpy.dot(output_grad, self.W.T)

    def params(self):
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def set_params(self, W, b):
        self.W = W
        self.b = b

    def output_shape(self, input_shape):
        return (input_shape[0], self.n_out)



class LogRegression(Layer):
    """ Softmax layer with cross-entropy loss function. """

    def forward_propogation(self, input):
        print("Forward  Propogation : Log. Reg.")
        e = numpy.exp(input - numpy.amax(input, axis=1, keepdims=True))
        return e / numpy.sum(e, axis=1, keepdims=True)

    def backward_propogation(self, output_grad):
        print("Backward Propogation : Log. Reg.")
        raise NotImplementedError(
            'LogRegression does not support back-propagation of gradients. '
            + 'It should occur only as the last layer of a NeuralNetwork.'
        )

    def input_grad(self, Y, Y_pred):
        # Assumes one-hot encoding.
        return -(Y - Y_pred)

    def loss(self, Y, Y_pred):
        # Assumes one-hot encoding.
        eps = 1e-15
        Y_pred = numpy.clip(Y_pred, eps, 1 - eps)
        Y_pred /= Y_pred.sum(axis=1, keepdims=True)
        loss = -numpy.sum(Y * numpy.log(Y_pred))
        return loss / Y.shape[0]

    def output_shape(self, input_shape):
        return input_shape
