from layers import *
import numpy
import time

class NeuralNetwork:
    def __init__(self,layers):
        self.layers = layers
        self.rng = numpy.random.RandomState()

    def _setup(self, X, Y):

        # Setup layers sequentially
        next_shape = X.shape
        for layer in self.layers:
            layer.setup(next_shape, self.rng)
            next_shape = layer.output_shape(next_shape)

    def fit(self, X, Y, learning_rate=0.1, max_iter=10, batch_size=40):

        self.file = open("weights.txt","w")

        old_error = 1.1

        n_samples = Y.shape[0]
        n_batches = n_samples // batch_size

        n_class =  len(numpy.unique(Y))
        Y_vector=self._vectorize(Y,n_class)


        start_time = time.time()

        iter = 0
        while iter < max_iter :
            start_iter_time = time.time()
            print("\nIteration : ",iter)
            iter += 1
            for b in range(n_batches):
                print("Batch : ",b)
                batch_begin = b * batch_size
                batch_end = batch_begin + batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = Y[batch_begin:batch_end]
                Y_batch_vector = self._vectorize(Y_batch,n_class)


                # Forward Propagation
                X_next = X_batch
                for layer in self.layers:
                    X_next = layer.forward_propogation(X_next)
                Y_pred = X_next # after Log Regression


                # Back propagation of partial derivatives
                next_grad = self.layers[-1].input_grad(Y_batch_vector, Y_pred)
                for layer in reversed(self.layers[:-1]):# Except the last layer
                    next_grad = layer.backward_propogation(next_grad)


                # Update parameters
                for i,layer in enumerate(self.layers):
                    if isinstance(layer, Conv) or isinstance(layer,Linear):
                        W,b = self.layers[i].params()
                        dW,db = self.layers[i].param_incs()
                        W = W - learning_rate*dW
                        b = b - learning_rate*db
                        layer.W = W
                        layer.b = b

            end_iter_time = time.time()
            print("Iteration time : ",(end_iter_time-start_iter_time)/60,"\n")
            print("Training Prediction : ")
            Y_pred = self.predict(X)
            print(Y_pred)
            error = self._error(Y_pred,Y)
            print("Training Error : ",error)
            self._save_intermediateweights(iter,Y_pred,error)
            self._savestate(iter)
            end_time = time.time()
            print("Training Time : ", (end_time - start_time)/60,"\n")

        self.file.close()


    def _vectorize(self,Y,n_class=40):
        # convert number to bit vector representation(one hot encoding)
        vector = numpy.zeros((Y.shape[0],n_class))

        for i,y in enumerate(Y):
            class_y = int(y[0])
            vector[i,class_y]=1

        return vector

    def _loss(self, X, Y_vector):
        print("Loss : \n")
        X_next = X
        for layer in self.layers:
            X_next = layer.forward_propogation(X_next)
        Y_pred = X_next
        return self.layers[-1].loss(Y_vector, Y_pred)


    def _error(self, Y_pred, Y):
        """ Calculate error on the given data. """
        error=numpy.zeros(len(Y_pred))
        for i in range(len(Y_pred)):
            if(Y_pred[i]!=Y[i]):
                error[i]=1
        return numpy.mean(error)


    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for layer in self.layers:
            X_next = layer.forward_propogation(X_next)
        Y_pred = X_next
        Y_pred = numpy.argmax(X_next,axis=-1)
        return Y_pred

    def _savestate(self,iter):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv) or isinstance(layer, Linear):
                W, b = self.layers[i].params()
                numpy.save("layer"+str(i)+"W_"+str(iter),W)
                numpy.save("layer"+str(i)+"b_"+str(iter),b)

    def _loadstate(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv) or isinstance(layer, Linear):
                W = numpy.load("layer"+str(i)+"W.npy")
                b = numpy.load("layer"+str(i)+"b.npy")
                layer.set_params(W,b)


    def _save_intermediateweights(self,iter,Y,error):
        self.file.write("Iteration : "+str(iter)+"\n")
        self.file.write(str(Y))
        self.file.write("\nError :\n")
        self.file.write(str(error))
        self.file.write("\nWeights : \n")
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv) or isinstance(layer, Linear):
                W, b = self.layers[i].params()
                self.file.write("Layer : "+str(i)+"\n")
                self.file.write(str(W))
                self.file.write("\n")
                self.file.write(str(b))
                self.file.write("\n")
        self.file.write("\n")
        self.file.flush()

