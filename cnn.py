from getdataset import GetDataSet
from neuralnetwork import NeuralNetwork
from layers import *
import numpy

class cnn:
    def run(self):
        getdataob = GetDataSet()

        X_train,Y_train,X_test,Y_test = getdataob.createDataSet(9)

        self.n_classes = len(numpy.unique(Y_train))

        # Create CNN Feature Extractor
        nn= NeuralNetwork(
            layers=[
                #CNN Feature Extractor
                Conv(
                    n_filters=4,
                    filter_shape=(3, 3),
                    weight_scale=0.1,
                    weight_decay = 0.01
                ),
                Activation('relu'),
                Pool(
                    pool_shape=(2, 2),
                    mode='max'
                ),
                Conv(
                    n_filters=8,
                    filter_shape=(3, 3),
                    weight_scale=0.1,
                    weight_decay=0.01
                ),
                Activation('relu'),
                Pool(
                    pool_shape=(2, 2),
                    mode='max'
                ),

                Flatten(), # Gives Feature Vectors

                # Classifier
                Linear(
                    n_out= self.n_classes,
                    weight_scale=0.1,
                    weight_decay=0.002
                ),
                LogRegression(),
            ],
            )
        # Initialize(Setup) The Layers of CNN
        nn._setup(X_train,Y_train)

        #Fit the Training Set to CNN to learn the task specific filters for Feature Extraction
        nn.fit(X_train,Y_train,learning_rate=0.01,max_iter=100,batch_size=40)

        print("\nModel Trained\n\n")

        print("\nTesting Prediction : \n")
        Y_pred = nn.predict(X_test)
        print("Actual  : \n",Y_test)
        print("Predicted : \n",Y_pred)
        error = nn._error(Y_pred,Y_test)
        print("Testing Error : ",error)
        file = open("weights.txt", "w")
        file.write("\nTesting Error : "+str(error))


ob = cnn()
ob.run()