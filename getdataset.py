import numpy
import re

class GetDataSet:

    HEIGHT = 112
    WIDTH = 92
    N_SUBJECTS = 40
    N_POSES = 10

    def read_pgm(self,filename, byteorder='>'):
        """
        Return image data from a raw PGM file as numpy array.
        Format specification: http://netpbm.sourceforge.net/doc/pgm.html
        """
        with open(filename, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"  # Magic Number
                b"(\d+)\s(?:\s*#.*[\r\n])*"  # Width
                b"(\d+)\s(?:\s*#.*[\r\n])*"  # Height
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)"  # Maximum Gray Value
                , buffer).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)

        return numpy.frombuffer(buffer,
                                dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                                count=int(width) * int(height),
                                offset=len(header)
                                ).reshape((int(height), int(width)))


    def selector_generator(self, num_trainposes): # Selector for the poses present in training and testing sets
        poses = list(range(self.N_POSES)) # pose indexes i.e [0 1 2 3 .. 8 9]
        numpy.random.shuffle(poses) # Randomly arrange the pose indexes

        train_selector= poses[0:num_trainposes]
        test_selector = poses[num_trainposes:]

        return train_selector,test_selector

    def createDataSet(self,num_trainposes):
        # Path of dataset
        address="att_faces\\s"

        train_selector, test_selector = self.selector_generator(num_trainposes)

        # initially our image is of size 112 X 92 pixels
        trainset = numpy.ones((self.N_SUBJECTS*len(train_selector),1,self.HEIGHT,self.WIDTH)) #Image_Number X Channel X Image_Height X Image_Width
        testset = numpy.ones((self.N_SUBJECTS*len(test_selector),1,self.HEIGHT,self.WIDTH))

        train_labels = numpy.ones((self.N_SUBJECTS*len(train_selector),1))
        test_labels = numpy.ones((self.N_SUBJECTS*len(test_selector),1))

        train_counter = 0
        test_counter = 0
        # Generate Train and Test Sets
        for subject in range(1,self.N_SUBJECTS+1):

            for pose in train_selector:
                image = self.read_pgm(address+str(subject)+"\\"+str(pose+1)+".pgm",byteorder='<')
                trainset[train_counter,0,:,:]=image
                train_labels[train_counter,0]=subject - 1
                train_counter += 1

            for pose in test_selector:
                image = self.read_pgm(address + str(subject) + "\\" + str(pose+1) + ".pgm", byteorder='<')
                testset[test_counter, 0, :, :] = image
                test_labels[test_counter, 0] = subject - 1
                test_counter += 1

        return trainset,train_labels,testset,test_labels

