# Pose Invariant Face Recognition using Covolutional Nueral Network

This repository contains the implementation of Convolution Neural Network For Pose Invariant Face Recognition in Python.

### Datasets Used

**Dataset:** AT&T Faces ( 'The ORL Database of Faces' )

**Link:** http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

### Getting Started
To run this projects execute `python file cnn.py`

This network uses two different layers for training & testing as follows,

* Phase I - CNN is used for feature extraction in the
* Phase II -  After training the neural network, we replace the last layer (i.e. Softmax Classifier) with our custom classifier, Extreme Learning Machine.

### References

* [Convolutional Neural Networks - CS231](http://cs231n.github.io/convolutional-networks/)
* [Pose Invariant Face Recognition - ieeexplore](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=840642)