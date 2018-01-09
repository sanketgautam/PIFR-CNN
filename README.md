# Pose Invariant Face Recognition using Covolutional Neural Network

This repository contains the implementation of Convolution Neural Network For Pose Invariant Face Recognition in Python.

### Datasets Used

**Decription:** AT&T Faces ( 'The ORL Database of Faces' )

**URL:** http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

### Getting Started
To run this projects execute `python file cnn.py`

This network uses following phases for training & testing,

* Phase I - Initially, CNN is used for feature extraction
* Phase II -  After that, last layer (i.e. Softmax Classifier) is replaced with our custom classifier, Extreme Learning Machine (ELM).

### References

* [Convolutional Neural Networks - CS231](http://cs231n.github.io/convolutional-networks/)
* [Pose Invariant Face Recognition - ieeexplore](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=840642)