Online Semi-supervised Growing When Required Neural Gas
-------------------------------------------------------

Python implementation of the Online Semi-supervised Growing When Required
(OSS-GWR)Neural Gas by Parisi et al. [1].

A detailed description of the algorithm is given in the reference, here
I just give the flavor of the algorithm.
In essence, you start with two random neurons, disconnected, represented
by a vector in an n-dimensional space. At
 every iteration, you produce a (potentially labelled) input sample. The
 best and second best matching neurons (i.e. closest in space to the
 input sample) are found, and a connection between them is created.
 When the neural gas is presented with a new data point,
 if the already present neurons are close enough to it,
 the positions of the best and second best matching neurons are slightly
  updated to better represent the data. On the other hand, if the
  closest
 neurons are too far off from the sample, a new neuron is created and
 connected to the best and second best matching neurons (hence the
 Growing When Required).

 Basically, you have a cloud of points (neurons) in space
  which learns to match the distribution of your data, in which
  connections are created between the points and new points are generated
  whenever needed to speed up the process.

  The points can be either labelled or unlabelled. Whenever labels are
  present, they are propagated through the gas, so that knowledge
  stored in the labels of existing neurons can be used to fill in the
  labels when unlabelled data is presented.

  Then the idea is that you can process data of different sensory
  streams in separate hierarchical chains of gases, where you consider
  increasingly larger time sequences of your data, and make them converge
  in a higher-dimensional gas which learns the distribution of your
  multimodal data (the hierarchically-structured chain is still
  in progress).

[1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
Emergence of multimodal action representations from neural network
 self-organization. Cognitive Systems Research, 43, 208-221.