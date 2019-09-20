#20190920 Tensorflow low level APIs 
	Tensor flow Core
[Learning Resource](https://www.tensorflow.org/guide/low_level_intro)

- [ ] Introduction
- [ ] Tensors
- [ ] Variables
- [ ] Graphs and Sessions
- [ ] Save and Restore
- [ ] Control Flow
- [ ] Ragged Tensors


## Introduction
Tensorflow core	**Purpose**: Csutomized Tensorflow program tensorflow runtime operation

Hight level **components**: datasets, layers, feature columns, training loop

**Tensor values**: A tensor consists of a set of primitive value shaped into an array of any number of dimensions
	tensor units: rank shape

**WalkThough** 1. Building graph 2. Running
	graph consists of: tf.Operation (nodes) + tf.Tensor (edges)

A **session** encapsulates the state of the TensorFlow runtime; TensorFlow operations

A **placeholder** is a promise to provide a value later like function argument
	feed_dict = { }

**Datasets** tf.data streaming data into a model 
	importing Data

## Tensors


## Variables


## Graphs and Sessions


## Save and Restore


## Control Flow


## Ragged Tensors





