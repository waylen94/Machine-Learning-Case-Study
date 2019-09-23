# 20190920 Tensorflow low level APIs 

	Tensor flow Core
[Learning Resource](https://www.tensorflow.org/guide/low_level_intro)

- [x] Introduction ------------20190920 
- [x] Tensors -----------------20190920
- [x] Variables ---------------20190921
- [x] Graphs and Sessions -----20190921
- [x] Save and Restore --------20190922
- [x] Control Flow ------------20190922
- [x] Ragged Tensors ----------20190922


## Introduction
Tensorflow core	**Purpose**: Csutomized the own Tensorflow program, tensorflow runtime, operation

Hight level **components** in this low level environment: datasets, layers, feature columns, training loop

**Tensor values**: A tensor consists of a set of primitive value shaped into an array of any number of dimensions
	
	tensor units: rank shape

**WalkThough** 1. Building graph 2. Running
	
	graph consists of: tf.Operation (nodes) + tf.Tensor (edges)

A **session** encapsulates the state of the TensorFlow runtime; TensorFlow operations

A **placeholder** is a promise to provide a value later like function argument
	
	feed_dict = { }

**Datasets** tf.data streaming data into a model 
	
	importing Data

**Layers** Modify values in the graph to get new outputs with the same inputs
	
	package both (variables + operations)
		Creating initializing executing

**Feature** ??????????????????????????????????

**Training** Classification && Regression model
	
	Sample: I. Define data -> II. Define Model -> III. Loss -> IV. Training

	Procedure: I. tf.constant() ---------------------Define simulation data
				II. tf.layers.Dense() ---------------inputs outputs execution
				III. tf.losses.mean_squared.error() --------Optimization losses
				IV. tf.train.GradientDescentOptimizer() ----Optimization method

	Mean Square error, a standard loss for regression problems

**Optimizers:**  Standard optimization algorithm
	
	They incrementally change each variable in order to minimize the loss

	tf.train.Optimizer()



## Tensors
run computations involving tensors

	TensorFlow represents tensors as n-dmensional arrays of base datatypes
	
	Properties: data type ; data shape

**Rank:**  number of dimensions
	
	Synonyms: order, degree, n-dimension

	relation with math:
	1. Rank 0 --------- Scalar ( Magnitude )
	2. Rank 1 --------- Vector ( Magnitude + Direction)
	3. Rank 2 --------- Matrix ( Table of numbers )
	4. Rank 3 --------- 3-Tensor ( Cube of numbers)
	5. Rnak n --------- n-Tensor

	tf.rank()

**Shape:** number of elements in each dimension
	tensorflow -- number of elements -- product of sizes of all its shapes
	
	tf.reshape: keeping its elements fixed, changing shape

	Data types: tf.cast()

	Evaluating Tensors                      Printing Tensors  

## Variables
represent shared, persistent, state manipulated by the program

**Creating** tf.get_variable('name',shape)

	Variable collections: Access disconnected parts of a atensor flow program

	tf.GraphKeys.GLOBAL_VARIABLES
	tf.GraphKeys.TRAINABLE_VARIABLES

**Device Placement**: place variables on particular devices ????????

**Initializing variables**
	
	tf.global_variables_initializer()

**Using variables:** Tensorflow graph simply treat variables like a normal tf.Tensor

**Share variables** ???????????

	Scope decide (Creating new OR reuse existing ones)

## Graphs and Sessions
**Concept**: dataflwo graph represent your comutationin terms of the dependencies between individual operations 

**low level programming model** 
		Define dataflow graph -> create Tensorflow session -> run

**reason of dataflow graphs** Dataflow: common programming model for parallel computing 

	nodes: units off computation 
	edges: data consumed or produced by a computation 


**tf.graph**:

	graph structure nodes and edges

	graph collections: general mechanism for storing cllections of metadata

Building tf.graph 
	tf.Operation (node)
							combine to 	tf.Graph
	tf.Tensor (edge) 			

	providing Default graph as implicit arguments to All API function in some context

**Naming operations** automatically or name-scope
placting operations on different devices tf.device

**tensor like objects** 
	tf.Tensor 
	tf.Variable 
	numpy.ndarray
	list
	scalar python types 
	bool 
	float 
	int 
	str

**Executing agraph in tf.Session**
	Operation Procedure

		tf.Session() as sess:
			sess.run()

**TensorBoard** Graph visualizer

**Multiple Graphs Programming**


## Save and Restore
	tf.train.Saver    operating **save** and **restore**

	**Save and Restore variables**
		tf.train.Saver
			tf.train.Saver.save
			tf.train.Saver.restore

	**Specific variables to save and restore**

	**Inspect variables in a checkpoint**

	**save and restore model**

	**Build and load a SavedModel** Model can be used in other infrastructure (device) based framework
		~~more about save restore model~~
		
## Control Flow
	~~Content~~
	Graph code?????????


## Ragged Tensors
	~~Content~~
	Tensor? Ragged?????????????



