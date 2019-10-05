# concrete content

[2017 New New New TensorFlow CODE repository](https://github.com/MorvanZhou/Tensorflow-Tutorial)


## Introduction [Artifical Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)

[Optimization](https://en.wikipedia.org/wiki/Gradient_descent)
	
**Brief**
	
	 	Input -> Process layer 1 2 3 ......-> Results
			result provide feedback (reward) 
			Activation function
			example(Training data) -> Training
			final(testing data) -> testing
	
**Definition**:

	 Artifical neural network(ANN) is a collection of connected artifician neurons, which inspired by Bio-neural network.
	supervised or not: manualley labeld or not
	
**Method**: 

	primarily focus on oprimization method 
			Gradient Descent
			Cost function
			obtaining Global minimum or local Minimum
			a first order iterative optimization algorithm

			asymptotic rate of convergence is inferior???

## Preparation(Tensor Flow):

Installation

	```
		pip3 install --upgrade tensorflow

		import tensorflow
	```

	Input(example) -> Output(Results)
	usage: prediction

	overfitting line 

[more](https://en.wikipedia.org/wiki/Overfitting)

	underfitting line

## Basis Architecture
#### tensor flow basis

	input -> hide layer -> output
	1. input : dataset 1 2 3 4.......
	2. hide layer: weights, biases, activation rules
	3. output
	4. Gradient Descent optimize metrics......

![Gradient Descent](https://www.tensorflow.org/images/tensors_flowing.gif)

#### tensor flow basis coding

	key idea of TF: error back-propagation algorithm through optimization method
	optimization method this time: gradient descent

[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_UseCase_001.py)

#### tf Session

[FuLL Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_UseCase_session.py)

#### tf Variable

[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_UseCase_variable.py)

#### tf Placeholder

#### tf Activation Function

#### manual add layer
[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_add_layer.py)

#### Basis TensorFlow
	1. basis architecture
[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_basis_architecture.py)
	2. visualization
[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_basis_architecture_visualization.py)
	



#### Speeding up training and Optimizer
	1. stochastic Gradient Descent(SGD)
	2. Momentum (using push) 
	3. AdaGrad (using rule to discpline)
	4. RMSProp (basic rule + rule)
	5. Adam (fully fule + rule)
![diagram](http://cs231n.github.io/assets/nn3/opt2.gif)
![diagram](http://cs231n.github.io/assets/nn3/opt1.gif)
[more link](http://cs231n.github.io/neural-networks-3/)



#### Tensorboard
inputs -> layer 1 2 3 4..... ->outputs
overview graph:       as well as components graph:
[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_tensorboard.py)
[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_tensorboard2.py)



## Advanced Content
#### 1. classification solution case
	regression problem sovling -> output value
	classification problem solving -> output possiblity

	activation function:
		regression using relu
		classification using softmax

	cost function:
		regression: real - prediction
		classification: cross_entropy

	optimizer: both gradient descent

[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_advanced_classification.py)

#### 2. solving overfitting
![diagram from MOFAN](https://morvanzhou.github.io/static/results/tensorflow/5_02_1.png)
	For general situation rather than specific training data
	states: underfit just_right overfit
	Tensorflow solve: drop out
[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_advanced_overfitting.py)

#### 3. Convoltutional neural network(CNN)
	Purpose: image text vedio (computer vision recognization
	
	Method: object Components.s.s -> combine -> combine ..->.->.... -> object classfied

	pooling? 

	example: image->covolution->max pooling->connected -> classified

[great resource](https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/64063017560923)

#### 4. CNN buidling 1
	**code concept**:image recognition, gradually compress hight and width in order to increase thickness (processing data in validation below 'cp') 
	
	CNN filtering quintessence

	**Overview picture in example**:

	image -> convolutionary -> pooling (cp)-> cp .......-> tensorflow(fully connected tf) -> tf -> classifer  

	key attribute in code: patch(compression related with padding) stride(step related pooling)

	padding method: valid padding ; same padding
	pooling method: max pooling ; average pooling

#### 5. CNN building 2
	TensorFlow Libraries implment CNN
	conv2a function
	pooling module
	padding module
	key features: stride, patch, padding, pooling 
[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_advanced_CNN2.py)

#### 6. CNN building 3
	feeling: capture data -> reframe(reshape) data -> optimizer data
	practical: padding, pooling, conv2d, adamoptimizer 

[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_advanced_CNN3.py)

#### 7. Saver

	purpose: store the trained parameter like(weights and biases)
	issue: save parameters -> reload parameters
	key: tf.train.saver

[Full code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0917_advanced_saver.py)

#### 8. Recurrent Neural Network (RNN) 1
	key: order matter issue solutions (sentence)

	common method: long term short term memory (LSTM)

	collecting continum experience from previous experience(state) -> incfluence -> next experience (state)

	using LSTM enable avoid 
		gradient vanishing and gradient exploding 

	LSTM structure: Write(pre) - forget (mid) - read (after) ------------three gates handle


[resource1](https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/64063017560923#)
[resource2](https://classroom.udacity.com/courses/ud730/lessons/6378983156/concepts/63770919610923#)	

#### 9. Recurrent Neural Network (RNN) 2 Classification
RNN LSTM
	
	inputs hiden layer -> Cell -> outputs hiden layer

	every cell has write forget read gates function
		solving for gradient vanishing or gradient exploding

		1. initial_states
		2. c_state
		3. m_state (basis only have this state)

	Tensorflow
	'''
		tf.nn.dynamic_rnn()

	'''
	difficulties: time_major, transpose

[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0919_advanced_RNN_1.py)
	

Noremal style or Tensorflow style
[Extensive BPTT](https://r2rt.com/styles-of-truncated-backpropagation.html)


#### 10. Recurrent Neural Network (RNN) 3 Regression

	1. data handling --- batch
	2. LSTM (input, cell, output)
	3. Optimizer
	'''
		key attribute: input_size, output_size, cell_size

		1. input layer

		2. cell
			Basic LSTM Cell
			Lstm_cell.zero_state
			tf.nn.dynamic_rnn

		3. output leyer
			losses = tf.nn.seq2seq.sequence_loss_by_example()

	'''

	general difficulties: 
		datashape => reshape function

RNN key: Truncate Backpropagration  Again:
[Extensive BPTT](https://r2rt.com/styles-of-truncated-backpropagation.html)

[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0919_advanced_RNN_2.py)


#### 11. Recurrent Neural Network (RNN) 4 Visualization
	
	plt_plot()

[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0919_advanced_Aensorflow_visualization.py)


#### 12. Autoencoder
	Much like PCA
	it is unsupervised learning
	feel like trained compiler
	encoder -> activation function
	decoder -> activation function

	tf.nn.sigmoid

	difficulties: learning rate

[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0919_advanced_Autoencoder.py)

#### 13. Scope
	corresponded with variable
	'''
	tf.name_scope
	tf.variable_scope

		tf.get_variable() 
		tf.Variable()

	'''

	Understanding scope mechanism can reuse variable 
	especially in situation(RNN  TrainConfig && TestConfig)
	'''
		scope.reuse_variables()

	'''
[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0919_advanced_scope0.py)
[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0919_advanced_scope.py)



#### 14. Batch Normalization (BN)
	significance: if not, neurons lose efficacy, sensor capacity vanish
	procedure:forwarding layer process BN after every time activation function

[Full Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Tensorflow/TensorFlow_0919_advanced_BatchNormalization.py)


[extensive resource](https://arxiv.org/abs/1502.03167)




#### 15. Transfer Learning 

(continum to learn when start again these bracket should be vanished)

	current exsited model to operate
	
[Extensive resource](http://cs231n.github.io/transfer-learning/)