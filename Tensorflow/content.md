# concrete content
## Introduction [Artifical Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)

	[Optimization](https://en.wikipedia.org/wiki/Gradient_descent)
	
	**Brief**: Input -> Process layer 1 2 3 ......-> Results
			result provide feedback (reward) 
			Activation function
			example(Training data) -> Training
			final(testing data) -> testing
	
	**Definition**: Artifical neural network(ANN) is a collection of connected artifician neurons, which inspired by Bio-neural network.
	supervised or not: manualley labeld or not
	
	**Method**: primarily focus on oprimization method 
			Gradient Descent
			Cost function
			obtaining Global minimum or local Minimum
			a first order iterative optimization algorithm

			asymptotic rate of convergence is inferior???

## Preparation(Tensor Flow):
	Installation
	'''
		pip3 install --upgrade tensorflow

		import tensorflow
	'''
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

[Full Code](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf5_example2/full_code.py)

#### tf Session

#### tf Variable
 
#### tf Placeholder

#### tf Activation Function

#### manual add layer
[Full code]()
#### Basis TensorFlow
	1. basis architecture
[Full Code]()
	2. visualization
[Full code]()
	
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
[Full code]()

## Advanced Content
1. 

