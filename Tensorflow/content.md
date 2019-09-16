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

#### tf session
 