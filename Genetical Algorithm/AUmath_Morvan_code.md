#Genetical Algorithm learning record

**resource from Au.mathworks + Morvan form below**

[AU Mathworks GA](https://au.mathworks.com/help/gads/how-the-genetic-algorithm-works.html)
[Morvan GA](https://morvanzhou.github.io/)

Biological Evolution

1. Selection

2. Crossover

3. Mutation

Terminology:

* Fitness: objective function
* Individuals: Apply fitness function
* Populations and Generations: array of individuals
* Diversity: distance between individuals
* Fitness value: value of fitness
* Parent and Children: individuas enable reproduce

Outline of the Algorithm:

	Initial population 
		Eliteare 	individuals 	crossover 	mutation

	Terminar condition
		Generations ; 
		time; 
		fitness limit; 
		tolerance

**resource from Media**

Reflects the precoss of natural selection

Populations from Indicidual from Chromosome from Gene

Start
Generate the initial population

	Compute fitness
	Repeat
		Selection
		Crossover
		Mutation
		Compute Fitness

		Until population has converged
			Stop

**Morvan Version Learning Code**

Functions:

	get_fitness
	translateDNA
	select
	crossover
	mutate

Attributes:

	DNA_size
	Pop_size
	Cross_Rate
	Mutation_Rate
	N_Generations
	X_Bound

[GA Basis Code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Genetical%20Algorithm/Genetic%20Algorithm_basis.py)



[GA based String Matching Algorithm](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Genetical%20Algorithm/Genetic%20Algorithm_string_match.py)

Key: ASCII


[Travel Sales Problem(TSP)](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Genetical%20Algorithm/Genetic%20Algorithm_Travel_Sales.py)

key: DNA encode with Traveling order



[Microbial Generical Problem](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Genetical%20Algorithm/Genetic%20Algorithm_Microbial_GA.py)

key: Elitism fixed


**Advance GA -> Evolution Strategy**

[basis ES code](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Genetical%20Algorithm/Evolution%20strategy.py)


[1+1-ES](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Genetical%20Algorithm/Evolution%20strategy_1%2B1.py)




**Current Natural Evolutionary Strategy (NES) + Policy gradient**

Mathmatics is essential

1. Average 
2. Standard Deviation
3. Covariance Matrix


[Neural Network Gradient Descent meet with Genetical Algorithm](https://github.com/waylen94/Machine-Learning-Case-Study/blob/master/Genetical%20Algorithm/Evolution%20strategy_gradient.py)









