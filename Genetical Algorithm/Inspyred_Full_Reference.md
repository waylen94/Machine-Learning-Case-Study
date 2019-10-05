# Genetical Algorithm Inspyred Library full reference Weilun's learning-record 

Example section


# Content context:

I. standard
II. customized
III. advanced 

I.	Standard Algorithm

	1.	Genetic algorithm
	2.	Evolution strategy
	3.	Simulated annealing
	4.	Differential evolution algorithm
	5.	Estimation of distribution algorithm
	6.	Pareto archived evolution strategy (PAES)
	7.	Non-dominated sorted genetic algorithm (NSGA II)
	8.	Particle swarm optimization
	9.	Ant colony optimization

II.	Customized algorithm

	1.	Custom evolutionary computation
	2.	Custom archive
	3.	Custom observer
	4.	Custom replacer
	5.	Custom selector
	6.	Custom terminator
	7.	Custom variator
 
III.	Advanced usage

	1.	Discrete optimization
	2.	Evaluating individuals concurrently
	3.	Island models
	4.	Replacement via Niching

Code example:
[Evolving polygons]
[Lunar explorer]

	Space probe(orbital, height, mass, boost velocity, initial velocity)

Library reference

I.	Library reference – Evolutionary computation

1.	ec
2.	emo
3.	analysis
4.	utilities
5.	operators
		a.	archivers
		b.	evaluators
		c.	generators
		d.	migrators
		e.	observers
		f.	replacers
		g.	selectors
		h.	terminators
		i.	variators

II.	Swarm intelligence

	1.	Swarm
	2.	topologies

III.	Benchmark problems

	1.	Benchmarks
	2.	Single-objective Benchmark
	3.	Multi-objective Benchmark
	4.	Discrete Optimization Benchamrks

#	1. ec-Evolutionary computation framework 

A framework creating evolutionary computation

Class inspyred.ec.Bounder (lower, upper)

	Example
	Usage: Bounder(0,1) or ([0,0,0],[1,1,1]) => [0.1,-0.2,3] =>[0.1,0,1]

	Meaning: evolutionary operators respect the legal bounds for candidates 

	'''
		
	'''


Class inspyred.ec.DiscreteBounder(values)

	Example 
	Usage: values=[1,4,8,16]     [6,10,13,3] => [4,8,16,4]


Class inspyred.ec.Individual(candidate, maxmize)
	
	Meaning: represent individual in an evolutionary computation
	Usage: candidates owned fitness function generated fitness value ruled by maximize above


Class inspyred.ec.EvolutionaryComputation(random)

	Encapsulates components of a genetic evolutionary computation
	1.	Selection mechanism
	2.	Variation operators 
	3.	Replacement mechanism
	4.	Migration scheme 
	5.	Archival mechanism
	6.	Teminators
	7.	Observers
	Ps. Variator observer terminator can be specified as lists
		(Pipeline) (sequence)(or)

	General 
	Function evolve(generator,
			Evaluator,
			pop_size=100,
			seeds=None,
			maximize=true,
			bounder=None,
			args)


	Generator generate candidate
	Evaluator evaluate candidate
	Pop_size number of individuals
	Seeds iterable collection candidate in initial
	Maximize Boolean of maximization
	Bounder bound candidate
	Args dictionary of keyword arguments


Below GA ES EDA DEA SA   Inhereted Evolutionary computation

Class inspyred.ec.GA(random)

	Key: rank selection; n-point crossover; bit-flip mutation
	Optional parameter:
	Number_selecter: individuals number to be selected
	Crossover_rate: rate at crossover is performed
	Num_crossover_points: crossover points
	Mutation_rate: mutation
	Num_elites: elites

Class inspyred.ec.ES(random)

	Canonical evolution strategy
	Key: selection, mutation, replacement with candidate solution of a sequence of real value

Class inspyred.ec.EDA(random)

	Canonical estimation of distribution algorithm
	Key: truncation selection, estimation of distribution variation, generation replacement, candidate solution is a sequence of real values

Class inspyred.ec.DEA(random)

	Differential evolutionary algorithm
	Key: tournament selection, heuristic crossover, Gaussian mutation, stedy-state replacement, candidate solution of real values

Class inspyred.ec.SA(random)

	Simulated annealing
	Key: selection, Gaussian mutation, simulated annealing replacement, candidate solution of real values

### emo (Evolutionary multi-objective optimization)

	Framework making multi objective evolutionary computation 

Class inspyred.ec.emo.NSGA2

	Non-dominated sorting genetic algorithm of Kalyanmoy Deb et al.
	Key: non-dominated sorting with binary tournament selection, replacement, pareto archival strategy

Class inspyred.ec.emo.PAES(random)

	Pareto archived evlution strategy of Joshua Knowles and David Corne
	Key: (1+1)-ES, adaptive grid archive replacement

Class inspyred.ec.emo.Pareto(values=None, maximize=True)

	Pareto multi-objective solution
	Key: better rely on if it is better than or equal to the other solution in all objectives and strictly better in at least one objective


### .analysis ---optimization result analysis

	Analysis methods for the results of evolutionary computations

Inspyred.ec.analysis.allele_plot(file, normalize=false,alleles=None,generation=None)

	(single) meaning: plot the alleles from each generation from the individuals file

Inspyred.ec.analysis.fitness_statistics(population)

	Basic statistics of the populations fitness values

Inspyred.ec.analysis.generation_plot(file,errorbars=true)

	(single)basic statistics of the population’s fitness values

Inspyred.ec.analysis.hypervolume(pareto_set,reference_point=None)

	Calculates hypervolume by slicing objectives (HSO)
[resource](                 )

### Utilities ------optimization utility functions

Class inspyred.ec.utilities.objectify(func)

	Key: provide each object own set of independent attributes

Inpyred.ec.utilities.memoize(func=None, maxlen=None)

	Cache a function’s return value each time’s called instead of being re-evaluated.



### Operators 

1.	Arcchiver  store separate solution
2.	Evaluator  fitness value
3.	Generator  generate new candidates
4.	Migratory individual migration
5.	Observers  view progress
6.	Replacer  survivors of generation
7.	Selector  parents of generation
8.	Terminators  termination
9.	Variators  modify candidate


###	Archivers

General arguments:
1.	Random
2.	Population
3.	Archive
4.	Args
Inspyred.ec.archivers.adaptive_grid_archiver(1,2,3,4)

	Meaning: best individuals, fixed grid
	Typically for Pareto archived evolution strategy (PAES)

Inspyred.ec.archivers.best_archiver(1,2,3,4)

	Meaning: best, remove inferior
	Typically for pareto archive
 
Inspyred.ec.archivers.default_archiver(1,2,3,4)

	Do nothing
		Default return existing archive

Inspyred.ec.archivers.population_archivers(1,2,3,4)

	Meaning: replace archive with pop

###	Evaluator
General arguments:
1.	Candidates
2.	Args
Inspyred.ec.evaluators.evaluator(evaluate)

	Evaluate inside contain fitness function

Inspyred.ec.evaluators.parallel_evaluation_mp(1,2)

	Multiprocessing evaluation

Inspyred.ec.evaluators.parallel_evaluation_pp(1,2)

	Parallel python evaluation

###	Generators
	Create initial set of candidates
	General arguments
	1.	Random 
	2.	Args

Class inspyred.ec.generators.diversify(generator)

	Ensure uniqueness

Inspyred.ec.generaors.Strategize(generator)

	Extend candidate with strategy parameters

### migrators

	return uploaded population
	general arguments
	1.	random
	2.	population
	3.	args
	typically for island model evolutionary computation

class inspyred.ec.migrators.MultiprocessingMigrator(Max_migrants=1)

	multiprocessing migration

inspyred.ec.migrators.default_migration(1,2,3)

	do nothing return existing pop

### Observers
	General arguments:
	Population 
	Num_generations
	Num_evaluations
	Args

Class inspyred.ec.observers.emailobserver
Inspyred.ec.observers.archive_observer 
Inspyred.ec.observers.best_observer
Inspyred.ec.observers.default_observer 
Inspyred.ec.observers.file_observer 
Inspyred.ec.observers.plot_observer 
Inspyred.ec.observers.population_observer 
Inspyred.ec.observers.stats_observer 

### Replacers
	Survivor mechanism
	General arguments:
	1.	Random
	2.	Population
	3.	Parents
	4.	Offspring
	5.	Args
	Return surviving individual list
	Prefix: inspyred.ec.replacers

Comma_replacements(1,2,3,4,5)

	‘comma’ replacement
	Key: replaced size of the offspring at least large as the original population

Crowding_replacement(1,2,3,4,5)

	Crowding replacement
	Key: closest individual to the current offspring replaced by the offspring

Default_replacement(1,2,3,4,5)

	Key: origin population

Generational_replacement(1,2,3,4,5)

	Key: offspring truncating to the pop size if larger

Nsga_replacement(1,2,3,4,5)

	Key: replaces population using non-dominated sorting technique from NSGA-II

Paes_replacement(1,2,3,4,5)

	Key: replaces using Pareto Archived evolution strategy method

Plus_replacement(1,2,3,4,5)

	Replaces by the best population many elements from the combined set of parents and offspring

Random_replacement(1,2,3,4,5)

	Key: replace random number of the pop

Simulated_annealing_replacement(1,2,3,4,5)

	Key: simulated annealing schedule

Steady_state_replacement(1,2,3,4,5)

	Key: keep at least individuals in the existing population

Truncation_replacement(1,2,3,4,5)

	Key: Best replaced from current population and offspring

###	terminators
	return Boolean value true for ending
	general arguments:
	1.	population
	2.	num_generations
	3.	num_evaluations
	4.	args
	prefix: inspyred.ec.terminators

average_fitness_termination(1,2,3,4)

	when average fitness near best fitness

default_termination(1,2,3,4)

	default always return true

diversity_termination(1,2,3,4)

	when population diversity less minimum

evaluation_termination(1,2,3,4)

	evaluation fitness meets or exceeds a maximum

generation_termination(1,2,3,4)

	number of generations meets or exceeds
	
no_improvement_termination(1,2,3,4)

	best fitness value none change for a certain number of generations

time_termination(1,2,3,4)

	when elapsed time meet

user_termination(1,2,3,4)

	when user press key to

###	variators
	return the list of modified individuals
	general arguments:
	1.	random
	2.	candidates
	3.	args
	crossover variators
		pair of parents => a pair of offspring
	mutation variators
		candidate => single mutant
	prefix: inspyred.ec.variators

default_variation(1,2,3)

	do nothing return the set of candidates

crossover(cross)

	function decorator
	sample: @crossover
		def cross(random, mom, dad, args)

arithmetic_crossover(random, mom, dad, args)

	(AX) arithmetic crossover
	Key: weight of allele 
		Return the offspring of arithmetic crossover

Blend_crossover(random, mom, dad, args)

	Blend crossover (BLX)
	Key: AX plus A bit of mutation

Heuristic_crossover(1,2,3)

	Heuristic crossover(HX)
	Used for particle swarm optimization required candidates can be pickled

Laplace_crossover(random, mom, dad, args)

	Laplace crossover(LX)
	Deep and Thakur proposed crossover mutation

N_point_crossover(random, mom, dad, args)

	n-point (NPX)
	random cut and recombine

partially_matched_crossover(random, mom, dad, args)

	partially matched crossover (PMX)
	used for discrete values permutations

simulated_binary_crossover(random, mom, dad, args)

	simulated binary crossover (SBX) with NSGA-II
	cross_over_rate
	Sbx_distribution_index    down    far ok
					Up    far   not ok

Uniform_crossover(random, mom, dad, args)

	uniform crossover(UX)
	biases coin flipped to determine offspring

mutator(mutate)

	function decorator
		example @ mutator
			def mutate(1,2,3)

bit_flip_mutation(1,2,3)

	bit_flip mutation
	key: bit-rate-flip
		no bit – unchanged

Gaussian_mutation(1,2,3)

	Gaussian mutation
	Key: mean, standard deviation, normal distribution

Inversion_mutation(1,2,3)

	Key: random location -> reverse sliced value

Nonuniform_mutation(1,2,3)

	Key: nonuniform mutation specified in Michalewicz ”GA+DS=Evolution Program” 1996

Random_reset_mutation(1,2,3)

	Key: randomly choosing new values

Scramble_mutation(1,2,3)

	Key: randomly location -> scramble the sliced value

#	 Swarm Intelligence

Class inspyred.swarm.ACS (random, components)

	Ant Colony system discrete optimization

Class inspyred.swarm.PSO(random)

	Basic particle swarm optimization algorithm
	Deb and Padhge proposed

Class inspyred.swarm.TrailComponent(element, value, maximize=True, delta=1,epsilon=1)

	Used as a discrete component of a trail of in ant colony optimization


Topologies ---- swarm topologies

	Return list of lists of neighbors
	Swarm intelligence --  particle swarms
	Make use of topologies to determine relationship

Inspyred.swarm.topologies.ring_topology(random,population,args)

	Key: ring topology – all particles in a specified sized neighborhoods

Inspyred.swarm.topologies.star_topology(random,population,args)

	Key: star topology --- all particles as neighbors for all other particles


# Benchmark Problems
1.	Benchmarks 
Benchmark optimization function
2.	Single objective benchmarks
3.	Multi-objective benchmarks
4.	Discrete optimization benchmarks

##	benchmarks
Class inspyred.benchmark(dimensions, objectives = 1)

	Abstract class define global optimization problem
				Generator(candidates, args)
				Evaluator(random, args)
	Public attribute
		Dimensions; objectives; bounder; maximize

Class inspyred.benchmarks.Binary(benchmark, dimension_bits)

	Existing benchmark problem
		Represent by binary

##	single-objective benchmarks

class inspyred.benchmarks.Ackley(dimensions=2)

	Ackley benchmark problem (global optimization problem)

Class inspyred.benchamrks.Griewank(dimensions = 2)

	Griewank benchmark problem (gop)

Class inspyred.benchmarks.Rastrigin(dimensions = 2)

	Rastrigin benchmark

Class inspyred.benchmarks.Rosebrock(dimensions = 2)

	Rosenbrock benchmark

Class inspyred.benchmarks.Schwefel(dimensions=2)

	Schwefel benchmark

Class inspyred.benchamrks.Sphere(dimensions = 2)

	Sphere benchmark

##	multi-objective benchmarks
class inspyred.benchmarks.Kursawe(dimensions = 2)

	kursawe multiobjective benchmark
	key: n-dimensions to two dimensions

class inspyred.benchmarks.DTLZ1-2-3-4-5-6-7
N-dimensional inputs to m-dimensional outputs

##	discrete optimization benchmarks
class inspyred.benchmarks.Knapsack(capacity, items, duplicates = false)

	knapsack benchmark
	key: find the set of maximal value 
		that fit within a knapsack of 
			fixed weight capacity

class inspyred.benchmarks.TSP(weights)

	traveling salesman benchmark
	key: find the shortest visit route






































