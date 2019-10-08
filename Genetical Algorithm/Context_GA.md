# Genetical Algorithm Context with current Code

	Fixed interval GA -> runcase -> problemGen


![context_GA_procedure](https://raw.githubusercontent.com/waylen94/Machine-Learning-Case-Study/master/Genetical%20Algorithm/context_GA_procedure_version_2.png)
```python

final_pop = ea.evolve(generator=prob_bi.generator, 
                          evaluator=prob_bi.evaluator, 
                          pop_size=prob_bi.info["parameters"]["size"],
                          maximize=prob_bi.maximize, #A flag to denote maximize (here is True)
                          bounder=prob_bi.bounder, #A basic bounding function (lower bound and upper bound)
                          max_generations=prob_bi.info["parameters"]["generation"],
                          crossover_rate=0.8,
                          mutation_rate=0.2)

```

## Generator 

	Initial random network topology base on binary

```python

	def generator(self, random, args):
		candidate = [random.choice([0, 1]) for _ in range(0, self.dimension_bits)]
		return candidate

```

## Evaluator

	Decode – network 
		Return fitness considered by Pareto multi-objective (decoy path, defense cost, MTTSF)
```python
	def evaluator(self, candidates, args):
        fitness = []
        for c_binary in candidates:
            print("binary solution:", c_binary)
            #shuffle net work
            net = add_solution(self.decoy_net, c_binary, self.info, self.decoy_list)
            newnet = add_attacker(net)
#             print("Add attacker:")
#             printNet(newnet)
            harm = constructHARM(newnet)

            f1 = decoyPath(harm) 
            f3 = solutionCost(c_binary, self.info)
            f2 = 0.0
            for i in range(0, self.sim_num):
                f2 += computeMTTSF(harm, self.net, self.info["threshold"])
                
#             print(f1, f2, f3)
            fitness.append(emo.Pareto([f1, float(f2/self.sim_num), f3])) # a Pareto multi-objective solution

        return fitness



         """Represents a Pareto multiobjective solution.
    
    A Pareto solution is a set of multiobjective values that can be 
    compared to other Pareto values using Pareto preference. This means 
    that a solution dominates, or is better than, another solution if it 
    is better than or equal to the other solution in all objectives and
    strictly better in at least one objective.
    
    Since some problems may mix maximization and minimization among
    different objectives, an optional `maximize` parameter may be
    passed upon construction of the Pareto object. This parameter
    may be a list of Booleans of the same length as the set of 
    objective values. If this parameter is used, then the `maximize`
    parameter of the evolutionary computation's ``evolve`` method 
    should be left as the default True value in order to avoid
    confusion. (Setting the `evolve`'s parameter to False would
    essentially invert all of the Booleans in the Pareto `maximize`
    list.) So, if all objectives are of the same type (either
    maximization or minimization), then it is best simply to use
    the `maximize` parameter of the `evolve` method and to leave
    the `maximize` parameter of the Pareto initialization set to
    its default True value. However, if the objectives are mixed
    maximization and minimization, it is best to leave the ``evolve``'s
    `maximize` parameter set to its default True value and specify
    the Pareto's `maximize` list to the appropriate Booleans.
    
    The typical usage is as follows::
    
        @inspyred.ec.evaluators.evaluator
        def my_evaluator(candidate, args):
            obj1 = 1 # Calculate objective 1
            obj2 = 2 # Calculate objective 2
            obj3 = 3 # Calculate objective 3
            return emo.Pareto([obj1, obj2, obj3])
    
    """
```

## Observer

	None
```python

```

## Terminator

	Inspyred.ec.terminators.generation_termination
```python
	ea.terminator = [inspyred.ec.terminators.generation_termination]
```
## Selector

	NSGA2 based Tournament_selection

    Key:
    
```python
	ea = inspyred.ec.emo.NSGA2(prng)

	class NSGA2(ec.EvolutionaryComputation):

	    def __init__(self, random):
	        ec.EvolutionaryComputation.__init__(self, random)
	        self.archiver = ec.archivers.best_archiver
	        self.replacer = ec.replacers.nsga_replacement
	        self.selector = ec.selectors.tournament_selection
	    
	    def evolve(self, generator, evaluator, pop_size=100, seeds=None, maximize=True, bounder=None, **args):
	        args.setdefault('num_selected', pop_size)
	        args.setdefault('tournament_size', 2)
	        return ec.EvolutionaryComputation.evolve(self, generator, evaluator, pop_size, seeds, maximize, bounder, **args)

	def tournament_selection(random, population, args):
    """
    .. Arguments:
       random -- the random number generator object
       population -- the population of individuals
       args -- a dictionary of keyword arguments

    Optional keyword arguments in args:
    
    - *num_selected* -- the number of individuals to be selected (default 1)
    - *tournament_size* -- the tournament size (default 2)
    
    """
    num_selected = args.setdefault('num_selected', 1)
    tournament_size = args.setdefault('tournament_size', 2)
    if tournament_size > len(population):
        tournament_size = len(population)
    selected = []
    for _ in range(num_selected):
        tourn = random.sample(population, tournament_size)
        selected.append(max(tourn))
    return selected

    
```
## Variator


    Key:


	Crossover: n-point-crossover
	Mutation: bit_flip_mutation
```python
	ea.variator = [inspyred.ec.variators.n_point_crossover, 
                   inspyred.ec.variators.bit_flip_mutation]


    @crossover
def n_point_crossover(random, mom, dad, args):
    """Return the offspring of n-point crossover on the candidates.

    This function performs n-point crossover (NPX). It selects *n* 
    random points without replacement at which to 'cut' the candidate 
    solutions and recombine them.

    .. Arguments:
       random -- the random number generator object
       mom -- the first parent candidate
       dad -- the second parent candidate
       args -- a dictionary of keyword arguments

    Optional keyword arguments in args:
    
    - *crossover_rate* -- the rate at which crossover is performed 
      (default 1.0)
    - *num_crossover_points* -- the number of crossover points used (default 1)
    
    """
    crossover_rate = args.setdefault('crossover_rate', 1.0)
    num_crossover_points = args.setdefault('num_crossover_points', 1)
    children = []
    if random.random() < crossover_rate:
        num_cuts = min(len(mom)-1, num_crossover_points)
        cut_points = random.sample(range(1, len(mom)), num_cuts)
        cut_points.sort()
        bro = copy.copy(dad)
        sis = copy.copy(mom)
        normal = True
        for i, (m, d) in enumerate(zip(mom, dad)):
            if i in cut_points:
                normal = not normal
            if not normal:
                bro[i] = m
                sis[i] = d
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children
    


    @mutator
def bit_flip_mutation(random, candidate, args):
    """Return the mutants produced by bit-flip mutation on the candidates.

    This function performs bit-flip mutation. If a candidate solution contains
    non-binary values, this function leaves it unchanged.

    .. Arguments:
       random -- the random number generator object
       candidate -- the candidate solution
       args -- a dictionary of keyword arguments

    Optional keyword arguments in args:
    
    - *mutation_rate* -- the rate at which mutation is performed (default 0.1)
    
    The mutation rate is applied on a bit by bit basis.
    
    """
    rate = args.setdefault('mutation_rate', 0.1)
    mutant = copy.copy(candidate)
    if len(mutant) == len([x for x in mutant if x in [0, 1]]):
        for i, m in enumerate(mutant):
            if random.random() < rate:
                mutant[i] = (m + 1) % 2
    return mutant               
```

## Replacer

	NSGA2_nsga_replacement


    Key:
    
```python

	def nsga_replacement(random, population, parents, offspring, args):
    """Replaces population using the non-dominated sorting technique from NSGA-II.
    
    .. Arguments:
       random -- the random number generator object
       population -- the population of individuals
       parents -- the list of parent individuals
       offspring -- the list of offspring individuals
       args -- a dictionary of keyword arguments
    
    """
    survivors = []
    combined = list(population)
    combined.extend(offspring)
    
    # Perform the non-dominated sorting to determine the fronts.
    fronts = []
    pop = set(range(len(combined)))
    while len(pop) > 0:
        front = []
        for p in pop:
            dominated = False
            for q in pop:
                if combined[p] < combined[q]:
                    dominated = True
                    break
            if not dominated:
                front.append(p)
        fronts.append([dict(individual=combined[f], index=f) for f in front])
        pop = pop - set(front)
    
    for i, front in enumerate(fronts):
        if len(survivors) + len(front) > len(population):
            # Determine the crowding distance.
            distance = [0 for _ in range(len(combined))]
            individuals = list(front)
            num_individuals = len(individuals)
            num_objectives = len(individuals[0]['individual'].fitness)
            for obj in range(num_objectives):
                individuals.sort(key=lambda x: x['individual'].fitness[obj])
                distance[individuals[0]['index']] = float('inf')
                distance[individuals[-1]['index']] = float('inf')
                for i in range(1, num_individuals-1):
                    distance[individuals[i]['index']] = (distance[individuals[i]['index']] + 
                                                         (individuals[i+1]['individual'].fitness[obj] - 
                                                          individuals[i-1]['individual'].fitness[obj]))
                
            crowd = [dict(dist=distance[f['index']], index=f['index']) for f in front]
            crowd.sort(key=lambda x: x['dist'], reverse=True)
            last_rank = [combined[c['index']] for c in crowd]
            r = 0
            num_added = 0
            num_left_to_add = len(population) - len(survivors)
            while r < len(last_rank) and num_added < num_left_to_add:
                if last_rank[r] not in survivors:
                    survivors.append(last_rank[r])
                    num_added += 1
                r += 1
            # If we've filled out our survivor list, then stop.
            # Otherwise, process the next front in the list.
            if len(survivors) == len(population):
                break
        else:
            for f in front:
                if f['individual'] not in survivors:
                    survivors.append(f['individual'])
    return survivors
```

## Migratory

	None- server for multi-population typically for island model
```python

```

## Archiver 

	NSGA2 based best-archiver

    Key:

```python

	def best_archiver(random, population, archive, args):
	    """Archive only the best individual(s).
	    
	    This function archives the best solutions and removes inferior ones.
	    If the comparison operators have been overloaded to define Pareto
	    preference (as in the ``Pareto`` class), then this archiver will form 
	    a Pareto archive.
	    
	    .. Arguments:
	       random -- the random number generator object
	       population -- the population of individuals
	       archive -- the current archive of individuals
	       args -- a dictionary of keyword arguments
	    
	    """
	    new_archive = archive
	    for ind in population:
	        if len(new_archive) == 0:
	            new_archive.append(ind)
	        else:
	            should_remove = []
	            should_add = True
	            for a in new_archive:
	                if ind.candidate == a.candidate:
	                    should_add = False
	                    break
	                elif ind < a:
	                    should_add = False
	                elif ind > a:
	                    should_remove.append(a)
	            for r in should_remove:
	                new_archive.remove(r)
	            if should_add:
	                new_archive.append(ind)
	    return new_archive
```
