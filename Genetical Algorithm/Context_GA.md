#Genetical Algorithm Context with current Code

	Fixed interval GA -> runcase -> problemGen

Generator 

	Initial random network topology base on binary

	```python
		def generator(self, random, args):
	        candidate = [random.choice([0, 1]) for _ in range(0, self.dimension_bits)]
	        return candidate
	```

Evaluator

	Decode – network 
		Return fitness considered by Pareto multi-objective (decoy path, defense cost, MTTSF)

Observer

	None


Terminator

	Inspyred.ec.terminators.generation_termination

Selector

	NSGA2 based Tournament_selection

Variator

	Crossover: n-point-crossover
	Mutation: bit_flip_mutation

Replacer

	NSGA2_nsga_replacement

Migratory

	None- server for multi-population typically for island model

Archiver 

	NSGA2 based best-archiver
