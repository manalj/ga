## Description
This script implements a simple genetic algorithm. The implementation was done in Python 2.7.

## Approach
Overall, the implementation starts by initializing a random population, we then loop until we reach the best fitness, or we reach a maximum number of generations we set. Each file includes a specific fitness function to the problem it's treating.

The implementation starts by initializing a random population, then we loop until we reach the best fitness, or we reach a tolerated number of generations. The example used in this implementation is maximizing the integer value a bitstring (a string of binary values). The fitness is calculated by calculating the integer value, given a string of binary values (bitstring), e.g: fitness of [0010] is 2.
To proceed in the algorithm, we choose two parents according to Roulette wheel selection, in order to use them for crossover.
Cross over is then used, by splitting the bit strings into two substrings in a random position chosen between 1 and the number of genes - 1.
Mutation is then used by choosing a random real number between 0 and 1, if the random float is greater than the mutation rate, a random bit in the bitstring is flipped.

## Requirements
This project runs on python 2.7, and needs Numpy and matplotlib libraries to be installed.

## Details about the implementation
The script calculates the best fitness in each generation of the population. The fitness could go from 0 to 100% depending on the problem. The higher value of the fitness, the more we're actually gaining from the GA.

## Installation

The script accepts exactly 4 arguments. First, the number of individuals in each population, then the number of genes of the individuals/bitstrings. The third argument is the maximum number of generations for which we'll run the GA, and last the mutation rate.

To run the GA, just use the following command:
```
$ ./max_value.py <n_individuals> <n_genes> <max_generations> <mutation_rate>
```

The result shows the best fitness reached, and in which generation. The script then shows the graph of the minimum, maximum and average fitness over the generations.


## Author
Manal Jazouli,
Email: <mjazouli@uoguelph.ca>
