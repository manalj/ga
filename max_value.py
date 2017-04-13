#!/usr/bin/python

from operator import attrgetter
import numpy as np
import copy
import sys

dtype = 'float32'


class Population(object):
    """docstring for Population."""
    def __init__(self, mutation_rate, n_chromosomes, n_genes, random=True):
        super(Population, self).__init__()
        self.n_genes = n_genes
        self.mutation_rate = mutation_rate
        if(random):
            self.individuals = [Chromosome(n_genes) for x in range(n_chromosomes)]
        else:
            self.individuals = n_chromosomes ## If it's not the first generation, pass chromosomes as an array

    ## Creates new generation, children
    def evolve(self):
        children = []
        n_chromosomes = len(self.individuals)
        ## Loop until new children are created
        while len(children)<n_chromosomes:
            parentA = self.selection()
            parentB = self.selection()
            childA, childB = self.cross_over(parentA, parentB)
            children.append(childA.mutate(self.n_genes, self.mutation_rate))
            children.append(childB.mutate(self.n_genes, self.mutation_rate))
        return Population(self.mutation_rate, children,self.n_genes, random=False)

    ## Selects a parent using accumulated normalized fitness
    def selection(self):
        self.individuals.sort(key=lambda c: c.fitness)   # sort by fitness
        ## Calculate accumulated fitness, and pick random individual
        max     = sum([c.fitness for c in self.individuals])
        pick    = np.random.uniform(0, max)
        current = 0
        for chromosome in self.individuals:
            current += chromosome.fitness
            if current > pick:
                return chromosome

    ## Cross over between two chromosomes
    def cross_over(self, A, B):
        split_index = np.random.randint(1, self.n_genes-1)
        childA_genes  = np.concatenate([A.genes[:split_index], B.genes[split_index:]])
        childB_genes  = np.concatenate([B.genes[:split_index], A.genes[split_index:]])
        return Chromosome(childA_genes, False), Chromosome(childB_genes, False)

    ## returns best fitness in a population
    def return_bf(self):
        return max(c.fitness for c in self.individuals)

    ## returns worst fitness in a population
    def return_wf(self):
        return min(c.fitness for c in self.individuals)
        # return min(self.individuals, key=attrgetter('fitness'))

    ## returns average fitness in a population
    def return_mf(self):
        return sum(c.fitness for c in self.individuals) / float(len(self.individuals))


class Chromosome(object):
    """docstring for Chromosome."""
    def __init__(self, length, random=True):
        super(Chromosome, self).__init__()
        if(random):
            self.genes = np.random.randint(2, size=length)
        else:
            self.genes = length ## In this case length holds the actual values of the genes

        self.fitness = self.calc_fitness()

    ## Calculates current element fitness - integer value from the bitstring
    def calc_fitness(self):
        tot = 0
        for bit in np.ones(len(self.genes), dtype='int8'):
            tot = (tot << 1) | bit

        out = 0
        for bit in self.genes:
            out = (out << 1) | bit

        return (out/float(tot))*100

    ## Mutate the individual
    def mutate(self, n_genes, mutation_rate):
        pick = np.random.uniform(0, 1)
        if(pick<mutation_rate):
            bit_to_flip = np.random.randint(0, n_genes)
            mutant = copy.copy(self)
            mutant.genes[bit_to_flip] = 1-mutant.genes[bit_to_flip]
            return mutant
        return self


#def main(mutation, n_indivs, n_genes, max_pop):
if __name__ == '__main__':

    if len(sys.argv) != 5:
        print "Not enough variables"
        print "Usage: %s [n_individuals] [n_genes] [maximum_n_population] [mutation_rate]" % sys.argv[0]
        print "Where the first parameter must be interpretable as a string, and the rest as integers."
        sys.exit(-1)

    n_indivs = int(sys.argv[1])
    n_genes = int(sys.argv[2])
    max_pop = int(sys.argv[3])
    mutation = float(sys.argv[4])

    print "## Parameters ##"
    print "Number of individuals: %d"%n_indivs
    print "Mutation rate: %.3f"%mutation
    print "Length of bitstring: %.d"%n_genes
    print "Maximum number of generations: %d\n"%max_pop

    P = []
    generation = 0
    best_fit = 0
    min_fit = np.zeros(max_pop, dtype=dtype)
    max_fit = np.zeros(max_pop, dtype=dtype)
    avg_fit = np.zeros(max_pop, dtype=dtype)
    ## Create random population
    P.append(Population(mutation, n_indivs, n_genes)) ## number of individuals, number of genes for each individual (100,256)

    while generation < max_pop and best_fit != 100:
        ## Evolve the population, and return best fitness of children
        P.append(P[generation].evolve())
        fit_indiv = P[generation].return_bf()

        min_fit[generation] = P[generation].return_wf()
        max_fit[generation] = P[generation].return_bf()
        avg_fit[generation] = P[generation].return_mf()

        if best_fit < fit_indiv:
            best_fit = fit_indiv
            best_fit_gen = generation
        generation += 1

    print "Best fitness reached: %.2f %% in generation: %d \n"%(best_fit, best_fit_gen)
    return [min_fit, max_fit, avg_fit]

    ## Plot graphs
    plt.figure()
    plt.title('Mutation rate: %.3f Size of population: %d'%(mutation,n_indivs))
    plt.plot(min_fitness, '-r', label="Average over runs of minimum fitness")
    plt.plot(max_fitness, '-b', label="Average over runs of maximum fitness")
    plt.plot(avg_fitness, '-g', label="Average over runs of average fitness")
    plt.xlabel("Generations")
    plt.ylabel("Average fitness")
    plt.legend(loc='lower right')
    plt.show()
