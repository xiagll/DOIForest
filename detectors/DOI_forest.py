import math
import numpy as np
import heapq
import copy as cp


from .DOI_tree import L2HashTree
from detectors import VSSampling

class L2HashForest:
    def __init__(self, num_trees, sampler, l2h_family):
        self._num_trees = num_trees
        self._sampler = sampler
        self._l2h_family = l2h_family
        self._trees = []

    def fit(self, data):
        # Important: clean the tree array
        self._trees = []

        # Use the first row to index the data, mainly for sampling
        indices = range(len(data))
        indexed_data = np.c_[indices, data]
        # Sampling data
        self._sampler.fit(indexed_data)
        sampled_indexed_datas = self._sampler.draw_samples(indexed_data)
        # Transform back the data
        sampled_datas = []
        for sampled_indexed_data in sampled_indexed_datas:
            sampled_datas.append(np.array(sampled_indexed_data)[:, 1:])

        # Build Learning to Hash (L2Hash) trees
        for i in range(self._num_trees):
            sampled_data = sampled_datas[i]
            tree = L2HashTree(self._l2h_family)
            tree.fit(sampled_data)
            self._trees.append(tree)

    def decision_function(self, data):
        scores = []
        data_size = len(data)
        for i in range(data_size):
            d_scores = []
            for j in range(self._num_trees):
                transformed_data = data[i]
                if self._sampler._bagging is not None:
                    transformed_data = self._sampler._bagging_instances[j].get_transformed_data(np.mat(data[i])).A1
                d_scores.append(self._trees[j].decision_function(transformed_data))
            scores.append(d_scores)

        avg_scores = []
        for i in range(data_size):
            score_avg = 0.0
            for j in range(self._num_trees):
                score_avg += scores[i][j]
            score_avg /= self._num_trees
            avg_scores.append(score_avg)

        return np.array(avg_scores)

    def get_avg_branch_factor(self):
        sum = 0.0
        for t in self._trees:
            sum += t.get_avg_branch_factor()
            #print(t.get_area_para())
        return sum / self._num_trees

    def get_avg_isolation_efficiency(self):
        sum = 0.0
        for t in self._trees:
            sum += t.get_isolation_effi()
        return sum / self._num_trees


class DOIForest(L2HashForest):
    def __init__(self, num_trees, sampler, l2h_family):
        super().__init__(
            num_trees=num_trees,
            sampler=sampler,
            l2h_family=l2h_family,
        )
        self._trees = []

    def gen2_algorithm(self, data, num_trees, change_rate=0.5, MUTATION_RATE=0.5):
        trees = self._trees

        # First mutate: each tree will mutate to many (e.g. 20) trees and select the best one
        selected_trees = []
        for i in range(num_trees):
            new_trees = self.mutate(trees[i], num_trees, change_rate, MUTATION_RATE)
            new_trees.append(trees[i])
            abfs = []
            for t in new_trees:
                abfs.append(self.F(t))
            new_trees = np.array(new_trees)
            selected_tree = self.select(new_trees, abfs, 1)
            selected_trees = selected_trees + selected_tree

        # Second mutate: use new samples (10) to generate new trees, and select the best 100 trees from the original trees and new trees.
        build_trees = self.build_newtrees(num_trees, change_rate, data)
        selected_trees = selected_trees + build_trees
        fitness = []
        for t in selected_trees:
            fitness.append(self.F(t))
        selected_trees = np.array(selected_trees)
        # tree selection
        self._trees = self.select(selected_trees, fitness, num_trees)
        #print(777, self._trees)


    ## objective function for getting compaired values
    def F(self, x):
        return x.get_isolation_effi()

    ## no use now (but for preparation)
    def get_fitness(self, pred):
        pred = np.asarray(pred)
        return np.abs(pred-math.e)

    ## select the objective trees
    def select(self, pop, fitness, POP_SIZE):    # nature selection wrt pop's fitness
        max_number = heapq.nlargest(POP_SIZE, fitness)   # select the max POP_SIZE number items from fitness
        max_index = map(fitness.index, max_number)
        idx = list(max_index)
        # idx = []
        # for i in min_number:
        #     index = fitness.index(i)
        #     idx.append(index)
        #     fitness[index] = 0
        return pop[idx].tolist()

    def mutate(self, tree, num_trees, change_rate, MUTATION_RATE):
        new_trees = []

        for i in range(num_trees):
            if np.random.rand() < change_rate:
                new_tree = cp.deepcopy(tree)
                new_tree.mutate_tree(MUTATION_RATE)
                if tree.get_isolation_area()!=new_tree.get_isolation_area():
                    # print(111111111, tree.get_isolation_area(),new_tree.get_isolation_area(), tree.get_isolation_effi(),new_tree.get_isolation_effi())
                    new_trees.append(new_tree)
        return new_trees

    def build_newtrees(self, num_trees, change_rate, data):
        new_trees = []
        itera=int(num_trees*change_rate)

        indices = range(len(data))
        indexed_data = np.c_[indices, data]
        # Sampling data
        sampler = VSSampling(itera)
        sampler.fit(indexed_data)
        sampled_indexed_datas = sampler.draw_samples(indexed_data)
        # Transform back the data
        sampled_datas = []
        for sampled_indexed_data in sampled_indexed_datas:
            sampled_datas.append(np.array(sampled_indexed_data)[:, 1:])
        # Build Learning to Hash (L2Hash) trees
        for i in range(itera):
            sampled_data = sampled_datas[i]
            tree = L2HashTree(self._l2h_family)
            tree.fit(sampled_data)
            new_trees.append(tree)

        return new_trees