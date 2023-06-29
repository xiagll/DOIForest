import warnings

import numpy as np

from sklearn.cluster import KMeans

class SimilarityHash():
    ''' The base class of similarity hash functions'''
    def __init__(self):
        pass
        
    # Interfaces
    # get_hash_type(self)
    # fit(self, data)
    # display_hash_func_parameters(self)
    # get_hash_value(self, x)
    # get_number_of_bins(self)

class EuclideanLSH(SimilarityHash):
    def __init__(self, norm=2, bin_width=4):
        super().__init__()
        self._dimensions = -1
        self._norm = norm
        self._bin_width = bin_width
        self._a = None
        self._b = None


    def get_hash_type(self):
        return 'L'+str(self._norm)+'LSH'


    def fit(self, data):
        if np.array(data == None).all():
            return
        self._dimensions = len(data[0])

        if self._norm == 1:
            self._a = np.random.standard_cauchy(self._dimensions)
        elif self._norm == 2:
            #self._a = np.random.normal(0.0, 1.0, self._dimensions)
            self._a = np.random.normal(0.0, 1.0, self._dimensions)

        self._b = np.random.uniform(0.0, self._bin_width)


    def display_hash_func_parameters(self):
        print("a: ", self._a)
        print("b: ", self._b)
        print("bin width: ", self._bin_width)


    def __str__(self):
        return "("+str(self._dimensions)+", "+str(self._norm)+", "+str(self._bin_width)+", "+str(self._b)+")"


    def get_hash_value(self, x):
        return int(np.floor((np.dot(x, self._a)+self._b)/self._bin_width))



class SimpleL2Hash(EuclideanLSH):
    def __init__(self, norm=2, number_of_bin=2):
            super().__init__(norm=norm)
            self._number_of_bin = number_of_bin


    def get_hash_type(self):
            return 'L'+str(self._norm)+'SimpleL2Hash'

    def fit(self, data):
            # Check if data are available.
            if np.array(data == None).all():
                raise ValueError("The data is null!")

            self._dimensions = len(data[0])

            # Check if the data contain the same data instance.
            if len(np.unique(data, axis=0)) == 1:
                raise ValueError("Only one unique sample in the data!")
            
            # We need to handle highly similar data instances, particularly the case 'norm=-1' (i.e., random selection of a dimension)
            projected_range = 0.0
            while projected_range == 0.0:
                if self._norm == 1:
                    self._a = np.random.standard_cauchy(self._dimensions)
                elif self._norm == 2:
                    self._a = np.random.normal(0.0, 1.0, self._dimensions)
                elif self._norm == -1: # The norm used in IForest
                    self._a = np.zeros(self._dimensions)
                    self._a[np.random.randint(0, self._dimensions)] = 1

                projections = np.dot(data, self._a)
                projected_range = np.ptp(projections)
                if projected_range == 0.0:
                    warnings.warn("The projected range is Zero! Try another round.")

            self._bin_width = projected_range/(self._number_of_bin-1) # Note that the denominator is number_of_bin-1
            self._b = np.random.uniform(0.0, self._bin_width)

    def __str__(self):
        return "("+str(self._dimensions)+", "+str(self._norm)+", "+str(self._number_of_bin)+", "+str(self._bin_width)+", "+str(self._b)+")"




class KMeansHash(SimilarityHash):
    def __init__(self, number_of_clusters=2):
        super().__init__()
        self._k_means = KMeans(number_of_clusters)

    def get_hash_type(self):
        return "KMeans_Hash"


    def fit(self, data):
        # Check if there are enough data for KMeans clustering 
        data_size_unique = len(np.unique(data, axis=0))
        if data_size_unique < self._k_means.n_clusters:
            self._k_means = KMeans(data_size_unique)

        self._k_means.fit(data)


    def get_hash_value(self, x):
        X=[]
        X.append(x)
        return self._k_means.predict(np.array(X))[0]


    def __str__(self):
        return "("+str(self._k_means.inertia_)+")"

