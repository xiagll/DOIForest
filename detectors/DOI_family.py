from .l2h import *

class SimilarityHashFamily():
    ''' The base class of similarity hash families'''
    def __init__(self):
        pass

    # Interfaces
    # get_hash_function(self)



class EuclideanLSHFamily():
    def __init__(self, norm=2, bin_width=4):
        self._norm = norm
        self._bin_width = bin_width


    def get_hash_function(self):
        return EuclideanLSH(norm=self._norm, bin_width=self._bin_width)



class SimpleL2HashFamily():
    def __init__(self, norm=2, number_of_bin=0):
        self._norm = norm
        self._number_of_bin = number_of_bin


    def get_hash_function(self):
        num_of_bin = self._number_of_bin

        # The expectation of the following random number of bins will be e
        if(num_of_bin == 0):
            probability_sum = 0.0
            while probability_sum <= 1:
                probability_sum += np.random.rand()
                num_of_bin += 1

        return SimpleL2Hash(norm=self._norm, number_of_bin=num_of_bin)



class KMeansHashFamily():
    def __init__(self, number_of_bin=0):
        self._number_of_bin = number_of_bin


    def get_hash_function(self):
        num_of_bin = self._number_of_bin

        # The expectation of the following random number of bins will be e
        if(num_of_bin == 0):
            probability_sum = 0.0
            while probability_sum <= 1:
                probability_sum += np.random.rand()
                num_of_bin += 1

        return KMeansHash(number_of_clusters=num_of_bin)
