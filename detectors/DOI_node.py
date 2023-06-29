class L2HashNode:
    def __init__(self, data, data_size=0, hash_function=None, children={}, children_count={}, index=-1):
        self._data_size = data_size
        self._hash_function = hash_function
        self._children = children
        self._children_count = children_count
        self._data = data
        self._index = index


    def display(self):
        print(self)


    def __str__(self):
        return "("+str(self._data_size)+", "+str(self._hash_function)+", "+str(self._children_count)+")"


    def get_data_size(self):
        return self._data_size 


    def get_hash_function(self):
        return self._hash_function


    def get_children(self):
        return self._children


    def get_children_count(self):
        return self._children_count

    def get_data(self):
        return self._data

    def get_index(self):
        return self._index
