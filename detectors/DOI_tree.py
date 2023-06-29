import numpy as np

from .DOI_node import *

class L2HashTree:
    def __init__(self, l2h_family=None, depth_limit=-1):
        self._l2h_family = l2h_family
        self._depth_limit = depth_limit
        self._root = None
        self._n_samples = 0 
        self._avg_branch_factor = 0
        self._reference_path_length = 0

        self._br = 0  ## new adding
        self._br_count = 0
        self._dp = 0
        self._dp_count = 0


    def fit(self, data):
        self._n_samples = len(data)
        self._depth_limit = self._depth_limit if self._depth_limit > 0 else np.inf if self._depth_limit == 0 else self.get_binary_random_height(self._n_samples)
        data = np.array(data)
        self._root = self._recursive_fit(data)
        self._avg_branch_factor = self._get_avg_branch_factor()
        self._reference_path_length = self.get_random_path_length_symmetric(self._n_samples)


    def _recursive_fit(self, data, pre_depth=-1, ref_depth = -1):
        n_samples = len(data)
        
        if n_samples == 0:
            return None

        cur_depth = pre_depth + 1
        ref_depth = ref_depth + 1

        if len(np.unique(data, axis=0)) == 1 or cur_depth > self._depth_limit:
            self._dp += len(data)
            self._dp_count += len(data)*ref_depth
            return L2HashNode(data, len(data), None, {}, {}, cur_depth)

        else:
            hash_function = self._l2h_family.get_hash_function()
            hash_function.fit(data)
            partition = self._split_data(data, hash_function)

            while len(partition) == 1 and cur_depth <= self._depth_limit:
                cur_depth += 1
                hash_function = self._l2h_family.get_hash_function()
                hash_function.fit(data)
                partition = self._split_data(data, hash_function)
            if cur_depth > self._depth_limit:
                self._dp += len(data)
                self._dp_count += len(data) * ref_depth
                return L2HashNode(data, len(data), hash_function, {}, {}, cur_depth)

            self._br += 1 ####
            self._br_count += len(partition) ####

            children_count = {}
            for key in partition.keys():
                children_count[key] = len(partition.get(key))

            children = {}
            for key in partition.keys():
                child_data = partition.get(key)
                children[key] = self._recursive_fit(child_data, cur_depth, ref_depth)
            
            return L2HashNode(data, len(data), hash_function, children, children_count, cur_depth)


    def _split_data(self, data, hash_function):
        ''' Split the data using the given hash function '''
        partition = {}
        for i in range(len(data)):
            key = hash_function.get_hash_value(np.array(data[i]))
            if key not in partition:
                partition[key] = [data[i]]
            else:
                sub_data = partition[key]
                sub_data.append(data[i])
                partition[key] = sub_data

        return partition


    def get_num_instances(self):
        return self._n_samples


    def display(self):
        self._recursive_display(self._root)	

    def _recursive_display(self, l2hash_node, leftStr=''):
        if l2hash_node is None:
            return

        print(leftStr+'('+str(len(leftStr))+'):'+str(l2hash_node))
        
        children = l2hash_node.get_children()

        for key in children.keys():
            self._recursive_display(children[key], leftStr+' ')


    def decision_function(self, x):
        path_length = self._recursive_get_search_depth(self._root, 0, x)
        return pow(2.0, -1.0*path_length/self._reference_path_length)


    def _recursive_get_search_depth(self, l2hash_node, cur_depth, x):
        if l2hash_node is None:
            return -1

        children = l2hash_node.get_children()
        if not children:
            real_depth = l2hash_node.get_index()
            adjust_factor = self.get_random_path_length_symmetric(l2hash_node.get_data_size())
            #return cur_depth+adjust_factor
            return cur_depth * np.power(1.0 * real_depth / max(cur_depth, 1.0), 1) + adjust_factor
        else:
            key = l2hash_node.get_hash_function().get_hash_value(x)
            if key in children.keys():
                return self._recursive_get_search_depth(children[key], cur_depth+1, x)
            else:
                cur_depth = cur_depth + 1
                real_depth = l2hash_node.get_index()
                return cur_depth * np.power(1.0 * real_depth / max(cur_depth, 1), 1)
                #return cur_depth+1

    def get_avg_branch_factor(self):
        return self._avg_branch_factor

    # def display_area(self):
    #     print(111,self._avg_branch_factor,self._dp, self._dp_count)
    #     dp, dp_count = self._get_avg_depth(self._root, 0)
    #     print(222,dp, dp_count)


    def _get_avg_branch_factor(self):
        i_count, bf_count = self._recursive_sum_BF(self._root)
        
        # Single node PATRICIA trie
        if i_count == 0:
            return 2.0

        return bf_count*1.0/i_count	


    def _recursive_sum_BF(self, l2hash_node):
        if l2hash_node is None:
            return None, None

        children = l2hash_node.get_children()
        if not children:
            return 0, 0
        else:
            if len(children)==1:    ####  move the node with one child
                i_count, bf_count = 0, 0
                print("childen=1")
            else:
                i_count, bf_count = 1, len(children)
            for key in children.keys():
                i_c, bf_c = self._recursive_sum_BF(children[key])
                i_count += i_c
                bf_count += bf_c
            return i_count, bf_count


    def get_random_path_length_symmetric(self, num_samples):
        if num_samples <= 1:
            return 0
        elif num_samples > 1 and num_samples <= round(self._avg_branch_factor):
            return 1
        else:
            return (np.log(num_samples)+np.log(self._avg_branch_factor-1.0)+0.5772)/np.log(self._avg_branch_factor)-0.5


        # Binary tree has the highest height
    def get_binary_random_height(self, num_samples):
        return 2*np.log2(num_samples)+0.8327



    def get_isolation_area(self):
        if self._dp != 0:
            area = self._avg_branch_factor * (self._dp_count*1.0/self._dp)
        else:
            area = 0
        return area

    def get_isolation_effi(self):
        if self._dp != 0:
            # print(self._avg_branch_factor, self._dp_count*1.0/self._dp)

            area = self._avg_branch_factor * (self._dp_count*1.0/self._dp)
            # print(1111,self._n_samples,area)
            efficiency = self._n_samples / area
            #print(efficiency)
        else:
            efficiency = 0.0
        return efficiency

    def mutate_tree(self, mutate_rate):
        self._root = self.recursive_mutate_tree(self._root, mutate_rate, -1)
        self._avg_branch_factor = self._get_avg_branch_factor()
        self._dp, self._dp_count = self._get_avg_depth(self._root, 0)

    def _get_avg_depth(self, l2hash_node, cur_depth):
        if l2hash_node is None:
            return None, None
        children = l2hash_node.get_children()
        data_size = l2hash_node.get_data_size()
        if not children:
            return data_size, data_size*cur_depth
        else:
            cur_depth += 1
            dp = 0
            dp_count = 0
            for key in children.keys():
                dp0, dp_count0 = self._get_avg_depth(children[key], cur_depth)
                dp += dp0
                dp_count += dp_count0
            return dp, dp_count

    def recursive_mutate_tree(self, lsh_node, mutate_rate, pre_depth):

        if lsh_node is None:
            return None
        children = lsh_node.get_children()
        if not children:
            return lsh_node

        if np.random.rand() < mutate_rate:
            data1 = lsh_node.get_data()
            current_node = self._recursive_fit(data1, pre_depth)
            return current_node
        else:
            cur_depth = pre_depth + 1

            data = lsh_node.get_data()
            data_size = lsh_node.get_data_size()
            hash_function = lsh_node.get_hash_function()
            children_count = lsh_node.get_children_count()

            new_children = {}
            for key in children.keys():
                if np.random.rand() < mutate_rate:
                    data2 = children[key].get_data()
                    temp_node = self._recursive_fit(data2, cur_depth)
                    new_children[key] = temp_node
                else:
                    temp_node = children[key]
                    new_children[key] = self.recursive_mutate_tree(temp_node, mutate_rate, cur_depth)

            return L2HashNode(data, data_size, hash_function, new_children, children_count, cur_depth)
