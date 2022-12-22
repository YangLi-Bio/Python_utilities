#################################################################################
#                                                                               #
#                     Basic utility mathematical functions                      #
#                                                                               #
#################################################################################


# Function list: 
# 1. intersect : calculate the intersection between two lists
# 2. union : calculate the union of multiple lists
# 3. concatenate_lists : concatenate two lists by retaining repetitions
# 4. concatenate_sort_lists : concatenate two lists by retaining repetitions and 
#    rearranging the order
# 5. get_indices : get the indices of elements in a list against another one


script_dir = "/fs/ess/PCON0022/liyang/Python_utilities/Functions/"



#################################################################################
#                                                                               #
#        1. intersect : calculate the intersection between two lists            #
#                                                                               #
#################################################################################


# Input : 
# 1. lst1 : the first list
# 2. lst2 : the second list


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))



#################################################################################
#                                                                               #
#                 2. union : calculate the union of multiple lists              #
#                                                                               #
#################################################################################


# Input : 
# 1. lst1 : the first list
# 2. lst2 : the second list


def union(lst1, lst2, ...):
    return set.union(lst1, lst2, ...)



#################################################################################
#                                                                               #
#     3. concatenate_lists : concatenate two lists by retaining repetitions     #
#                                                                               #
#################################################################################


# Input : 
# 1. lst1 : the first list
# 2. lst2 : the second list


def concatenate_lists(lst1, lst2):
    final_list = lst1 + lst2
    return final_list



#################################################################################
#                                                                               #
# 4. concatenate_sort_lists : concatenate two lists by retaining repetitions    #
#    and rearranging the order                                                  #
#                                                                               #
#################################################################################


# Input : 
# 1. lst1 : the first list
# 2. lst2 : the second list


def concatenate_sort_lists(lst1, lst2):
    final_list = sorted(lst1 + lst2)
    return final_list



#################################################################################
#                                                                               #
# 5. get_indices : get the indices of elements in a list against another one    #
#                                                                               #
#################################################################################


# Input : 
# 1. lst1 : the first list
# 2. lst2 : the second list

def get_indices(lst1, lst2):
    
    out_lst = []
    for i in lst2:
        out_lst.append(lst1.index(i))
    
    
    return(out_lst)
