import numpy as np
from scipy.optimize import linear_sum_assignment


def best_map(l1, l2):
    """
    Permute labels of L2 to match L1 as good as possible.
    
    :param l1: numpy array
    :param l2: numpy array
    :return new_l2: numpy array
    """
    l1 = l1.ravel()
    l2 = l2.ravel()
    # Check if the size of l1 and l2 are the same
    if len(l1) != len(l2):
        raise ValueError('size(l1) must == size(l2)')

    label1, inv1 = np.unique(l1, return_inverse=True)
    label2, inv2 = np.unique(l2, return_inverse=True)

    n_class1 = len(label1)
    n_class2 = len(label2)

    g_mat = np.zeros((n_class1, n_class2))
    np.add.at(g_mat, (inv1, inv2), 1)

    _, col_ind = linear_sum_assignment(-g_mat.T)

    new_l2 = label1[col_ind[inv2]]
    return new_l2