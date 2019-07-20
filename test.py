import numpy as np


def pair_matrices(matrix1, matrix2, pair_mode='extend'):
    if pair_mode == 'inter':
        return inter_matrices(matrix1, matrix2)
    else:
        return extend_matrices(matrix1, matrix2)


def inter_matrices(matrix1, matrix2):
    # TODO: Intersect-mode of matrix pairing
    return matrix1, matrix2


# pads both matrices to be the same size, assuming they're both centered and 2-d
def extend_matrices(matrix1, matrix2):

    # calculate length, width of m1, m2
    dims_m1 = matrix1.shape
    dims_m2 = matrix2.shape
    h1, w1 = dims_m1[1]/2, dims_m1[0]/2
    h2, w2 = dims_m2[1]/2, dims_m2[0]/2

    # calculate horizontal/vertical padding for matrices
    h_pad1 = max(w2 - w1, 0)
    v_pad1 = max(h2 - h1, 0)

    h_pad2 = max(w1 - w2, 0)
    v_pad2 = max(h1 - h2, 0)

    # do the padding
    padded_matrix1 = np.pad(matrix1, ((int(np.ceil(h_pad1)), int(np.trunc(h_pad1))),
                                      (int(np.ceil(v_pad1)), int(np.trunc(v_pad1)))), 'constant')
    padded_matrix2 = np.pad(matrix2, ((int(np.ceil(h_pad2)), int(np.trunc(h_pad2))),
                                      (int(np.ceil(v_pad2)), int(np.trunc(v_pad2)))), 'constant')
    return padded_matrix1, padded_matrix2


if __name__ == '__main__':
    # testing extend matrices function
    height_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    width_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    padded_height, padded_width = pair_matrices(height_matrix, width_matrix, pair_mode='extend')
