from scipy.sparse import coo_matrix


def matrix(D):
    items = list(D.items())  # list only needed for python3
    d = [v[0] for (i, j), v in items]
    ii = [i for (i, j), v in items]
    jj = [j for (i, j), v in items]
    D_matrix = coo_matrix((d, (ii, jj))).toarray()
    return D_matrix
