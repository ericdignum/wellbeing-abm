# Stolen from the authors of the SDA paper:
# Paper: https://arxiv.org/abs/1907.07055
# GitHub: https://github.com/sztal/sda-model/tree/master

"""Simple network models and related utilities."""
import numpy as np
from scipy.optimize import toms748
from numpy.random import random, uniform
from random import uniform as _uniform, choice as _choice


# @njit
def _rn_undirected_nb(X, p):
    for i in range(X.shape[0]):
        for j in range(i):
            if random() <= p:
                X[i, j] = X[j, i] = 1
    return X

def random_network(N, p=None, k=None, directed=False):
    """Generate a random network.

    Parameters
    ----------
    N : int
        Number of nodes.
    p : float
        Edge formation probability.
        Should be set to ``None`` if `k` is used.
    k : float
        Average node degree.
        Should be set to ``None`` if `p` is used.
    directed : bool
        Should network be directed.

    Notes
    -----
    `p` or `k` (but not both) must be not ``None``.

    Returns
    -------
    (N, N) array_like
        Adjacency matrix of a graph.
    """
    if p is None and k is None:
        raise TypeError("Either 'p' or 'k' must be used")
    if p is not None and k is not None:
        raise TypeError("'p' and 'k' can not be used at the same time")
    if k is not None:
        if k > N-1:
            raise ValueError(f"average degree of {k:.4} can not be attained with {N} nodes")
        p = k / (N-1)
    if directed:
        X = np.where(uniform(0, 1, (N, N)) <= p, 1, 0)
        np.fill_diagonal(X, 0)
    else:
        X = np.zeros((N, N), dtype=int)
        X = _rn_undirected_nb(X, p)
    return X

# @njit
def _am_undirected_nb(P, A):
    for i in range(A.shape[0]):
        for j in range(i):
            if random() <= P[i, j]:
                A[i, j] = A[j, i] = 1
    return A

def make_adjacency_matrix(P, directed=False):
    """Generate adjacency matrix from edge formation probabilities.

    Parameters
    ----------
    P : (N, N) array_like
        Edge formation probability matrix.
    directed : bool
        Should network be directed.
    """
    # pylint: disable=no-member
    if directed:
        A = np.where(uniform(0, 1, P.shape) <= P, 1, 0)
        A = A.astype(int)
        np.fill_diagonal(A, 0)
    else:
        A = np.zeros_like(P, dtype=int)
        A = _am_undirected_nb(P, A)
    return A

def get_edgelist(A, directed=False):
    """Get ordered edgelist from an adjacency matrix.

    Parameters
    ----------
    A : (N, N) array_like
        An adjacency matrix.
    directed : bool
        Is the graph directed.
    """
    E = np.argwhere(A)
    E = E[E.sum(axis=1).argsort()]
    sum_idx = E.sum(axis=1)
    max_idx = E.max(axis=1)
    max1c_idx = (E[:, 0] > E[:, 1])
    E = E[np.lexsort((max1c_idx, sum_idx, max_idx))]
    if directed:
        dual = np.full((E.shape[0], 1), -1)
    else:
        dual = np.arange(E.shape[0])
        dual[::2] += 1
        dual[1::2] -= 1
        dual = dual.reshape(E.shape[0], 1)
    E = np.hstack((E, dual))
    return E

def rewire_edges(A, p=0.01, directed=False, copy=False):
    """Randomly rewire edges in an adjacency matrix with given probability.

    Parameters
    ----------
    A : (N, N) array_like
        An adjacency matrix.
    p : float
        Rewiring probability.
    directed : bool
        Is the graph directed.
    copy : bool
        Should copy of the adjacency array be returned.
    """
    if copy:
        A = A.copy()
    E = get_edgelist(A, directed=directed)
    loop = range(0, E.shape[0]) if directed else range(0, E.shape[0], 2)
    for u in loop:
        rand = _uniform(0, 1)
        if rand <= p:
            i, j = E[u, :2]
            if not directed and rand <= p/2:
                new_i = j
            else:
                new_i = i
            idx = np.nonzero(np.where(A[new_i, :] == 0, 1, 0))[0]
            idx = idx[idx != new_i]
            if idx.size == 0:
                continue
            new_j = _choice(idx)
            A[i, j] = 0
            A[new_i, new_j] = 1
            if not directed:
                A[j, i] = 0
                A[new_j, new_i] = 1
    return A


class SDA:
    """Social distance attachment network model.

    Attributes
    ----------
    P : (N, N) array_like
        Edge formation probability matrix.
    k : float
        Expected average node degree.
    b : float
        Characteristic length scale (positive value).
        This is the distance value at which edge formation
        probability equals ``1``.
    alpha : float
        Homophily value. It determines rate at which edge formation
        probability decreases/increases with the distance from the characteristic
        length scale `b`. Usually it should be set to a positive value.
    p_rewire : float
        Defualt probability of random edge rewiring when generating
        adjacency matrices. This is used to ensure small worldedness.
    directed : bool
        Should directed networks be generated by default.
    degseq : (N,) array_like, optional
        Default node degree sequence used in the configuration model.
    """
    def __init__(self, P, k, b, alpha, p_rewire=0.01, directed=False):
        """Initialization method."""
        self.P = P
        self.k = k
        self.b = b
        self.alpha = alpha
        self.p_rewire = p_rewire
        self.directed = directed
        self.degseq = None

    def __repr__(self):
        nm = self.__class__.__name__
        D = 'D' if self.directed else 'U'
        a = self.alpha
        p = self.p_rewire
        return f"<{nm} {D}{self.N} k={self.k} b={self.b:.2} a={a} p={p}>"

    @property
    def N(self):
        return self.P.shape[0]

    def sort_degseq(self, degseq):
        """Sort degseq accordingly to centrality of nodes in the social space."""
        w = self.P.sum(axis=1)
        return degseq[np.lexsort((w, w.argsort()[::-1]))]

    def set_degseq(self, degseq, sort=True):
        """Set default degree sequence.

        Parameters
        ----------
        sort : bool
            Should degree sequence be sorted to align with highest
            average edge formation probabilities per node.
        """
        degseq = degseq.copy()
        if sort and degseq is not None:
            degseq = self.sort_degseq(degseq)
        self.degseq = degseq

    @staticmethod
    def prob_measure(D, b, alpha):
        """Compute *SDA model* probability measure.

        Parameters
        ----------
        D : (N, N) array_like
            Distance matrix.
        b : float
            Characteristic length scale (positive value).
        alpha :
            Homophily value.
        """
        if b == 0:
            return np.zeros_like(D, dtype=float)
        P = 1 / (1 + (D/b)**alpha)
        np.fill_diagonal(P, 0)
        return P

    @classmethod
    def optim_b(cls, D, k, alpha, b_optim_min=0, b_optim_max=None,
                xtol=10**-12, **kwds):
        """Find optimal value of the characteristic length scale `b`.

        Parameters
        ----------
        D : (N, N) array_like
            Distance matrix.
        b : float
            Characteristic length scale (positive value).
        alpha :
            Homophily value.
        b_optim_min : float
            Lower bound of the interval used in optimizing `b`.
        b_optim_max : float, optional
            Uppe bound of the interval used in optimizing `b`.
            If ``None`` then defaults to ``D.max()``.
        **kwds :
            Keyword parameters passed to :py:func:`scipy.optimize.brentq`.
        """
        def b_optim(b):
            P = cls.prob_measure(D, b, alpha)
            Ek = P.sum(axis=1).mean()
            return k - Ek

        if b_optim_max is None:
            b_optim_max = D.max()
        b = toms748(b_optim, a=b_optim_min, b=b_optim_max, xtol=xtol, **kwds)
        return b

    @classmethod
    def from_dist_matrix(cls, D, k, alpha, p_rewire=0.01, directed=False, **kwds):
        """Constructor method based on a distance matrix.

        Parameters
        ----------
        D : (N, N) array_like
            Distance matrix.
        **kwds :
            Keyword parameters passed to `optim_b`.
        """
        b = cls.optim_b(D, k, alpha, **kwds)
        P = cls.prob_measure(D, b, alpha)
        return cls(P, k, b, alpha, p_rewire=p_rewire, directed=directed)

    @classmethod
    def from_weighted_dist_matrices(cls, k, alpha, dm, weights=None, p_rewire=0.01,
                                    directed=False, **kwds):
        """Constructor method based on a weighted sequence of distance matrices.

        Parameters
        ----------
        dm : (m,) sequence of (N, N) array_like
            Sequence of distance matrices.
        weights : (m,) array_like, optional
            Weights used for combining arrays.
            If ``None`` then defaults to equal weights.
        **kwds :
            Keyword parameters passed to `optim_b`.
        """
        P = None
        sum_w = 0
        loop = product(dm, (1,)) if weights is None else zip(dm, weights)
        for D, w in loop:
            b = cls.optim_b(D, k, alpha, **kwds)
            _P = cls.prob_measure(D, b, alpha)
            if P is None:
                P = _P*w
            else:
                P += _P*w
            sum_w += w
        P = P / sum_w
        return cls(P, k, b, alpha, p_rewire=p_rewire, directed=directed)

    def adjacency_matrix(self, sparse=True, p_rewire=None, directed=None):
        """Generate an adjacency matrix.

        Parameters
        ----------
        sparse : bool
            Should sparse matrix be used.
            If ``True`` then :py:class:`scipy.sparse.csr_matrix`
            is used.
        p_rewire : float, optional
            Random edge rewiring probability.
            If ``None`` then defaults to the class attribute value.
        directed : bool or None
            Should directed graph be generated.
            If ``None`` then class attribute is used.
        """
        if p_rewire is None:
            p_rewire = self.p_rewire
        if directed is None:
            directed = self.directed
        A = make_adjacency_matrix(self.P, directed=directed)
        if p_rewire > 0:
            A = rewire_edges(A, p=p_rewire, directed=directed, copy=False)
        if sparse:
            A = csr_matrix(A)
        return A