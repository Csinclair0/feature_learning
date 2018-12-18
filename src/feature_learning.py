import numpy as np
from sklearn.metrics import pairwise_distances
import math
import warnings
warnings.filterwarnings('ignore')


def weighted_euclidean(X, V, weights):
    """Weighted euclidean distance function

    Parameters
    ----------
    X : array
        first object
    V : array
        second object
    weights : array
        feature weights

    Returns
    -------
    float
        weighted distance

    """
    dists = X- V
    return np.sqrt(np.sum((dists * weights) **2))


def single_delta(X, V, F):
    """Distance using one single parameter

    Parameters
    ----------
    X : type
        Description of parameter `X`.
    V : type
        Description of parameter `V`.
    F : type
        Description of parameter `F`.

    Returns
    -------
    type
        Description of returned object.

    """
    d = X[F] - V[F]
    return d


def calc_beta(X, d):
    """calculate beta calue for feature weight learning

    Parameters
    ----------
    X : array
        data set
    d : array
        distance matrix

    Returns
    -------
    float
        beta value

    """
    n = X.shape[0]
    for b in np.linspace(0,1,10000):
        p = 1/(1+b*d)
        p = np.triu(p, 1)
        if (2 / (n*(n-1))) *np.sum(p)< .5:
            return b


def return_weights(X, b, d, mincols, threshold, learning_rate, max_iter):
    """returns learned feature weights, given the data set, beta, the distance matrix and the minimum number of columns

    Parameters
    ----------
    X : array
        data set
    b : float
        beta value
    d : array
        distance atrix
    mincols : int
        minimum number of columns to return that have weights
    threshold : float
        minimum threshold when to stop learning
    n : int
        learning rate

    Returns
    -------
    array
        learned feature weights

    """

    w= np.empty((1,X.shape[1]))
    w.fill(1)
    p_1 = 1/(1+b*d)
    n = X.shape[0]
    E_old = 1
    for i in np.arange(0, max_iter):
        d = pairwise_distances(X,X, metric = weighted_euclidean, **{'weights':w})
        grad_w = np.empty((1,X.shape[1]))
        part_pq = -b/((1+b*d)**2)
        p = 1/(1+b*d)
        E = (2/(n*(n-1))) * np.sum(np.triu(.5*((p*(1-p_1) + p_1*(1-p))), 1))
        if E_old - E < threshold:
            break
        E_old = E
        part_eq = (1-2*p_1)
        w_valid = np.where(w > 0)[1]

        if w_valid.shape[0] == mincols:
            break

        for j in w_valid:
            d_w = pairwise_distances(X, X, metric = single_delta, **{'F':j})
            part_w = w[0, j]*(d_w)**2 / d
            part_w = np.triu(part_w, 1)
            grad_w_j = 1/(n*(n-1)) * part_eq * part_pq * part_w
            grad_w_j = np.triu(grad_w_j, 1)
            grad_w[ 0, j] = np.nansum(grad_w_j)
        grad_w = grad_w * learning_rate
        w = w-grad_w
        w = w.clip(min=0)
        #if i %100 == 0: #and i > 0:
            #print("Iteration {} Finished".format(i))
            #print("Weights : {} ".format(w))
            #print("Function Improvement : {}".format(E))

    wmax = np.max(w)
    w = w / wmax
    print("{} Iterations Required".format(i))
    return w

def update(X, u2, m, weights, n, c):
    """update the fuzzy c-means process

    Parameters
    ----------
    X : array
        data being clustered
    u2 : array
        current fuzziness matrix
    m : float
        fuzzy factor
    weights : array
        distance metrics
    n : int
        sample size
    c : int
        numebr of cluster

    Returns
    -------
    4 arrays used in c-means

    """
    u = u2.copy()
    um = u ** m
    #update cluster centeers matrix
    numerator = um.T.dot(X)
    denominator = um.T.sum(axis = 1)
    V = numerator.T/(denominator)
    V = V.T

    # update distance matrix
    d = pairwise_distances(X, V, metric = weighted_euclidean, **{'weights':weights})
    #update fuzziness
    for i in np.arange(0, n):
        for j in np.arange(0, c):
            newdenom = (d[i, j] / d[i, :]) ** (2/(m-1))
            u[i, j] = 1 / np.sum(newdenom, axis = 0)

    #update loss
    J = (u *d**2).sum()

    return V, u, J, d

def return_weighted_distance(X,  mincols = 0, sample_size = 1, threshold = .0005, n = 1, max_iter = 1000):
    """takes in data set and completes entire process of feature weight learning.
    1. Calculate Pairwise Distances
    2. Calculate Beta
    3. Learn Feature weights through gradient descent

    Parameters
    ----------
    sku : class transformation
        class transformation where to add feature weights
    X : array
        data set
    mincols : int
        minimum number of columns to returdn
    sample_size :  float
        fraction of dataset to use in feature weight learning

    Returns
    -------
    array
        weighted dataset

    """
    numsample = math.ceil(sample_size * X.shape[0])
    sample = np.random.choice(X.shape[0],numsample, replace = False )
    X_S = X#X[sample]
    d = pairwise_distances(X_S, X_S, metric = 'euclidean')
    b = calc_beta(X_S, d)
    w = return_weights(X_S, b, d, mincols, threshold, n, max_iter)
    w = w.reshape(1,-1)
    return w


class c_means():
    """ fuzzy cmeans class

    Parameters
    ----------
    c : int
        number of clusters
    m : float
        fuzzification index
    max_iter : int
        maximum iterations
    threshold : float
        threhold of improvement

    Attributes
    ----------
    c_ : c
    max_iter
    threshold
    m

    """
    def __init__(self, c = 3, m = 2,  max_iter = 1000, threshold = .01):
        self.c_ = c
        self.max_iter = max_iter
        self.threshold = threshold
        self.m = m

    def fit(self, X, weights):
        """ performs clustering using given weights

        Parameters
        ----------
        X : array
            data to be clustered
        weights : array
            distance weight matrix

        Returns
        -------
        class
            attributes of clustering

        """
        self.weights = weights
        c = self.c_
        m = self.m
        self.u_ = np.empty((X.shape[0], c))
        d = X.shape[1]
        n = X.shape[0]
        V = np.random.random((c, d))
        u_0 = np.random.random((n, c))
        J = np.zeros(0)
        num_iter = 0
        u = u_0
        max_iter = self.max_iter
        threshold = self.threshold
        while num_iter < max_iter - 1:
            u2 = u.copy()
            V, u, Jm, d = update(X, u2, m, weights, n, c)
            J = np.hstack((J, Jm))
            num_iter += 1

            if np.linalg.norm(u - u2) < threshold:
                break

        self.error_improvement = np.linalg.norm(u - u_0)
        self.f_p_coeff = np.trace(u.dot(u.T)) / float(n)
        self.cluster_centers = V
        self.fuzzy_partition = u
        self.loss = J
