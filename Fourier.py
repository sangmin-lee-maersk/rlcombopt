# -*- coding: utf-8 -*-

#%%
import itertools
import numpy as np

#%%
class TrivialBasis(object):
    """Uses the features themselves as a basis. However, does a little bit of basic manipulation
    to make things more reasonable. Specifically, this allows (defaults to) rescaling to be in the
    range [-1, +1].
    """

    def __init__(self, nvars, ranges):
        self.numTerms = nvars
        self.ranges = np.array(ranges)

    def scale(self, value, pos):
        if self.ranges[pos,0] == self.ranges[pos,1]:
            return 0.0
        else:
            return (value - self.ranges[pos,0]) / (self.ranges[pos,1] - self.ranges[pos,0])

    def getNumBasisFunctions(self):
        return self.numTerms

    def computeFeatures(self, features):
        if len(features) == 0:
            return np.ones((1,))
        return (np.array([self.scale(features[i],i) for i in range(len(features))]) - 0.5)*2.

#%%

class FourierBasis(TrivialBasis):
    """Fourier Basis linear function approximation. Requires the ranges for each dimension, and is thus able to
    use only sine or cosine (and uses cosine). So, this has half the coefficients that a full Fourier approximation
    would use.
    From the paper:
    G.D. Konidaris, S. Osentoski and P.S. Thomas.
    Value Function Approximation in Reinforcement Learning using the Fourier Basis.
    In Proceedings of the Twenty-Fifth Conference on Artificial Intelligence, pages 380-385, August 2011.
    """

    def __init__(self, nvars, ranges, order=3):
        nterms = pow(order + 1.0, nvars)
        self.numTerms = nterms
        self.order = order
        self.ranges = np.array(ranges)
        iter = itertools.product(range(order+1), repeat=nvars)
        self.multipliers = np.array([list(map(int,x)) for x in iter])

    def computeFeatures(self, features):
#        if len(features) == 0:
#            return np.ones((1,))
        basisFeatures = np.array([self.scale(features[i],i) for i in range(len(features))])
        return np.cos(np.pi * np.dot(self.multipliers, basisFeatures))