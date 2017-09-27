"""
class for angular quadrature in 2D geometry
"""
import numpy as np
from math import pi, cos, sin

class AQ(object):
    def __init__(self, sn_ord):
        '''@brief Constructor of aq class

        @param sn_ord Sn angular quadrature order
        '''
        # int for sn order
        self.sn_ord = sn_ord
        # dictionary for containing angular quandrature directions and weights
        self.aq_data = dict()
        # number of directions
        self.n_dir = (sn_ord+2) * sn_ord / 2
        # make aq data
        self._quad2d()

    def _quad2d(self):
        '''@brief Internal function used to calculate aq data

        @param self Reference to the class
        @return aq_data dictionary
        '''
        ct, quad1d, solid_angle = 0, np.polynomial.legendre.leggauss(self.sn_ord), 4*pi
        # loop over relevant polar angles
        for m in range(self.sn_ord/2, self.sn_ord):
            # calculate number of points per level
            n_level = 4 * (self.sn_ord - m)
            delta = 2.0 * pi / n_level
            # get the polar angles
            mu = quad1d[0][m]
            # calculate point weight
            w = quad1d[1][m] * solid_angle / n_level
            # loop over azimuthal angles
            for i in range(n_level):
                phi = (i + 0.5) * delta
                ox, oy = (1-mu**2.)**0.5 * cos(phi), (1-mu**2.)**0.5 * sin(phi)
                self.aq_data[ct] = {'omega':np.array([ox, oy]), 'wt':w}
                ct += 1
        assert ct==self.n_dir "number of total directions are wrong"
        self.aq_data['n_dir'] = ct

    def get_aq_data(self):
        '''@brief Interface function to get aq_data

        @param self Reference to the class
        @return aq_data dictionary
        '''
        return self.aq_data
