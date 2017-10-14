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
        assert sn_ord%2==0, 'SN order must be even'
        # int for sn order
        self._sn_ord = sn_ord
        # number of directions
        self._n_dir = (sn_ord+2) * sn_ord / 2
        # dictionary for containing angular quandrature directions and weights
        self._aq_data = {'omega':{},'wt':{},'dir_prods':{},'wt_tensor':{},
                         'bd_angle':{},'bd_vec_n':{},'refl_dir':{},'n_dir':self._n_dir}
        # make aq data
        self._quad2d()
        # store the outward normal vectors on boundaries
        self._aq_data['bd_vec_n'] = {
        'x_min':np.array([-1.,0]),'x_max':np.array([1.,0]),
        'y_min':np.array([0,-1.]),'y_max':np.array([0,1.])}
        # get incident and reflective directions
        self._boundary_info()

    def _quad2d(self):
        '''@brief Internal function used to calculate aq data

        @param self Reference to the class
        @return aq_data dictionary
        '''
        # initialize dictionary data
        ct, quad1d, solid_angle = 0, np.polynomial.legendre.leggauss(self._sn_ord), 4*pi
        # loop over relevant polar angles
        for m in xrange(self._sn_ord/2, self._sn_ord):
            # calculate number of points per level
            n_level = 4 * (self._sn_ord - m)
            delta = 2.0 * pi / n_level
            # get the polar angles
            mu = quad1d[0][m]
            # calculate point weight
            w = quad1d[1][m] * solid_angle / n_level
            # loop over azimuthal angles
            for i in range(n_level):
                phi = (i + 0.5) * delta
                ox, oy = (1-mu**2.)**0.5 * cos(phi), (1-mu**2.)**0.5 * sin(phi)
                self._aq_data['omega'][ct] = np.array([ox, oy])
                self._aq_data['wt'][ct] = w
                self._aq_data['wt_tensor'][ct] = w*np.outer(np.array([ox, oy]),np.array([ox, oy]))
                self._aq_data['dir_prods'][ct] = {'oxox':ox*ox,'oxoy':ox*oy,'oyoy':oy*oy}
                ct += 1
        assert ct==self._n_dir, "number of total directions are wrong"

    def _boundary_info(self):
        '''@brief Internal function to produce geometry related angular info, e.g.
        reflective angle index per boundary side and incident/outgoing angle on bd
        '''
        # bd_names
        bd_names = ['x_min','y_min','x_max','y_max']
        # boundary normal vectors
        vn = [np.array([-1.,0]),np.array([0,-1.]),np.array([1.,0]),np.array([0,1.])]
        self._aq_data['bd_vec_n'] = {k:v for k,v in zip(bd_names,vn)}
        # calculate boundary angles per dir, positive/negative means outgoing/incoming
        for bd in bd_names:
            for i in xrange(self._n_dir):
                vec_n,omega = self._aq_data['bd_vec_n'][bd],self._aq_data['omega'][i]
                self._aq_data['bd_angle'][(bd,i)]=np.dot(vec_n,omega)
                # get reflective angle
                if self._aq_data['bd_angle'][(bd,i)]>0:
                    self._aq_data['refl_dir'][(bd,i)]=None
                else:
                    # calculate reflective omega per boundary bd
                    r_dir = omega - 2.*vec_n*np.dot(omega,vec_n)
                    # loop over all angles
                    for ind,r_omega in self._aq_data['omega'].items():
                        # collect index if the angle is equal to calculated refl angle
                        if np.allclose(r_dir,r_omega,rtol=1.0e-10,atol=1.0e-10):
                            self._aq_data['refl_dir'][(bd,i)] = ind
                            break

    def get_aq_data(self):
        '''@brief Interface function to get aq_data

        @param self Reference to the class
        @return aq_data dictionary
        '''
        return self._aq_data
