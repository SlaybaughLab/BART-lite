from nose.tools import *
from material import mat_map
from material import mat_lib
import numpy as np

testData_loc = './tests/testData/materials/'

class TestFunctionality:
    # Tests to verify the material library class
    
    @classmethod
    def setup_class(cls):
        files = ['test_mat.xml', 'test_mat2.xml']
        filelocs = [testData_loc + f for f in files]
        cls.lib = mat_lib(n_grps = 2, files = filelocs)
        cls.mat_dict = {'1': 'test_mat', '2': 'test_mat2'}
        goodmap = """ 1 1 1 1
                          1 2 2 1
                          1 2 2 1
                          1 1 1 1 """
        cls.also_good = """ 1 1 1 1 1 20 2 1 1 2 2 1 1 1 1 1 """
        cls.badmap = """ 1 1 1 1
                         1 20 2
                         1 2 2 1
                         1 1 1 1 """
        cls.testmap = mat_map(lib=cls.lib, layout=goodmap,
                              mat_dict = cls.mat_dict, x_max = 10, n=20)

    # TEST CONSTRUCTOR ###############################################

    def test_init_with_good_map(self):
        """ Initializing with a good map should build the mapping
        correctly """
        eq_(self.testmap.x, [0, 10], "correct x domain")
        eq_(self.testmap.y, [0, 10], "correct y domain")
        eq_(self.testmap.layout, [['1', '1', '1', '1'],
                                  ['1', '2', '2', '1'],
                                  ['1', '2', '2', '1'],
                                  ['1', '1', '1', '1']])

    @raises(AssertionError)
    def test_init_with_bad_map(self):
        badMap = mat_map(lib=self.lib, layout=self.badmap,
                         mat_dict = self.mat_dict, x_max = 10, n=20)
