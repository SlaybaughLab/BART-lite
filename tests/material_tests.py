from nose.tools import *
from material import material

class TestClass:

    @classmethod
    def setup_class(cls):
        cls.testMat = material()

    @raises(AssertionError)
    def test_mat_bad_filename(self):
        """ Reading a bad filename should return an assertion error. """
        filename = './tests/test_data/badname.xml'
        self.testMat.read(filename)
    
