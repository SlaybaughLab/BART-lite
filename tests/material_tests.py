from nose.tools import *
from material import material
from material import _mat

class TestClass:

    @classmethod
    def setup_class(cls):
        filename = './tests/testData/test_mat.xml'
        cls.testmat = _mat(filename)

    @raises(AssertionError)
    def test_mat_bad_filename(self):
        """ Reading a bad filename should return an assertion error. """
        filename = './tests/testData/badname.xml'
        badMat = _mat(filename)

    def test_mat_read(self):
        """ Reading a file should save the correct values to class vars. """
        ok_(type(self.testmat.nu) == float, "Nu should be a float")
        ok_(self.testmat.nu == 2.3, "Nu should be the correct value")
