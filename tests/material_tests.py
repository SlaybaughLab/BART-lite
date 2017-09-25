from nose.tools import *
from bart_io import mat_io

class TestClass:

    @classmethod
    def setup_class(cls):
        filename = './tests/testData/test_mat.xml'
        cls.testMat = mat_io(filename)
    
    @raises(AssertionError)
    def test_mat_io_nofile(self):
        """ mat_io function returns error if file is no found """
        mat_io(filename="testdata/badfile.xml")
