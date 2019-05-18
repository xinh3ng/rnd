# -*- coding: utf-8 -*-
"""Unit tests of data util functions
"""
import unittest
from joblib import Parallel, delayed
import numpy as np


def zero_array(filename, shape):
    return np.zeros(shape)

        
class ParallelImage2ArrayTest(unittest.TestCase):
    
    def test_shape_is_preserved(self):
        """Test if the numpy array shape is perserved during parallel processing
        
        This usage is similar to my usage in image to array conversion
        """
        input_shape = (20, 10)
        filenames = range(100)
        
        input_shape_array = [input_shape for _ in range(len(filenames))]
        X0 = Parallel(n_jobs=-1)(delayed(zero_array)(fn, s) for fn, s in zip(filenames, input_shape_array))
        X0 = np.array(X0)
        
        X1 = np.zeros((len(filenames),) + input_shape)
        idx = 0
        for fn in filenames:            
            X1[idx, ...] = zero_array(fn, input_shape)
            idx += 1
        
        self.assertTrue(X0.shape == X1.shape)
        self.assertEqual(X0.shape, (len(filenames),) + input_shape)
        self.assertEqual(X0.max(), 0)
        self.assertEqual(X0.min(), 0)
        return


if __name__ == '__main__':
    unittest.main()