from layer import ReLU
import unittest
import numpy as np

class Test_ReLU(unittest.TestCase):

    def setUp(self):
        self.x = np.array([ [0,-5.4,3.9,-7,0.2345] ])
        self.x_batch = np.array([ [0,-5.4,3.9,-7,0.2345], [0,-5.4,3.9,-7,0.2345], [0,-5.4,3.9,-7,0.2345], [0,-5.4,3.9,-7,0.2345], [0,-5.4,3.9,-7,0.2345] ])

        self.target = ReLU(3)
        self.target.num_unit = 5

        self.target_after_forward = ReLU(3)
        self.target_after_forward.num_unit = 5
        self.target_after_forward.mask = np.array([ [True,True,False,True,False] ])

        self.target_after_forward_batch = ReLU(3)
        self.target_after_forward_batch.num_unit = 5
        self.target_after_forward_batch.mask = np.array([ [True,True,False,True,False], [True,True,False,True,False], [True,True,False,True,False], [True,True,False,True,False], [True,True,False,True,False] ])


    def tearDown(self):
        pass

    def test_constructor_int(self):
        val = 5
        expect = 5
        self.smpl = ReLU(val)
        actual = self.smpl.num_unit
        self.assertEqual(actual, expect)

    def test_constructor_nega(self):
        val = -5
        with self.assertRaises(ValueError) as er:
            self.smpl = ReLU(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_constructor_disimal(self):
        val = 1.345
        with self.assertRaises(ValueError) as er:
            self.smpl = ReLU(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_forward(self):
        val = self.x
        expect = np.array([[0, 0, 3.9, 0, 0.2345]])
        expect_mask = np.array([[True, True, False, True, False]])
        actual = self.target.forward(val)
        actual_mask = self.target.mask
        np.testing.assert_array_equal(expect,actual)
        np.testing.assert_array_equal(expect_mask,actual_mask)

    def test_forward_1d(self):
        val = np.array([0,-5.4,3.9,-7,0.2345])
        expect = np.array([[0, 0, 3.9, 0, 0.2345]])
        expect_mask = np.array([[True, True, False, True, False]])
        actual = self.target.forward(val)
        actual_mask = self.target.mask
        np.testing.assert_array_equal(expect,actual)
        np.testing.assert_array_equal(expect_mask,actual_mask)

    def test_forward_batch(self):
        val = self.x_batch
        expect = np.array([[0, 0, 3.9, 0, 0.2345], [0, 0, 3.9, 0, 0.2345], [0, 0, 3.9, 0, 0.2345], [0, 0, 3.9, 0, 0.2345], [0, 0, 3.9, 0, 0.2345]])
        expect_mask = np.array([[True, True, False, True, False], [True, True, False, True, False], [True, True, False, True, False], [True, True, False, True, False], [True, True, False, True, False]])
        actual = self.target.forward(val)
        actual_mask = self.target.mask
        np.testing.assert_array_equal(expect,actual)
        np.testing.assert_array_equal(expect_mask,actual_mask)

    def test_forward_array(self):
        val = [0, 0, 3.9, 0, 0.2345]
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in ReLU')


    def test_forward_string(self):
        val = np.array([[0.1, 0.1, 0.1, 'abcde', 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in ReLU')

    def test_forward_int(self):
        val = 4
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in ReLU')

    def test_forward_wrong_size(self):
        val = np.array([[0.1, 0.1, 0.1, 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in ReLU')


    def test_backward_11111(self):
        val_dout = np.array([[1, 1, 1, 1, 1]])
        expect_dx = np.array([[0, 0, 1, 0, 1]])
        actual_dx = self.target_after_forward.backward(val_dout)
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)

    def test_backward_disimal(self):
        val_dout = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        expect_dx = np.array([[0, 0, 0.1, 0, 0.1]])
        actual_dx = self.target_after_forward.backward(val_dout)
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)

    def test_backward_1d(self):
        val_dout = np.array([1, 1, 1, 1, 1])
        expect_dx = np.array([[0, 0, 1, 0, 1]])
        actual_dx = self.target_after_forward.backward(val_dout)
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)

    def test_backward_batch(self):
        val_dout = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [3.4, 3.4, 3.4, 3.4, 3.4]])
        expect_dx = np.array([[0, 0, 1, 0, 1], [0, 0, 1, 0, 1], [0, 0, 2, 0, 2], [0, 0, 1, 0, 1], [0, 0, 3.4, 0, 3.4]])
        actual_dx = self.target_after_forward_batch.backward(val_dout)
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[0]):
            for j in range(actual_dx.shape[1]):
                self.failUnlessAlmostEqual(actual_dx[i, j],expect_dx[i, j],10)

    def test_backward_array(self):
        val_dout = [0.1, 0.1, 0.1, 0.1, 0.1]
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in ReLU')

    def test_backward_string(self):
        val_dout = np.array([[0.1, 0.1, 0.1, 'abcde', 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in ReLU')

    def test_backward_int(self):
        val_dout = 4
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in ReLU')

    # def test_backward_wrong_size(self):
    #    val_dout = np.array([[0.1, 0.1, 0.1, 0.1]])
    #    with self.assertRaises(ValueError) as er:
    #        actual = self.target_after_forward.backward(val_dout)
    #    self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in ReLU')
