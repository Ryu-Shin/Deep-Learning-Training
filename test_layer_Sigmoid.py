from layer import Sigmoid
import unittest
import numpy as np

class Test_Sigmoid(unittest.TestCase):

    def setUp(self):
        self.x = np.array([ [0,-5.4,3.9,-7,0.2345] ])
        self.x_batch = np.array([ [0,-5.4,3.9,-7,0.2345], [0,-5.4,3.9,-7,0.2345], [0,-5.4,3.9,-7,0.2345] ])

        self.target = Sigmoid(3)
        self.target.num_unit = 5

        self.target_after_forward = Sigmoid(3)
        self.target_after_forward.num_unit = 5
        self.target_after_forward.y =  np.array([ [0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189] ])


        self.target_after_forward_batch = Sigmoid(3)
        self.target_after_forward_batch.num_unit = 5
        self.target_after_forward_batch.y = np.array([ [0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189], \
        [0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189], \
        [0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189] ])


    def tearDown(self):
        pass

    def test_constructor_int(self):
        val = 5
        expect = 5
        self.smpl = Sigmoid(val)
        actual = self.smpl.num_unit
        self.assertEqual(actual, expect)

    def test_constructor_nega(self):
        val = -5
        with self.assertRaises(ValueError) as er:
            self.smpl = Sigmoid(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_constructor_disimal(self):
        val = 1.345
        with self.assertRaises(ValueError) as er:
            self.smpl = Sigmoid(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_forward(self):
        val = self.x
        expect = np.array([[0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189]])
        expect_y = np.array([[0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189]])
        actual = self.target.forward(val)
        actual_y = self.target.y
        self.assertEqual(actual.shape, expect.shape)
        self.assertEqual(actual_y.shape, expect_y.shape)
        for i in range(actual.shape[1]):
            self.failUnlessAlmostEqual(actual[0, i],expect[0, i],6)
        for i in range(actual_y.shape[1]):
            self.failUnlessAlmostEqual(actual_y[0, i],expect_y[0, i],6)

    def test_forward_1d(self):
        val = np.array([0,-5.4,3.9,-7,0.2345])
        expect = np.array([[0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189]])
        expect_y = np.array([[0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189]])
        actual = self.target.forward(val)
        actual_y = self.target.y
        self.assertEqual(actual.shape, expect.shape)
        self.assertEqual(actual_y.shape, expect_y.shape)
        for i in range(actual.shape[1]):
            self.failUnlessAlmostEqual(actual[0, i],expect[0, i],6)
        for i in range(actual_y.shape[1]):
            self.failUnlessAlmostEqual(actual_y[0, i],expect_y[0, i],6)

    def test_forward_batch(self):
        val = self.x_batch
        expect = np.array([[0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189], [0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189], [0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189]])
        expect_y = np.array([[0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189], [0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189], [0.5, 0.00449627316, 0.9801596943, 0.0009110511944, 0.5583578189]])
        actual = self.target.forward(val)
        actual_y = self.target.y
        self.assertEqual(actual.shape, expect.shape)
        self.assertEqual(actual_y.shape, expect_y.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i,j],expect[i,j],6)
        for i in range(actual_y.shape[0]):
            for j in range(actual_y.shape[1]):
                self.failUnlessAlmostEqual(actual_y[i,j],expect_y[i,j],6)

    def test_forward_array(self):
        val = [0, 0, 3.9, 0, 0.2345]
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in Sigmoid')

    def test_forward_string(self):
        val = np.array([[0.1, 0.1, 0.1, 'abcde', 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in Sigmoid')

    def test_forward_int(self):
        val = 4
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in Sigmoid')

    def test_forward_wrong_size(self):
        val = np.array([[0.1, 0.1, 0.1, 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in Sigmoid')

    def test_backward_11111(self):
        val_dout = np.array([[1, 1, 1, 1, 1]])
        expect_dx = np.array([[0.25, 0.004476056687, 0.019446668, 0.0009102211801, 0.246594365]])
        actual_dx = self.target_after_forward.backward(val_dout)
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)

    def test_backward_disimal(self):
        val_dout = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        expect_dx = np.array([[0.025, 0.0004476056687, 0.0019446668, 0.00009102211801, 0.0246594365]])
        actual_dx = self.target_after_forward.backward(val_dout)
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)

    def test_backward_1d(self):
        val_dout = np.array([1, 1, 1, 1, 1])
        expect_dx = np.array([[0.25, 0.004476056687, 0.019446668, 0.0009102211801, 0.246594365]])
        actual_dx = self.target_after_forward.backward(val_dout)
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)

    def test_backward_batch(self):
        val_dout = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        expect_dx = np.array([[0.25, 0.004476056687, 0.019446668, 0.0009102211801, 0.246594365], [0.25, 0.004476056687, 0.019446668, 0.0009102211801, 0.246594365], \
        [0.25, 0.004476056687, 0.019446668, 0.0009102211801, 0.246594365]])
        actual_dx = self.target_after_forward_batch.backward(val_dout)
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[0]):
            for j in range(actual_dx.shape[1]):
                self.failUnlessAlmostEqual(actual_dx[i, j],expect_dx[i, j],10)

    def test_backward_array(self):
        val_dout = [0.1, 0.1, 0.1, 0.1, 0.1]
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in Sigmoid')

    def test_backward_string(self):
        val_dout = np.array([[0.1, 0.1, 0.1, 'abcde', 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in Sigmoid')

    def test_backward_int(self):
        val_dout = 4
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in Sigmoid')

    #def test_backward_wrong_size(self):
    #    val_dout = np.array([[0.1, 0.1, 0.1, 0.1]])
    #    with self.assertRaises(ValueError) as er:
    #        actual = self.target_after_forward.backward(val_dout)
    #    self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in Sigmoid')
