from layer import Affine
import os, sys, unittest
from unittest.mock import Mock
import numpy as np

class Test_Affine(unittest.TestCase):
    def setUp(self):
        if sys.flags.debug:
            print(os.linesep + '> setUp method is called.')
        W = np.array([ [2.1,-5.4,3.9,7,0.2345], [2.1,-5.4,3.9,7,0.2345], [2.1,-5.4,3.9,7,0.2345], [2.1,-5.4,3.9,7,0.2345] ])
        b = np.array([[2,5,3.9,-8.7,1.574324]])
        self.target = Affine(W, b)
        self.target.W = W
        self.target.b = b
        self.target.num_unit = 4

        self.target_after_forward = Affine(W, b)
        self.target_after_forward.W = W
        self.target_after_forward.b = b
        self.target_after_forward.num_unit = 4
        self.target_after_forward.x = np.array([[1,1,1,1]])

        self.target_after_forward_batch = Affine(W, b)
        self.target_after_forward_batch.W = W
        self.target_after_forward_batch.b = b
        self.target_after_forward_batch.num_unit = 4
        self.target_after_forward_batch.x = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1]])

        self.target_after_backward = Affine(W, b)
        self.target_after_backward.W = W
        self.target_after_backward.b = b
        self.target_after_backward.num_unit = 4
        self.target_after_backward.x = np.array([[1,1,1,1]])
        self.target_after_backward.dW = np.array([ [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]])
        self.target_after_backward.db = np.array([[0.78345, 0.78345, 0.78345, 0.78345, 0.78345]])

    def tearDown(self):
        if sys.flags.debug:
            print(os.linesep + 'tearDown method is called.')

    def test_constructor_nparray(self):
        val_W = np.array([ [1,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324] ])
        expect_W = val_W
        val_b = np.array([[2,5,3.9,-8.7,1.574324]])
        expect_b = val_b
        expect_num_unit = 5

        self.smpl = Affine(val_W, val_b)
        actual_W = self.smpl.W
        actual_b = self.smpl.b
        actual_num_unit = self.smpl.num_unit

        np.testing.assert_array_equal(expect_W, actual_W)
        np.testing.assert_array_equal(expect_b, actual_b)
        self.assertEqual(actual_num_unit, expect_num_unit)

    def test_constructor_wrong_size_Wb(self):
        val_W = np.array([ [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324] ])
        val_b = np.array([2,5,3.9,-8.7])
        with self.assertRaises(ValueError) as er:
            self.smpl = Affine(val_W, val_b)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Affine')

    def test_constructor_wrong_size_b(self):
        val_W = np.array([ [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324] ])
        val_b = np.array([ [2,5,3.9,-8.7,1.574324], [2,5,3.9,-8.7,1.574324] ])
        with self.assertRaises(ValueError) as er:
            self.smpl = Affine(val_W, val_b)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Affine')


    def test_constructor_string_W(self):
        val_W = np.array([ [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,'3.9',-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324] ])
        val_b = np.array([2,5,3.9,-8.7,1.574324])
        with self.assertRaises(ValueError) as er:
            self.smpl = Affine(val_W, val_b)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Affine')

    def test_constructor_string_b(self):
        val_W = np.array([ [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324] ])
        val_b = np.array([2,5,3.9,'-8.7',1.574324])
        with self.assertRaises(ValueError) as er:
            self.smpl = Affine(val_W, val_b)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Affine')

    def test_constructor_array_W(self):
        val_W = [ [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324] ]
        val_b = np.array([2,5,3.9,-8.7,1.574324])
        with self.assertRaises(ValueError) as er:
            self.smpl = Affine(val_W, val_b)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Affine')

    def test_constructor_array_b(self):
        val_W = np.array([ [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324] ])
        val_b = [2,5,3.9,-8.7,1.574324]
        with self.assertRaises(ValueError) as er:
            self.smpl = Affine(val_W, val_b)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Affine')

    def test_constructor_float_W(self):
        val_W = 1.23456
        val_b = np.array([2,5,3.9,-8.7,1.574324])
        with self.assertRaises(ValueError) as er:
            self.smpl = Affine(val_W, val_b)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Affine')

    def test_constructor_float_b(self):
        val_W = np.array([ [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324], [2.1,-5.4,3.9,7,0.2345], [2,5,3.9,-8.7,1.574324] ])
        val_b = -1.2345
        with self.assertRaises(ValueError) as er:
            self.smpl = Affine(val_W, val_b)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Affine')

    def test_forward_1111(self):
        val_x = np.array([[1, 1, 1, 1]])
        expect = np.array([[10.4, -16.6, 19.5, 19.3, 2.512324]])
        actual = self.target.forward(val_x)
        np.testing.assert_array_equal(val_x, self.target.x)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(expect.shape[1]):
            self.failUnlessAlmostEqual(actual[0, i],expect[0, i],10)
        W = np.array([ [2.1,-5.4,3.9,7,0.2345], [2.1,-5.4,3.9,7,0.2345], [2.1,-5.4,3.9,7,0.2345], [2.1,-5.4,3.9,7,0.2345] ])
        b = np.array([[2,5,3.9,-8.7,1.574324]])

    def test_forward_1111_disimal(self):
        val_x = np.array([[0.1, 0.1, 0.1, 0.1]])
        expect = np.array([[2.84, 2.84, 5.46, -5.9, 1.668124]])
        actual = self.target.forward(val_x)
        np.testing.assert_array_equal(val_x, self.target.x)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(expect.shape[1]):
            self.failUnlessAlmostEqual(actual[0, i],expect[0, i],10)

    def test_forward_1d(self):
        val_x = np.array([0.1, 0.1, 0.1, 0.1])
        expect = np.array([[2.84, 2.84, 5.46, -5.9, 1.668124]])
        actual = self.target.forward(val_x)
        np.testing.assert_array_equal(np.array([[0.1, 0.1, 0.1, 0.1]]), self.target.x)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(expect.shape[1]):
            self.failUnlessAlmostEqual(actual[0, i],expect[0, i],10)

    def test_forward_array(self):
        val_x = [0.1, 0.1, 0.1, 0.1, 0.1]
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val_x)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in Affine')

    def test_forward_string(self):
        val_x = np.array([[0.1, 0.1, 0.1, 'abcde', 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val_x)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in Affine')

    def test_forward_int(self):
        val_x = 4
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val_x)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in Affine')

    def test_forward_wrong_size(self):
        val_x = np.array([[0.1, 0.1, 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target.forward(val_x)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : forward in Affine')


    def test_backward_11111(self):
        val_dout = np.array([[1, 1, 1, 1, 1]])
        expect_dW = np.array([ [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1] ])
        expect_dx = np.array([[7.8345, 7.8345, 7.8345, 7.8345]])
        expect_db = np.array([[1, 1, 1, 1, 1]])
        actual_dx = self.target_after_forward.backward(val_dout)
        actual_dW = self.target_after_forward.dW
        actual_db = self.target_after_forward.db
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        self.assertEqual(actual_db.shape, expect_db.shape)
        self.assertEqual(actual_dW.shape, expect_dW.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)
        for i in range(actual_db.shape[1]):
            self.failUnlessAlmostEqual(actual_db[0, i],expect_db[0, i],10)
        for i in range(actual_dW.shape[0]):
            for j in range(actual_dW.shape[1]):
                self.failUnlessAlmostEqual(actual_dW[i, j],expect_dW[i, j],10)


    def test_backward_11111_disimal(self):
        val_dout = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        expect_dW = np.array([ [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]])
        expect_dx = np.array([[0.78345, 0.78345, 0.78345, 0.78345]])
        expect_db = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        actual_dx = self.target_after_forward.backward(val_dout)
        actual_dW = self.target_after_forward.dW
        actual_db = self.target_after_forward.db
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        self.assertEqual(actual_db.shape, expect_db.shape)
        self.assertEqual(actual_dW.shape, expect_dW.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)
        for i in range(actual_db.shape[1]):
            self.failUnlessAlmostEqual(actual_db[0, i],expect_db[0, i],10)
        for i in range(actual_dW.shape[0]):
            for j in range(actual_dW.shape[1]):
                self.failUnlessAlmostEqual(actual_dW[i, j],expect_dW[i, j],10)

    def test_backward_1d(self):
        val_dout = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        expect_dW = np.array([ [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]])
        expect_dx = np.array([[0.78345, 0.78345, 0.78345, 0.78345]])
        expect_db = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        actual_dx = self.target_after_forward.backward(val_dout)
        actual_dW = self.target_after_forward.dW
        actual_db = self.target_after_forward.db
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        self.assertEqual(actual_db.shape, expect_db.shape)
        self.assertEqual(actual_dW.shape, expect_dW.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)
        for i in range(actual_db.shape[1]):
            self.failUnlessAlmostEqual(actual_db[0, i],expect_db[0, i],10)
        for i in range(actual_dW.shape[0]):
            for j in range(actual_dW.shape[1]):
                self.failUnlessAlmostEqual(actual_dW[i, j],expect_dW[i, j],10)

    def test_backward_batch(self):
        val_dout = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3, 0.3]])
        expect_dW = np.array([ [0.6, 0.6, 0.6, 0.6, 0.6], [0.6, 0.6, 0.6, 0.6, 0.6], [0.6, 0.6, 0.6, 0.6, 0.6], [0.6, 0.6, 0.6, 0.6, 0.6] ])
        expect_dx = np.array([[0.78345, 0.78345, 0.78345, 0.78345], [1.5669, 1.5669, 1.5669, 1.5669], [2.35035, 2.35035, 2.35035, 2.35035]])
        expect_db = np.array([[0.6, 0.6, 0.6, 0.6, 0.6]])
        actual_dx = self.target_after_forward_batch.backward(val_dout)
        actual_dW = self.target_after_forward_batch.dW
        actual_db = self.target_after_forward_batch.db
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        self.assertEqual(actual_db.shape, expect_db.shape)
        self.assertEqual(actual_dW.shape, expect_dW.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)
        for i in range(actual_db.shape[1]):
            self.failUnlessAlmostEqual(actual_db[0, i],expect_db[0, i],10)
        for i in range(actual_dW.shape[0]):
            for j in range(actual_dW.shape[1]):
                self.failUnlessAlmostEqual(actual_dW[i, j],expect_dW[i, j],10)

    def test_backward_array(self):
        val_dout = [0.1, 0.1, 0.1, 0.1, 0.1]
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in Affine')

    def test_backward_string(self):
        val_dout = np.array([[0.1, 0.1, 0.1, 'abcde', 0.1]])
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in Affine')

    def test_backward_int(self):
        val_dout = 4
        with self.assertRaises(ValueError) as er:
            actual = self.target_after_forward.backward(val_dout)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in Affine')

    #def test_backward_wrong_size(self):
    #    val_dout = np.array([[0.1, 0.1, 0.1, 0.1]])
    #    with self.assertRaises(ValueError) as er:
    #        actual = self.target_after_forward.backward(val_dout)
    #    self.assertEqual(er.exception.args[0], 'The value is not permitted : backward in Affine')

    def test_update(self):
        W_arg = self.target_after_backward.W
        b_arg = self.target_after_backward.b
        dW_arg = self.target_after_backward.dW
        db_arg = self.target_after_backward.db
        mock = Mock()
        self.target_after_backward.update(mock)
        mock.update.assert_called_with(W_arg, b_arg, dW_arg, db_arg)
        pass

if __name__ == "__main__":
    unittest.main()
