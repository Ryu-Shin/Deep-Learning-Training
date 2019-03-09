from layer import Layer
from except_def import UnableToUseException
import os, sys, unittest
import numpy as np



class Test_Layer(unittest.TestCase):
    def setUp(self):
        if sys.flags.debug:
            print(os.linesep + '> setUp method is called.')
        self.target = Layer(5)

    def tearDown(self):
        if sys.flags.debug:
            print(os.linesep + 'tearDown method is called.')

    def test_constructor_int(self):
        val = 1
        expect = 1
        self.smpl = Layer(val)
        actual = self.smpl.num_unit
        self.assertEqual(actual, expect)

    def test_constructor_nega(self):
        val = -3
        with self.assertRaises(ValueError) as er:
            self.smpl = Layer(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_constructor_desimal(self):
        val = 0.34
        with self.assertRaises(ValueError) as er:
            self.smpl = Layer(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_constructor_string(self):
        val = 'abcde'
        with self.assertRaises(ValueError) as er:
            self.smpl = Layer(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_forward(self):
        with self.assertRaises(UnableToUseException) as er:
            self.target.forward()
        self.assertEqual(er.exception.args[0], 'Uable to use this method: Layer')

    def test_backward(self):
        with self.assertRaises(UnableToUseException) as er:
            self.target.backward()
        self.assertEqual(er.exception.args[0], 'Uable to use this method: Layer')

    def test_check_input_forward_11111(self):
        val = np.array([[1, 1, 1, 1, 1]])
        expect = True
        actual = self.target.check_input_forward(val)
        self.assertEqual(actual, expect)

    def test_check_input_forward_disimal(self):
        val = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        expect = True
        actual = self.target.check_input_forward(val)
        self.assertEqual(actual, expect)


    def test_check_input_forward_array(self):
        val = [0.1, 0.1, 0.1, 0.1, 0.1]
        expect = False
        actual = self.target.check_input_forward(val)
        self.assertEqual(actual, expect)

    def test_check_input_forward_string(self):
        val = np.array([[0.1, 0.1, 0.1, 'abcde', 0.1]])
        expect = False
        actual = self.target.check_input_forward(val)
        self.assertEqual(actual, expect)

    def test_check_input_forward_int(self):
        val = 4
        expect = False
        actual = self.target.check_input_forward(val)
        self.assertEqual(actual, expect)

    def test_check_input_forward_wrong_size(self):
        val = np.array([[0.1, 0.1, 0.1, 0.1]])
        expect = False
        actual = self.target.check_input_forward(val)
        self.assertEqual(actual, expect)

    def test_reshape_input_1d(self):
        val = np.array([0.1, 0.1, 0.1, 0.1])
        expect = np.array([[0.1, 0.1, 0.1, 0.1]])
        actual = self.target.reshape_input(val)
        np.testing.assert_array_equal(expect,actual)

    def test_reshape_input_2d(self):
        val = np.array([[0.1, 0.1, 0.1, 0.1]])
        expect = np.array([[0.1, 0.1, 0.1, 0.1]])
        actual = self.target.reshape_input(val)
        np.testing.assert_array_equal(expect,actual)

    def test_reshape_input_batch(self):
        val = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
        expect = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
        actual = self.target.reshape_input(val)
        np.testing.assert_array_equal(expect,actual)

    def test_check_input_backward_11111(self):
        val = np.array([[1, 1, 1, 1, 1]])
        expect = True
        actual = self.target.check_input_backward(val)
        self.assertEqual(actual, expect)

    def test_check_input_backward_disimal(self):
        val = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        expect = True
        actual = self.target.check_input_backward(val)
        self.assertEqual(actual, expect)


    def test_check_input_backward_array(self):
        val = [0.1, 0.1, 0.1, 0.1, 0.1]
        expect = False
        actual = self.target.check_input_backward(val)
        self.assertEqual(actual, expect)

    def test_check_input_backward_string(self):
        val = np.array([[0.1, 0.1, 0.1, 'abcde', 0.1]])
        expect = False
        actual = self.target.check_input_backward(val)
        self.assertEqual(actual, expect)

    def test_check_input_backward_int(self):
        val = 4
        expect = False
        actual = self.target.check_input_backward(val)
        self.assertEqual(actual, expect)

    #def test_check_input_backward_wrong_size(self):
    #    val = np.array([[0.1, 0.1, 0.1, 0.1]])
    #    expect = False
    #    actual = self.target.check_input_backward(val)
    #    self.assertEqual(actual, expect)



if __name__ == "__main__":
    unittest.main()
