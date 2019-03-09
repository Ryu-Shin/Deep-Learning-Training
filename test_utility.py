import utility as util
import unittest
import numpy as np


def function(type):
    if type == 'polynomial':
        def function_ret(a):
            return 2*a*a + a + 5
    elif type == 'sin':
        def function_ret(a):
            return np.sin(a)
    elif type == 'cos':
        def function_ret(a):
            return np.cos(a)
    else:
        def function_ret(a):
            return 1
    return function_ret


class Test_identity_function(unittest.TestCase):
    def test_posi(self):
        val = 10
        expect = 10
        actual = util.identity_function(val)
        self.assertEqual(expect, actual)

    def test_nega(self):
        val = -157
        expect = -157
        actual = util.identity_function(val)
        self.assertEqual(expect, actual)

    def test_desimal(self):
        val = 1.578
        expect = 1.578
        actual = util.identity_function(val)
        self.assertEqual(expect, actual)

    def test_string(self):
        val = 'abcde'
        with self.assertRaises(ValueError) as er:
            util.identity_function(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : identity_function')

    def test_string_num(self):
        val = '-123'
        with self.assertRaises(ValueError) as er:
            util.identity_function(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : identity_function')

    def test_string_twobitechar(self):
        val = 'あいうえお'
        with self.assertRaises(ValueError) as er:
            util.identity_function(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : identity_function')

    def test_nega_10(self):
        val = np.array([[0,1,2,3,4,5,6,7,8,9]])
        val = -val
        expect = val
        actual = util.identity_function(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_desimal(self):
        val = np.array([[0,1,2,3,4,5,6,7,8,9]])
        val = val / 10
        expect = val
        actual = util.identity_function(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_posi_1d(self):
        val = np.array([0,1,2,3,4,5,6,7,8,9])
        expect = np.array([[0,1,2,3,4,5,6,7,8,9]])
        actual = util.identity_function(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_posi_batch(self):
        val = np.array([[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]])
        expect = val
        actual = util.identity_function(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_string(self):
        val = np.array([['a','b','c','d',5,'F','G','H','I',0]])
        with self.assertRaises(ValueError) as er:
            util.identity_function(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : identity_function')

    def test_string_num(self):
        val = np.array([['1','2','3','4',5,'6','7','8','9',0]])
        with self.assertRaises(ValueError) as er:
            util.identity_function(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : identity_function')

    def test_string_twobitechar(self):
        val = np.array([['あ','い','う','え',5,'お','安','伊','宇',0]])
        with self.assertRaises(ValueError) as er:
            util.identity_function(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : identity_function')

    def test_array(self):
        val = ['あ','い','う','え',5,'お','安','伊','宇',0]
        with self.assertRaises(ValueError) as er:
            util.identity_function(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : identity_function')


class Test_softmax(unittest.TestCase):
    def test_posi_1(self):
        val = np.array([[0,0,0,0,0]])
        expect = np.array([[0.2,0.2,0.2,0.2,0.2]])
        actual = util.softmax(val)
        # np.testing.assert_array_equal(expect,actual)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)


    def test_posi_10(self):
        val = np.array([[0,1,2,3,4,5,6,7,8,9]])
        expect = np.array([[0.00007801341613,0.0002120624514,0.0005764455082,0.00156694135,0.004259388198, 0.01157821754, 0.03147285834, 0.08555209893, 0.2325547159, 0.6321492584]])
        actual = util.softmax(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_nega_10(self):
        val = np.array([[0,1,2,3,4,5,6,7,8,9]])
        val = -val
        expect = np.array([[0.6321492584,0.2325547159,0.08555209893,0.03147285834,0.01157821754, 0.004259388198, 0.00156694135, 0.0005764455082, 0.0002120624514, 0.00007801341613]])
        actual = util.softmax(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_desimal(self):
        val = np.array([[0,1,2,3,4,5,6,7,8,9]])
        val = val / 10
        expect = np.array([[0.06120702456,0.06764422353,0.07475842862,0.08262084119,0.09131015091, 0.1009133233, 0.1115264702, 0.1232558114, 0.1362187383, 0.150544988]])
        actual = util.softmax(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_desimal_1d(self):
        val = np.array([0,1,2,3,4,5,6,7,8,9])
        val = val / 10
        expect = np.array([[0.06120702456,0.06764422353,0.07475842862,0.08262084119,0.09131015091, 0.1009133233, 0.1115264702, 0.1232558114, 0.1362187383, 0.150544988]])
        actual = util.softmax(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_desimal_batch(self):
        val = np.array([[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]])
        val = val / 10
        expect = np.array([[0.06120702456,0.06764422353,0.07475842862,0.08262084119,0.09131015091, 0.1009133233, 0.1115264702, 0.1232558114, 0.1362187383, 0.150544988], \
        [0.06120702456,0.06764422353,0.07475842862,0.08262084119,0.09131015091, 0.1009133233, 0.1115264702, 0.1232558114, 0.1362187383, 0.150544988], \
        [0.06120702456,0.06764422353,0.07475842862,0.08262084119,0.09131015091, 0.1009133233, 0.1115264702, 0.1232558114, 0.1362187383, 0.150544988]])
        actual = util.softmax(val)
        self.assertEqual(actual.shape, expect.shape)
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                self.failUnlessAlmostEqual(actual[i, j],expect[i, j],10)

    def test_string(self):
        val = np.array([['a','b','c','d',5,'F','G','H','I',0]])
        with self.assertRaises(ValueError) as er:
            actual = util.softmax(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : softmax')

    def test_string_num(self):
        val = np.array([['1','2','3','4',5,'6','7','8','9',0]])
        with self.assertRaises(ValueError) as er:
            actual = util.softmax(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : softmax')

    def test_string_twobitechar(self):
        val = np.array([['あ','い','う','え',5,'お','安','伊','宇',0]])
        with self.assertRaises(ValueError) as er:
            actual = util.softmax(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : softmax')

    def test_array(self):
        val = ['あ','い','う','え',5,'お','安','伊','宇',0]
        with self.assertRaises(ValueError) as er:
            actual = util.softmax(val)
        self.assertEqual(er.exception.args[0], 'Not a ndarray')

class Test_numerial_diff(unittest.TestCase):
    def test_posi(self):
        val = 10
        expect = 41
        actual = util.numerical_diff(function('polynomial'),val)
        self.failUnlessAlmostEqual(actual,expect,6)

    def test_nega(self):
        val = -157
        expect = -627
        actual = util.numerical_diff(function('polynomial'),val)
        self.failUnlessAlmostEqual(actual,expect,6)

    def test_desimal(self):
        val = 1.578
        expect = 7.312
        actual = util.numerical_diff(function('polynomial'),val)
        self.failUnlessAlmostEqual(actual,expect,6)

    def test_desimal_sin(self):
        val = 1.246
        expect = 0.31911576
        actual = util.numerical_diff(function('sin'),val)
        self.failUnlessAlmostEqual(actual,expect,6)

    def test_desimal_cos(self):
        val = 1.328
        expect = -0.9706694
        actual = util.numerical_diff(function('cos'),val)
        self.failUnlessAlmostEqual(actual,expect,6)


    def test_string(self):
        val = 'abcde'
        with self.assertRaises(ValueError) as er:
            util.numerical_diff(function('polynomial'),val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : numerical_diff')

    def test_string_num(self):
        val = '-123'
        with self.assertRaises(ValueError) as er:
            util.numerical_diff(function('polynomial'),val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : numerical_diff')

    def test_string_twobitechar(self):
        val = 'あいうえお'
        with self.assertRaises(ValueError) as er:
            util.numerical_diff(function('polynomial'),val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : numerical_diff')

if __name__ == "__main__":
    unittest.main()
