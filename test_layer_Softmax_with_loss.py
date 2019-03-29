from layer import Softmax_with_loss
import unittest
import numpy as np

class Test_Sofmax_with_loss(unittest.TestCase):

    def setUp(self):
        self.x = np.array([ [0,-5.4,3.9,-7,0.2345] ])
        self.t =  np.array([ [0, 1, 0, 0, 0] ])
        self.x_batch = np.array([ [0,-5.4,3.9,-7,0.2345], [0,-5.4,3.9,-7,0.2345], [0,-5.4,3.9,-7,0.2345] ])
        self.t_batch =  np.array([ [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0] ])

        self.target = Softmax_with_loss(3)
        self.target.num_unit = 5

        self.target_after_forward = Softmax_with_loss(3)
        self.target_after_forward.num_unit = 5
        self.target_after_forward.y =  np.array([ [0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861] ])
        self.target_after_forward.t =  np.array([ [0, 1, 0, 0, 0] ])

        self.target_after_forward_batch = Softmax_with_loss(3)
        self.target_after_forward_batch.num_unit = 5
        self.target_after_forward_batch.y = np.array([ [0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861], \
        [0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861], \
        [0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861] ])
        self.target_after_forward_batch.t =  np.array([ [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0] ])

    def tearDown(self):
        pass

    def test_constructor_int(self):
        val = 3
        expect = 3
        self.smpl = Softmax_with_loss(val)
        actual = self.smpl.num_unit
        self.assertEqual(actual, expect)

    def test_constructor_nega(self):
        val = -5
        with self.assertRaises(ValueError) as er:
            self.smpl = Softmax_with_loss(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_constructor_disimal(self):
        val = 1.345
        with self.assertRaises(ValueError) as er:
            self.smpl = Softmax_with_loss(val)
        self.assertEqual(er.exception.args[0], 'The value is not permitted : constructor of Layer')

    def test_forward(self):
        val_x = self.x
        expect_y = np.array([[0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861]])
        actual_y = self.target.forward(val_x)
        self.assertEqual(actual_y.shape, expect_y.shape)
        for i in range(actual_y.shape[1]):
            self.failUnlessAlmostEqual(actual_y[0, i],expect_y[0, i],6)

#####

    def test_forward_with_loss(self):
        val_x = self.x
        val_t = self.t
        expect_loss = 9.34491903
        expect_y = np.array([[0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861]])
        expect_t = np.array([[0, 1, 0, 0, 0]])
        actual_loss = self.target.forward_with_loss(val_x, val_t)
        actual_y = self.target.y
        actual_t = self.target.t
        self.assertEqual(actual_y.shape, expect_y.shape)
        self.assertEqual(actual_t.shape, expect_t.shape)
        for i in range(actual_y.shape[1]):
            self.failUnlessAlmostEqual(actual_y[0, i],expect_y[0, i],6)
        for i in range(actual_t.shape[1]):
            self.failUnlessAlmostEqual(actual_t[0, i],expect_t[0, i],6)
        self.failUnlessAlmostEqual(actual_loss, expect_loss, 5)


    def test_forward_with_loss_1d(self):
        val_x = np.array([0,-5.4,3.9,-7,0.2345])
        val_t = np.array([0, 1, 0, 0, 0])
        expect_loss = 9.34491903
        expect_y = np.array([[0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861]])
        expect_t = np.array([[0, 1, 0, 0, 0]])
        actual_loss = self.target.forward_with_loss(val_x, val_t)
        actual_y = self.target.y
        actual_t = self.target.t
        self.assertEqual(actual_y.shape, expect_y.shape)
        self.assertEqual(actual_t.shape, expect_t.shape)
        for i in range(actual_y.shape[1]):
            self.failUnlessAlmostEqual(actual_y[0, i],expect_y[0, i],6)
        for i in range(actual_t.shape[1]):
            self.failUnlessAlmostEqual(actual_t[0, i],expect_t[0, i],6)
        self.failUnlessAlmostEqual(actual_loss, expect_loss, 5)

    def test_forward_with_loss_batch(self):
        val_x = self.x_batch
        val_t = self.t_batch
        expect_loss = 4.366752363
        expect_y = np.array([[0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861],\
         [0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861], [0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, 0.02446726861]])
        expect_t = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]])
        actual_loss = self.target.forward_with_loss(val_x, val_t)
        actual_y = self.target.y
        actual_t = self.target.t
        self.assertEqual(actual_y.shape, expect_y.shape)
        self.assertEqual(actual_t.shape, expect_t.shape)
        for i in range(actual_y.shape[0]):
            for j in range(actual_y.shape[1]):
                self.failUnlessAlmostEqual(actual_y[i,j],expect_y[i,j],6)
        for i in range(actual_t.shape[0]):
            for j in range(actual_t.shape[1]):
                self.failUnlessAlmostEqual(actual_t[i,j],expect_t[i,j],6)
        self.failUnlessAlmostEqual(actual_loss, expect_loss, 5)


    def test_backward(self):
        expect_dx = np.array([[0.0193527833, -0.9999125916, 0.9560748922, 0.00001764745408, 0.02446726861]])
        actual_dx = self.target_after_forward.backward()
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[1]):
            self.failUnlessAlmostEqual(actual_dx[0, i],expect_dx[0, i],10)

    def test_backward_batch(self):
        expect_dx = np.array([[0.0193527833, -0.9999125916, 0.9560748922, 0.00001764745408, 0.02446726861], [0.0193527833, 0.00008740841226, 0.9560748922, 0.00001764745408, -0.9755327314], \
        [0.0193527833, 0.00008740841226, -0.0439251078, 0.00001764745408, 0.02446726861]])
        expect_dx = expect_dx / 3.0
        actual_dx = self.target_after_forward_batch.backward()
        self.assertEqual(actual_dx.shape, expect_dx.shape)
        for i in range(actual_dx.shape[0]):
            for j in range(actual_dx.shape[1]):
                self.failUnlessAlmostEqual(actual_dx[i, j],expect_dx[i, j],10)
