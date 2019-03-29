import os, sys, unittest
from layer import Affine, Sigmoid, Softmax_with_loss
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle

class Test_prediction(unittest.TestCase):

    def setUp(self):
        (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=True)
        print(x_train.shape)
        print(t_train.shape)
        print(x_test.shape)
        print(t_test.shape)
        self.img = x_train[0]
        self.label = t_train[0]
        self.xtest = x_test
        self.ttest = t_test
        # img = img.reshape(28, 28)
        # pil_img = Image.fromarray(np.uint8(img))
        # pil_img.show()
        with open("sample_weight.pkl", 'rb') as f:
            self.network = pickle.load(f)
        self.W1 = self.network['W1']
        self.W2 = self.network['W2']
        self.W3 = self.network['W3']
        self.b1 = self.network['b1']
        self.b2 = self.network['b2']
        self.b3 = self.network['b3']
        self.b1 = self.b1.reshape(1, -1)
        self.b2 = self.b2.reshape(1, -1)
        self.b3 = self.b3.reshape(1, -1)
        print(self.W1.shape)
        print(self.b1.shape)
        print(self.W2.shape)
        print(self.b2.shape)
        print(self.W3.shape)
        print(self.b3.shape)

        self.layer1 = Affine(self.W1, self.b1)
        self.sigmoid1 = Sigmoid(self.b1.shape[1])
        self.layer2 = Affine(self.W2, self.b2)
        self.sigmoid2 = Sigmoid(self.b2.shape[1])
        self.layer3 = Affine(self.W3, self.b3)
        self.softmax_with_loss = Softmax_with_loss(self.b3.shape[1])

    def tearDown(self):
        pass

    def test(self):
        accuracy_cnt = 0
        for i in range(len(self.xtest)):
            a1 = self.layer1.forward(self.xtest[i])
            z1 = self.sigmoid1.forward(a1)
            a2 = self.layer2.forward(z1)
            z2 = self.sigmoid2.forward(a2)
            a3 = self.layer3.forward(z2)
            loss = self.softmax_with_loss.forward_with_loss(a3, self.ttest[i])
            y = self.softmax_with_loss.y
            p = np.argmax(y)
            if self.ttest[i][p] == 1:
                accuracy_cnt += 1
        Accuracy = float(accuracy_cnt) / len(self.ttest)
        expect = 0.9352
        print("Accuracy:" + str(Accuracy))
        self.failUnlessAlmostEqual(Accuracy,expect,10)

if __name__ == "__main__":
    unittest.main()
