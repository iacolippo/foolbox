from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import numpy as np
from lightonml.datasets import MNIST
import matplotlib.pyplot as plt

from ridge_extension import RidgeClassifierWithProba
from foolbox.models.sklearn import SklearnModel
from foolbox.models.pytorch import PyTorchModel
from foolbox.models.wrappers import CompositeModel
from foolbox.attacks import SaltAndPepperNoiseAttack, SinglePixelAttack
from foolbox.criteria import Misclassification


from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='~/MNIST')
X = mnist.data.astype('float32')
X /= 255
y = mnist.target

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print('Create pipeline')
clf = RidgeClassifierWithProba()
pipe = Pipeline(steps=[('clf', clf)])

print('Fit and score')
pipe.fit(X=X_train, y=y_train)
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)

print('Train accuracy: {}%\t Test accuracy: {}%'.format(train_acc, test_acc))

forward_model = SklearnModel(model=pipe, bounds=(0, 1), num_classes=10)

import torch.nn as nn
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x.view(x.size(0), -1))
        out = self.relu(out)
        out = self.fc2(out)
        return out
net = Net(784, 100, 10).eval()
backward_model = PyTorchModel(model=net, bounds=(0, 1), num_classes=10)

model = CompositeModel(forward_model, backward_model)

target_class = 1
criterion = Misclassification()

attack = SinglePixelAttack(model, criterion)

from foolbox.adversarial import Adversarial
sample = X_train[100].reshape(1, 28, 28)
label = np.argmax(model.predictions(sample))
print(label)
image = Adversarial(model, criterion, sample, 0)

adversarial = attack(image)


plt.subplot(1, 3, 1)
plt.imshow(sample[0], cmap='gray')

if adversarial is not None:
    label = np.argmax(model.predictions(adversarial))
    print(label)

    plt.subplot(1, 3, 2)
    plt.imshow(adversarial[0], cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(adversarial[0] - sample[0], cmap='gray')
plt.show()
