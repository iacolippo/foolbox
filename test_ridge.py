from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import numpy as np
from lightonml.datasets import MNIST
import matplotlib.pyplot as plt

from ridge_extension import RidgeClassifierWithProba
from foolbox.models.sklearn import SklearnModel
from foolbox.attacks import *
from foolbox.criteria import *


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

model = SklearnModel(model=pipe, bounds=(0, 1), num_classes=10)

target_class = 1
criterion = Misclassification()

attack = SaltAndPepperNoiseAttack(model, criterion)

from foolbox.adversarial import Adversarial
sample = X_train[10].reshape(1, 28, 28)
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
