import numpy as np
import torch.cuda
import torchvision.transforms as transforms
from sklearn.metrics import f1_score

from dataset import *
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch as t
import matplotlib.pyplot as plt

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    lambda x: x[:3],
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

profile_val = [];
profile_train = [];
profile_test = [];
Acc= [];
trainLoader,val, testLoader = getloaders(preprocess, bs = 4);

def accuracy(outputs, trues):
    # store 1 in acc for correct prdiction, 0 otherwise
    _, predicted = torch.max(outputs.data, 1) # Multi Dimantional Array

    acc = (predicted.view(-1) == trues.view(-1))
    acc = torch.sum(acc) / len(acc)
    return (acc.item())


def train (net, trloader, valoader, opt, loss, epochs =5,):
    net.to(device = "cuda" if torch.cuda.is_available() else "cpu")
    for e in range(epochs):
        train_loss_perEpo = []
        val_loss_perEpo = []
        for img, label in trloader:
            img = img.to(device = "cuda" if torch.cuda.is_available() else "cpu")
            label = label.to(device = "cuda" if torch.cuda.is_available() else "cpu")
          # Contain all acc
            pred = net(img)
            l = loss(pred, label)
            train_loss_perEpo.append(l.item())
            l.backward()
            opt.step()
            opt.zero_grad()
            # accuracy
            Acc.append(accuracy(pred,label))
        print("Accurac for train" , np.sum(Acc)/ len(Acc))

        profile_train.append(sum(train_loss_perEpo)/len(train_loss_perEpo))
        for img, label in valoader:
            img = img.to(device = "cuda" if torch.cuda.is_available() else "cpu")
            label = label.to(device = "cuda" if torch.cuda.is_available() else "cpu")

            pred = net(img)
            l = loss(pred, label)
            val_loss_perEpo.append(l.item())
            # accuracy
            Acc.append(accuracy(pred, label))
        print("Accurac for Validation" ,np.sum(Acc) / len(Acc))
        profile_val.append(sum(val_loss_perEpo)/len(val_loss_perEpo))


net = models.resnet18()
net.fc = nn.Linear(512, 5)
opt = optim.Adam(net.parameters())
loss = nn.CrossEntropyLoss()


train(net, trainLoader,val, opt, loss, epochs= 5)

print("loss for val",profile_val)
print("loss for train", profile_train)

########test#######

def test (net, testLoader, opt, loss, epochs =5,):
    net.to(device = "cuda" if torch.cuda.is_available() else "cpu")
    for e in range(epochs):
        test_loss_perEpo = []

        for img, label in testLoader:
            img = img.to(device = "cuda" if torch.cuda.is_available() else "cpu")
            label = label.to(device = "cuda" if torch.cuda.is_available() else "cpu")

            pred = net(img)
            l = loss(pred, label)
            test_loss_perEpo.append(l.item())
            l.backward()
            opt.step()
            opt.zero_grad()
            # accuracy
            Acc.append(accuracy(pred, label))
        print("Accurac for test" ,np.sum(Acc) / len(Acc))
        profile_test.append(sum(test_loss_perEpo)/len(test_loss_perEpo))



test(net, testLoader, opt, loss, epochs= 10)

print("loss for test", profile_test)

# حفظ ال model

#t.save(net.state_dict(), "model.h5")

#تحميل الmodel
"""""
w = t.load("model.h5")
net = models.resnet18()
net.load_state_dict(w)
"""""
########################
plt.plot(profile_train, label="train"); plt.plot(profile_val,label="validation");plt.plot(profile_test,label="test"); plt.legend()


