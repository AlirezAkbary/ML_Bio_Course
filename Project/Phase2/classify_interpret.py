import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from DataUtils import *
from model import *
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def allbutlast(iterable):
    it = iter(iterable)
    current = it.__next__()
    for i in it:
        yield current
        current = i



model = GINConvNet_Regression()
model.load_state_dict(torch.load('model_regression_upsample.pt'))


pretrained_dict = {}
model_dict = model.state_dict()
for x, y in model_dict.items():
    if 'out' not in x:
      pretrained_dict[x] = y


model = GINConvNet_Classification()
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


for child in allbutlast(model.children()):
  for param in child.parameters():
    param.requires_grad = False






device = None
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 50

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

dataset = ['davis', 'kiba'][int(sys.argv[1])]
print(dataset)


model_saved_str = 'model_classificationn_upsample.pt'



train_data = TestbedDataset(root='data', dataset=dataset + '_balanced_train')
val_data = TestbedDataset(root='data', dataset=dataset + '_validation')


train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)



model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
weights = [1.0, 2.0]
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))


best_val_f1 = 0



# writer = SummaryWriter("./runs/train")

for epoch in range(NUM_EPOCHS):
    train_loss = 0

    model.train()


    for iteration, data_object in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data_object.to(device))
        y_true = data_object.y.long().to(device)
        if dataset == "davis":
            y_true = torch.where(y_true > 7, torch.ones(y_true.shape).to(device), torch.zeros(y_true.shape).to(device))
        else:
            y_true = torch.where(y_true > 12.1, torch.ones(y_true.shape).to(device), torch.zeros(y_true.shape).to(device))
        loss = criterion(output, y_true.long().to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data_object.c_size.shape[0]

    train_loss = train_loss / len(train_loader.sampler)

    model.eval()
    total_preds = torch.Tensor()
    total_affinities = torch.Tensor()
    with torch.no_grad():
        for data_object in val_loader:
            output = model(data_object.to(device))
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_affinities = torch.cat((total_affinities, data_object.y.view(-1, 1).cpu()), 0)

    total_affinities = total_affinities.numpy().flatten()
    total_preds = total_preds.numpy()




    preds_classification = np.argmax(total_preds, axis=1)
    if dataset == "davis":
        affinities_classification = np.where(total_affinities > 7, 1, 0)
    else:
        affinities_classification = np.where(total_affinities > 12.1, 1, 0)

    tn, fp, fn, tp = confusion_matrix(affinities_classification, preds_classification).ravel()
    specificity = tn / (tn + fp)
    sensivity = recall_score(affinities_classification, preds_classification) #recall
    accuracy = accuracy_score(affinities_classification, preds_classification)
    f1 = f1_score(affinities_classification, preds_classification)

    if f1 > best_val_f1:
        print('\tValidation f1 increased ({:.6f} --> {:.6f}) at epoch: {}. Saving model ...'.format(best_val_f1, f1, epoch+1))
        best_val_f1 = f1
        torch.save(model.state_dict(), model_saved_str)

    print("Existed numbers of class 1: ", np.sum(affinities_classification==1))
    print("Predicted numbers of class 1: ", np.sum(preds_classification == 1))
    print('Epoch: {} Training Loss: {:.6f}  val acc: {:.6f}  val sensitivity: {:.6f}  val specifity: {:.6f}  val f1: {:.6f}'.format(epoch + 1, train_loss,
                                                                                            accuracy, sensivity, specificity, f1))


test_data = val_data = TestbedDataset(root='data', dataset=dataset + '_test')
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)

model = GINConvNet_Classification()
model.load_state_dict(torch.load(model_saved_str))
model.to(device)
model.eval()
total_preds = torch.Tensor()
total_affinities = torch.Tensor()
with torch.no_grad():
    for data_object in test_loader:
        output = model(data_object.to(device))
        total_preds = torch.cat((total_preds, output.cpu()), 0)
        total_affinities = torch.cat((total_affinities, data_object.y.view(-1, 1).cpu()), 0)

total_affinities = total_affinities.numpy().flatten()
total_preds = total_preds.numpy()

preds_classification = np.argmax(total_preds, axis=1)
if dataset == "davis":

    affinities_classification = np.where(total_affinities > 7, 1, 0)
else:

    affinities_classification = np.where(total_affinities > 12.1, 1, 0)

tn, fp, fn, tp = confusion_matrix(affinities_classification, preds_classification).ravel()
specificity = tn / (tn + fp)
sensivity = recall_score(affinities_classification, preds_classification)  # recall
accuracy = accuracy_score(affinities_classification, preds_classification)
f1 = f1_score(affinities_classification, preds_classification)

print("Existed numbers of class 1: ", np.sum(affinities_classification==1))
print("Predicted numbers of class 1: ", np.sum(preds_classification == 1))
print('Test acc: {:.6f}  Test sensitivity: {:.6f}  Test specifity: {:.6f}  Test f1: {:.6f}'.format(accuracy, sensivity, specificity, f1))
