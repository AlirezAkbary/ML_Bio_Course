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


device = None
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 100

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

dataset = ['davis', 'kiba'][int(sys.argv[1])]
print(dataset)
balanced_inbalanced = int(sys.argv[2])
balanced_inbalanced_str = None
model_saved_str = None
if balanced_inbalanced == 0:
    balanced_inbalanced_str = '_train'
    model_saved_str = 'model_regression_no_upsample.pt'
else:
    balanced_inbalanced_str = '_balanced_train'
    model_saved_str = 'model_regression_upsample.pt'
train_data = TestbedDataset(root='data', dataset=dataset + balanced_inbalanced_str)
val_data = TestbedDataset(root='data', dataset=dataset + '_validation')


train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)


model = GINConvNet_Regression()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = nn.MSELoss()

best_mse = np.Inf
best_ci = 0



# writer = SummaryWriter("./runs/train")

for epoch in range(NUM_EPOCHS):
    train_loss = 0

    model.train()

    for iteration, data_object in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data_object.to(device))
        loss = criterion(output, data_object.y.view(-1, 1).float().to(device))
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
    total_preds = total_preds.numpy().flatten()

    val_mse = mse(total_affinities, total_preds)
    val_ci = ci(total_affinities, total_preds)
    if val_mse < best_mse:
        print('\tValidation mse decreased ({:.6f} --> {:.6f}) at epoch: {}. CI:{:.6f} Saving model ...'.format(best_mse, val_mse, epoch+1, val_ci))
        best_mse = val_mse
        torch.save(model.state_dict(), model_saved_str)
        best_ci = val_ci

    if dataset == "davis":
        preds_classification = np.where(total_preds > 7, 1, 0)
        affinities_classification = np.where(total_affinities > 7, 1, 0)
    else:
        preds_classification = np.where(total_preds > 12.1, 1, 0)
        affinities_classification = np.where(total_affinities > 12.1, 1, 0)

    tn, fp, fn, tp = confusion_matrix(affinities_classification, preds_classification).ravel()
    specificity = tn / (tn + fp)
    sensivity = recall_score(affinities_classification, preds_classification) #recall
    accuracy = accuracy_score(affinities_classification, preds_classification)
    f1 = f1_score(affinities_classification, preds_classification)
    print("Existed numbers of class 1: ", np.sum(affinities_classification==1))
    print("Predicted numbers of class 1: ", np.sum(preds_classification == 1))
    print('Epoch: {} Training Loss: {:.6f}  val acc: {:.6f}  val sensitivity: {:.6f}  val specifity: {:.6f}  val f1: {:.6f}'.format(epoch + 1, train_loss,
                                                                                            accuracy, sensivity, specificity, f1))


print("------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------")
print("Test")

test_data = TestbedDataset(root='data', dataset=dataset + '_test')
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)

model.eval()
total_preds = torch.Tensor()
total_affinities = torch.Tensor()
with torch.no_grad():
    for data_object in test_loader:
        output = model(data_object.to(device))
        total_preds = torch.cat((total_preds, output.cpu()), 0)
        total_affinities = torch.cat((total_affinities, data_object.y.view(-1, 1).cpu()), 0)

total_affinities = total_affinities.numpy().flatten()
total_preds = total_preds.numpy().flatten()

test_mse = mse(total_affinities, total_preds)
test_ci = ci(total_affinities, total_preds)
print("Test mse : ", test_mse)
print("Test ci : ", test_ci)
if dataset == "davis":
    preds_classification = np.where(total_preds > 7, 1, 0)
    affinities_classification = np.where(total_affinities > 7, 1, 0)
else:
    preds_classification = np.where(total_preds > 12.1, 1, 0)
    affinities_classification = np.where(total_affinities > 12.1, 1, 0)

tn, fp, fn, tp = confusion_matrix(affinities_classification, preds_classification).ravel()
specificity = tn / (tn + fp)
sensivity = recall_score(affinities_classification, preds_classification)  # recall
accuracy = accuracy_score(affinities_classification, preds_classification)
f1 = f1_score(affinities_classification, preds_classification)

print("Existed numbers of class 1: ", np.sum(affinities_classification==1))
print("Predicted numbers of class 1: ", np.sum(preds_classification == 1))
print('Test acc: {:.6f}  Test sensitivity: {:.6f}  Test specifity: {:.6f}  Test f1: {:.6f}'.format(accuracy, sensivity, specificity, f1))