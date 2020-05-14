import numpy as np
import pandas as pd


def read_data(path_to_data_directory, label_column_name):
    df = pd.read_csv(path_to_data_directory+"heart.csv")
    y = df[label_column_name].to_numpy()
    del df[label_column_name]
    X = df.to_numpy()
    return X, y


def shuffle_data(X, y):
    dataset = np.concatenate((X, y.reshape((y.shape[0], 1))), axis=1)
    np.random.shuffle(dataset)
    return dataset[:, :-1], dataset[:, -1]


def data_split(X, y, ratio):
    """

    @param X:
    @param y:
    @param ratio: number betwen zero and one
    @return:
    """
    X_shuffled, y_shuffled = shuffle_data(X, y)
    index = int(X.shape[0] // (1/ratio))

    X_tr, y_tr, X_te, y_te = X_shuffled[:index], y_shuffled[:index], X_shuffled[index:], y_shuffled[index:]
    return X_tr, y_tr, X_te, y_te


def precision_calc(y_pred, y):
    return np.mean(y == y_pred)


def confusion_matrix(y_pred, y):
    TP, FP, FN, TN = 0, 0, 0, 0
    sum = y + y_pred
    TP = len(sum[sum==2])
    FP = np.sum(y_pred) - TP
    FN = len(sum[sum==0])
    TN = np.sum(y) - FN
    return TP, FP, TN, FN


def classification_report(y_pred, y):
    TP, FP, TN, FN = confusion_matrix(y_pred, y)
    Recall = TP / (TP+FN)
    Precision = TP / (TP + FP)
    Accuracy = (TP + TN) / (TP+TN+FP+FN)
    Specifity = TN/(TN+FP)
    f1score = 2*Recall*Precision/(Recall+Precision)
    return Accuracy, Precision, Recall, Specifity, f1score


def t_test(y_pred_one, y_pred_two, y):
    N = y.shape[0]
    vector_one, vector_two = np.zeros(N), np.zeros(N)
    vector_one[np.where(y_pred_one == y)] = 1
    vector_two[np.where(y_pred_two == y)] = 1

    mu_one = np.mean(vector_one)
    mu_two = np.mean(vector_two)

    vector_one_hat = vector_one - mu_one
    vector_two_hat = vector_two - mu_two

    t = (np.abs(mu_one-mu_two)) * np.sqrt((N*(N-1)) / (np.sum(np.square(vector_one_hat-vector_two_hat))))

    return t >= 1.64




