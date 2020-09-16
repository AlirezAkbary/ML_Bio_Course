from copy import deepcopy
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

class MLP:
    def __init__(self, lr = 1e-3):
        input_size = 4
        hidden_size = 3
        output_size = 3
        self.hist = {'loss':[], 'acc':[]}
        
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        
        self.lr = lr

    def softmax(self, x):
        scaled = x - np.max(x, axis = 1, keepdims = True)
        
        exps = np.exp(scaled)
        return exps / (np.sum(exps, axis=1, keepdims=True))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def cross_entropy(self, y, o):
        num_samples = y.shape[0]
        loss = -1 * np.sum(np.log(o[np.arange(num_samples), y] + 1e-5))
        return loss
    
    def forward(self, x):
        h1 = self.sigmoid(x @ self.W1 + self.b1)
        o = self.softmax(h1 @ self.W2 + self.b2)
        return h1, o

    def backward(self, x, y, o, h1):
        num_samples = y.shape[0]
        
        d_logits = deepcopy(o)
        d_logits[np.arange(num_samples), y] -= 1
        
        dW2 = h1.T @ d_logits
        db2 = np.sum(d_logits, axis=0)
        dh1 = d_logits @ self.W2.T
        
        dsigmoid = h1 * (1 - h1)
        dW1 = x.T @ dsigmoid
        db1 = np.sum(dsigmoid, axis=0)
        #print("dW2 : ", dW2)
        #print("dW1 : ", dW1)
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
    
    def train(self, x, y, epochs):
        for epoch in tqdm(range(1, epochs+1)):
            h1, o = self.forward(x)
            self.backward(x, y, o, h1)
            loss = self.cross_entropy(y, o)
            acc = accuracy_score(y, np.argmax(o, axis=1))
            self.hist['loss'] += [loss]
            self.hist['acc'] += [acc]
            print(epoch, 'loss:', loss, 'acc:', acc)