import numpy as np
class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr 
        self.n_iters = n_iters 
        self.weights = None 
        self.bias = None 
        self.loss_history = []
    
    def compute_gradients(self, a, y):
        n_samples = a.shape[0]
        y_pred =  self.predict(a) 
        error = y_pred - y
        dw = (1/n_samples) * np.dot(a.T, error)
        db = (1/n_samples) * np.sum(error)
        return dw, db
    
    def fit(self, a, y): 
        n_samples, n_features = a.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            dw, db = self.compute_gradients(a, y)
            self.weights = self.weights - (self.lr*dw)
            self.bias = self.bias - (self.lr*db)
            loss = self.mse(a, y)
            self.loss_history.append(loss)
            if i % 100 == 0:
                print(f'Loss{loss:4f}')
            


    def predict(self, a):
        y_pred = np.dot(a,self.weights) + self.bias
        return y_pred


    def mse(self, a,y):
        n_samples = a.shape[0]
        y_pred = self.predict(a)
        error = y_pred - y
        return 1/n_samples * np.sum(error)**2

    def score(self,a,y):
        y_pred = self.predict(a)

        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred )**2)
        r2 = 1 - (ss_residual / ss_total)
        return r2


class StandardScaler:
    def __init__(self):
        self.std = None
        self.mean = None 
    
    def fit(self, a):
        a = np.array(a)
        self.mean = np.mean(a, axis = 0)
        self.std = np.std(a, axis = 0)

    def transform(self, a):
        a = np.array(a)
        return (a - self.mean) / self.std

    def fit_transform(self, a):
        a = np.array(a)
        self.fit(a)
        return self.transform(a)

    def inverse_transform(self, x_scaled):
        return(x_scaled * self.std) + self.mean    

