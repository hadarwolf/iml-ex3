import numpy as np
import torch
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.w = None

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5) # transform the labels to -1 and 1, instead of 0 and 1.

        ########## YOUR CODE HERE ##########

        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv

        ####################################
        if len(Y.shape) == 1:
            Y = Y.reshape(1, -1)
        X_transpose = X.T
        N =X_transpose.shape[1]
        d = X_transpose.shape[0]
        ones_row = np.ones((1, N))
        X_bar = np.vstack([X_transpose, ones_row])
        I = np.eye(d+1)
        term_1 = (np.dot(X_bar, X_bar.T)/N) + (self.lambd * I)
        term_2 = np.dot(X_bar, Y.T)/N
        term_i_inv = np.linalg.inv(term_1)
        self.w = np.dot(term_i_inv, term_2)

    def predict(self, X):
        # 1. Prepare X to match the weights (d x N)
        # If X is (N, d), we flip it.
        # We check X.shape[1] because usually X is (N, 2) here.
        if X.shape[0] > X.shape[1]: 
             X = X.T

        # 2. Add the Bias Row (Row of 1s)
        # X is now (2, N). We need (3, N).
        N = X.shape[1]
        ones_row = np.ones((1, N))
        X_bar = np.vstack([X, ones_row])

        # 3. Calculate Scores
        # FIX IS HERE: Use .T to flip weights
        scores = self.w.T @ X_bar  

        # 4. Threshold and Flatten
        preds = np.where(scores >= 0, 1, -1).flatten()
        
        # 5. Convert to 0/1
        return (preds + 1) / 2






class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########

        # define a linear operation.
        self.linear = nn.Linear(input_dim, output_dim)
        ####################################
        pass

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        ########## YOUR CODE HERE ##########

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.
        return self.linear(x)

        ####################################

        pass

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x



