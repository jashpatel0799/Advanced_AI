import numpy as np
import pandas as pd
import sys
import subprocess
import math
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error


# create own multi-valued regression for phi1 and phi2 from scratch using only numpy
class multipleLinearRegression():

  def __init__(self):
    #No instance Variables required
    pass

  def forward(self,X,y,W):
    
    y_pred = sum(W * X)
    loss = ((y_pred-y)**2)/2    #Loss = Squared Error, we introduce 1/2 for ease in the calculation
    return loss, y_pred

  def updateWeights(self,X,y_pred,y_true,W,alpha,index):
    
    for i in range(X.shape[1]):
      #alpha = learning rate, rest of the RHS is derivative of loss function
      W[i] -= (alpha * (y_pred-y_true[index])*X[index][i])
    
    return W

  def train(self, X, y, epochs=200, alpha=0.001, random_state=42):
   
    num_rows = X.shape[0] #Number of Rows
    num_cols = X.shape[1] #Number of Columns
    np.random.seed(random_state) 
    # W = np.array([[0.18, -0.1, 15]]) #Weight Initialization
    W = np.array([[0.18, -0.27, 15]]) #Weight Initialization
    # print(W)
    #Calculating Loss and Updating Weights
    train_loss = []
    num_epochs = []
    train_indices = [i for i in range(X.shape[0])]
    for j in range(epochs):
      cost=0
      np.random.seed(random_state)
      np.random.shuffle(train_indices)
      for i in train_indices:
        loss, y_pred = self.forward(X[i],y[i],W[0])
        cost+=loss
        W[0] = self.updateWeights(X,y_pred,y,W[0],alpha,i)
      train_loss.append(cost)
      num_epochs.append(j)
    return W[0], train_loss, num_epochs



def get_phis(input_series):
    x1 = [i for i in range(len(input_series))]
    x2 = [0]
    x3 = [1 for k in range(len(input_series))]

    for j in range(len(input_series)-1):
        x2.append(j)

    X = []
    X.append(x1)
    X.append(x2)
    X.append(x3)

    X = np.array(X)
    # X = np.concatenate((X,np.ones((len(input_series),1))), axis = 1)
    y = np.array(input_series)

    # print(X)
    # print(X.T.shape)
    regressor = multipleLinearRegression()

    W_trained, train_loss, num_epochs = regressor.train(X.T, y, epochs=200, alpha=0.0001)

    # print(W_trained)

    

    return W_trained[0], W_trained[1], W_trained[2]

def ARIMA_Forecast(input_series:list, P: int, D: int, Q: int, prediction_count: int)->list:
    # Complete the ARIMA model for given value of input series, P, Q and D
    # return n predictions after the last element in input_series
    # print(input_series)
    # print(P)
    # print(D)
    # print(Q)
    # print(prediction_count)
    series = input_series.copy()

    new_series = input_series.copy()

    for i in range(D,len(new_series)):
       new_series[i] = new_series[i] - 2*new_series[i-1] + new_series[i-2]
    

    # print(new_series[2:], )
    # phi1, phi2 = 1.69048535, -0.7718199
    phi1, phi2, constant = get_phis(input_series[D:])
    # phi1 = 0.42#0.45#0.45
    # phi2 = -0.27#-0.30#-0.35

    thetas = []
    for i in range(Q):
        thetas.append(phi1**(i+1))

    for i in range(prediction_count):
        AR_ = 0
        AR = 0
        MA_ = 0
        MA = 0
        # constant = 0
        # for val in series:
        #     constant += val
        # constant /= len(series)

        AR_ = constant + (phi1*series[-1]) + (phi2*series[-2])

        AR = AR_ + (AR_ - series[-1])

        epsilons = []
        epsilons.append(AR - series[-1])
        for i in range(Q-1):
            epsilons.append(series[-(i+1)] - series[-(i+2)])
        # print(epsilons)

        MA_ = 0
        for i in range(Q):
            MA_ += (thetas[i]*epsilons[i])

        MA = constant + MA_

        series.append(AR+MA)

    # print(series)
    
    return series[-(prediction_count+1):] 
    # This is wrong, but it does create a list of the required size, 
    # and is used to showcase the `Plot()` function in the test program 


def HoltWinter_Forecast(input:list, alpha:float, beta:float, gamma:float, seasonality:int, prediction_count: int)->list:
    # Complete the Holt-Winter's model for given value of input series
    # Use either of Multiplicative or Additve model
    # return n predictions after the last element in input_series

    level = []
    trend = []
    seasonal = []

    level.append(input[0] * alpha)
    trend.append(beta * level[0])
    seasonal.append(gamma * (input[0]))

    for i in range(1, len(input)):
        if i > seasonality:
            # level.append((alpha * (input[i] - seasonal[i-seasonality])) + ((1-alpha) * (level[i-1] + trend[i-1])))
            level.append((alpha * (input[i]/seasonal[i-seasonality])) + ((1-alpha) * (level[i-1] + trend[i-1])))
        else:
            # level.append((alpha * input[i]) + ((1-alpha) * (level[i-1] + trend[i-1])))
            level.append((alpha * (input[i])) + ((1-alpha) * (level[i-1] + trend[i-1])))

        trend.append((beta * (level[i] - level[i-1])) + ((1-beta) * (trend[i-1])))
    
        if i > seasonality: 
            # seasonal.append((gamma * (input[i] - level[i-1] - trend[i-1])) + ((1-gamma) * seasonal[i-seasonality]))
            seasonal.append((gamma * (input[i]/(level[i-1] + trend[i-1]))) + ((1-gamma) * (seasonal[i-seasonality])))
        else:
            # seasonal.append(gamma * (input[i] - level[i-1] - trend[i-1]))
            seasonal.append(gamma * (input[i]/(level[i-1] + trend[i-1])))

    
    series = input.copy()
    c = len(input) - 1

    # print(min(level))
    # print(min(trend))
    # print(min(seasonal))
    for i in range(prediction_count):
    #    series.append(level[-1] + 1 * trend[-1] + seasonal[len(seasonal) -1 + 1 - seasonality])
       series.append((level[-1] + 1*trend[-1]) * seasonal[len(seasonal) - 1 + 1 - seasonality])

    #    level.append((alpha * (series[c+i] - seasonal[c+i-seasonality])) + ((1-alpha) * (level[c+i-1] + trend[c+i-1])))
       level.append((alpha * (series[c+i]/seasonal[c+i-seasonality])) + ((1-alpha) * (level[c+i-1] + trend[c+i-1])))

    #    trend.append((beta * (level[c+i] - level[c+i-1])) + ((1-beta) * (trend[c+i-1])))
       trend.append((beta * (level[c+i] - level[c+i-1])) + ((1-beta) * (trend[c+i-1])))
       

    #    seasonal.append((gamma * (series[c+i] - level[c+i-1] - trend[c+i-1])) + ((1-gamma) * seasonal[c+i-seasonality]))
       seasonal.append((gamma * (series[c+i]/(level[c+i-1] + trend[c+i-1]))) + ((1-gamma) * (seasonal[c+i-seasonality])))


    # print(level)
    # print()
    # print(trend)
    # print()
    # print(seasonal)
    # print()

    # print(series[-(prediction_count+1):])

    return series[-(prediction_count+1):]
    # return [0] * prediction_count
    # This is wrong, but it does create a list of the required size, 
    # and is used to showcase the `Plot()` function in the test program 


def ARIMA_Paramters(input_series:list)->tuple: # (P, D, Q)
    # install state model for ARIMA 
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'statsmodels', '--quiet'])
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn', '--quiet'])

    p_values = 2
    d_values = range(0, 2)
    q_values = range(0, 2)

    # for p in p_values:
    best_order = None
    Min_MSE = 1e7
    for d in d_values:
        for q in q_values:
            order = (2,d,q)
            # warnings.filterwarnings("ignore")
            input_series = np.array(input_series)
            train_data = pd.DataFrame(input_series[:math.floor(len(input_series)*0.75)])
            test_data = pd.DataFrame(input_series[math.floor(len(input_series)*0.75):])
            mod = sm.tsa.arima.ARIMA(train_data, order=order)
            model = mod.fit()
            predictions = model.predict(start=len(train_data), end=len(train_data) + len(test_data)-1)
            error = mean_squared_error(test_data, predictions)
            
            if Min_MSE > error:
               Min_MSE = error
               best_order = order
               
    print('ARIMA %s MSE=%.3f' % (best_order,Min_MSE))

    return best_order

def HoltWinter_Parameters(input_series:list)->tuple: # (Alpha, Beta, Gamma, Seasonality)

    alpha = [0.3, 0.7, 0.5]
    beta = [0.2, 0.6, 0.8]
    gamma = [0.25, 0.35, 0.65]
    seasonality = [2, 3, 5]
    

    best_order = None
    Min_MSE = 1e7

    for a in alpha:
       for b in beta:
          for g in gamma:
             for s in seasonality:
                order = (a, b, g, s)
                # warnings.filterwarnings("ignore")
                input_series = np.array(input_series)
                train_data = pd.DataFrame(input_series[:math.floor(len(input_series)*0.75)])
                test_data = pd.DataFrame(input_series[math.floor(len(input_series)*0.75):])
                mod = ExponentialSmoothing(train_data)
                model = mod.fit(a, b, g, s)
                predictions = model.predict(start=len(train_data), end=len(train_data) + len(test_data)-1)
                error = mean_squared_error(test_data, predictions)
                
                if Min_MSE > error:
                    Min_MSE = error
                    best_order = order

    
    print('HoltWinter %s MSE=%.3f' % (best_order,Min_MSE))

    return best_order

'''
    # add this code in tests.py file at the last of main function for part 3
    # update in tests for 3rd question
        S = []
        S.append(S1)
        S.append(S2)
        S.append(S3)
        S.append(S4)
        S.append(S5)

        for i in range(0,5):
            P, D, Q = forecasting.ARIMA_Paramters(S[i])
            ARIMA_S = forecasting.ARIMA_Forecast(S[i], P, D, Q, 20)
            Plot(S[i], ARIMA_S)

        for i in range(0,5):
            Alpha, Beta, Gamma, Seasonality = forecasting.HoltWinter_Parameters(S[i])
            HoltWinters_S = forecasting.HoltWinter_Forecast(S[i], Alpha, Beta, Gamma, Seasonality, 20)
            Plot(S[i], HoltWinters_S)
'''