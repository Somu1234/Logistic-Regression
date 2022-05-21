import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def logistic_regression_cost(x, y, w):
    h = sigmoid(x @ w)
    #print(np.log(h).shape)
    one = np.ones((y.shape[0], 1))
    #compute cost function for logistic regression
    #-(sum(y[i](log(h(x[i]))) + (1 - y[i])(log(1 - h(x[i])))))/n
    #where h(x) is sigmoid function
    return (-((y.T @ np.log(h)) + ((one - y).T @ np.log(one - h)))/(y.shape[0]))

def gradient_descent(x, y, w, alpha = 0.1, epochs = 10):
    m = x.shape[0]
    cost_all = []
    for _ in range(epochs):
        h_x = sigmoid(x @ w)
        cost_ = (1 / m) * (x.T @ (h_x - y))
        w = w - (alpha) * cost_
        cost_all.append(logistic_regression_cost(x, y, w))
    return w, cost_all 

def train(x, y):
    #Add a column of ones to the start of the matrix to account for 3 parameters
    #in weight matrix. Size : (100, 3)
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    #Create row vector from column vector y
    y = np.reshape(y, (y.shape[0], 1))
    #Weight matrix. Size : (3, 1)
    #W0 - decision boundary, W1 - Test 1 and W2 - Test 2
    w = np.zeros((x.shape[1], 1))
    #Converging the model over n epochs
    alpha = 0.001
    epochs = 100
    w, cost_all = gradient_descent(x, y, w, alpha, epochs)
    cost_final = logistic_regression_cost(x, y, w)
    print("Weight matrix of trained model : \n", w)
    print("Final cost value : ", cost_final)
    #Plot Cost vs Epochs
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.plot(np.array(range(1, 101)), np.array(cost_all)[:, 0, 0])
    plt.show()
    #Return weight matrix to be used for prediction
    return w

def predict(w, x):
    prob = float(sigmoid(x @ w))
    if prob >= 0.5:
        print("YES")
        return 1
    else:
        print("NO")
        return 0

if __name__ == "__main__":
    df = pd.read_csv("Dataset/admission_marks.csv")
    #Replace Yes and No with 1/0
    df = df.replace({"Result" : {'YES' : 1, 'NO' : 0}})
    #print(df)

    #Visualise data
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    for i in range(len(df)):
        if df.loc[i, "Result"] == 1:
            plt.plot(df.loc[i, "Test 1"], df.loc[i, "Test 2"], 'gX')
        else:
            plt.plot(df.loc[i, "Test 1"], df.loc[i, "Test 2"], 'y+')
    plt.show()
    #Plot shows that data has a decision boundary

    #Create numpy arrays
    x = np.array(df[["Test 1", "Test 2"]], dtype = float)
    y = np.array(df["Result"], dtype = float)
    #Train the model.
    w = train(x, y)
    
    #Testing for an example random value
    predict(w, [1, 50, 91])
