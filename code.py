import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


np.random.seed(42)

W1 = np.random.randn(2,2)
b1 = np.random.randn(1,2)

W2 = np.random.randn(2,1)
b2 = np.random.randn(1,1)

lr = 0.1

for epoch in range(10000):

    

    Z1 = np.dot(X,W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1,W2) + b2
    A2 = sigmoid(Z2)

    loss = np.mean((y-A2)**2)



    dA2 = A2 - y

    dZ2 = dA2 * sigmoid_derivative(A2)

    dW2 = np.dot(A1.T,dZ2)
    db2 = np.sum(dZ2,axis=0,keepdims=True)

    dA1 = np.dot(dZ2,W2.T)

    dZ1 = dA1 * sigmoid_derivative(A1)

    dW1 = np.dot(X.T,dZ1)
    db1 = np.sum(dZ1,axis=0,keepdims=True)

    # update

    W1 -= lr*dW1
    b1 -= lr*db1

    W2 -= lr*dW2
    b2 -= lr*db2

    if epoch % 1000 == 0:
        print(epoch, loss)

print("\nPredictions:")
print(A2)
