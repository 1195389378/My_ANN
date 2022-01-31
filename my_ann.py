import numpy as np
import matplotlib.pyplot as plt


# m denotes no. samples
# n_k denotes no. neurons in layer k

def load_training_set(path):
    '''
    return:
    X with dimension (no.features, m)
    labels with dimension (no.classes, m)
    '''
    
    #randomly initialize just for test:
    X = np.random.randn(100, 30)
    labels = np.concatenate((np.ones((1, 30)), np.zeros((3, 30))), axis = 0)
    m = X.shape[1]
    return X, labels, m

def initialize_network():

    FD = list(map(int,input('feature dimension of your input (e.g. 24*24*3) : ').split('*')))
    print()
    features=1
    for i in range(len(FD)):
        features *= FD[i]
    assert (features>0)
    
    depth = int(input('depth of your network (including the softmax layer and excluding input layer, e.g. 5) : '))
    print()
    N = [features] + list(map(int,input('no. neurons for each of the layer (the last number is no.classes, e.g. 3 4 5 6 7 4) : ').split()))
    print()
    assert (len(N)==depth+1)
    
    W = []
    B = []
    for i in range(1,depth+1):
        # w for layer L should be of dimension (n_L, n_L-1)
        w = np.random.randn(N[i], N[i-1])
        W.append(w)
        b = np.random.randn(N[i], 1)
        B.append(b)
    assert(len(W) == len(B) and len(B) == depth)
    return depth, W, B
    
    #worth noting that, W[i-1] is the parameters of layer_i
    
def relu(X):
    '''
    input:
    X is the input matrix with dimension (n_L, m)
    
    output:
    Y is the computing result of the activation with dimension (n_L, m)
    derivative is of same dimension
    '''
    Y = np.select([X>0, X<=0],[X, 0])
    derivative = np.select([X>0, X<=0], [1, 0])
    return Y, derivative

def leaky_relu(X):
    '''
    input:
    X is the input matrix with dimension (n_L, m)
    
    output:
    Y is the computing result of the activation with dimension (n_L, m)
    derivative is of same dimension
    '''
    Y = np.select([X>=0, X<0],[X, 0.02*X])
    derivative = np.select([X>=0, X<0], [1, 0.02])
    return Y, derivative

def softmax(X, label):
    '''
    input:
    X is the input matrix with dimension (n_L, m)
    label is the ground truth of training set
    
    output:
    Y is the computing result of the activation with dimension (n_L, m)
    derivative is of same dimension
    '''
    
    X-= np.max(X) # to prevent from overflow
    X = np.exp(X)
    Y = X / np.sum(X, axis=0, keepdims=True)
    dZ_L = Y-label
    return Y, dZ_L

def loss(Y, labels):
    '''
    input:
    Y,labels is of dimension (no.classes, m)
    output a real number
    '''
    return -np.sum(np.multiply(np.log(Y), labels))

def forprop(A_pre, W_L, b_L, activation, labels = None):
    '''
    input:
    A_pre is the input X of layer L  with dimension(n_L-1, m)
    W_L is the parameters W of layer L with dimension(n_L, n_L-1)
    b_L is the bias vector of layer L with dimension(n_L, 1)
    
    output:
    A_L is the output Y of layer L with dimension(n_L, m)
    Z_L is the computing result before activation function with dimension (n_L, m)
    '''
    
    Z_L = np.dot(W_L, A_pre) + b_L
    if labels is None:
        A_L , derivative = activation(Z_L)
        return Z_L, A_L, derivative
    else:
        Y, dZ_L = activation(Z_L, labels)
        return Y, dZ_L
    
    

def backprop(m, A_pre, W_L, derivative_L=None, dA_L=None, dZ_L=None):
    '''
    input: 
    A_pre is the input X of layer L with dimension (n_L-1, m)
    W_L is the parameters W in layer L with dimension (n_L, n_L-1)
    dA_L is the gradient of layer L with dimension (n_L, m)
    derivative_L is the derivative of the activation function in layer L with dimension (n_L, m)
    either input derivative and dA_L or input dZ_L
    
    output: 
    dA_pre is the gradient of layer L-1 with dimension (n_L-1, m)
    dW_L is the gradient of parameters with dimension (n_L, n_L-1)
    db_L is the gradient of bias vector with dimension (n_L, 1)
    '''
    
    assert (dZ_L is not None or (derivative_L is not None and dA_L is not None))
    if dZ_L is None:
        dZ_L = np.multiply(dA_L, derivative_L)
    dA_pre = np.dot(W_L.T, dZ_L)
    dW_L = np.dot(dZ_L, A_pre.T)/m
    db_L = np.sum(dZ_L, axis=1, keepdims=True)/m
    
    return dA_pre, dW_L, db_L

def update_par(W, b, dW, db, learning_rate):
    W -= learning_rate*dW
    b -= learning_rate*db
    return W, b

def draw_loss(lossList):
    plt.plot([i for i in range(len(lossList))], lossList)
    plt.show()

def main(learning_rate, iterations):
    path = ''
    X, labels, m = load_training_set(path)
    
    depth, W, B = initialize_network()
    
    lossList = [] # for storage of loss in each epoch
    
    for _ in range(iterations):
        #forprop
        cache_A = [X]
        cache_d = []
        cache_Z = []
        for layer in range(1,depth+1):
            if layer !=  depth:
                if layer == 1:
                    Z_L, A_L, derivative = forprop(X, W[layer-1], B[layer-1], leaky_relu)
                else:
                    Z_L, A_L, derivative = forprop(A_L, W[layer-1], B[layer-1], leaky_relu)
                cache_A.append(A_L)
                cache_d.append(derivative)
                cache_Z.append(Z_L)
            else:
                Y, dZ_L = forprop(A_L, W[layer-1], B[layer-1], softmax, labels)   
        cost = loss(Y, labels)
        lossList.append(cost)   
          
        #backprop
        for layer in range(depth,0,-1):
            if layer == depth:
                dA_pre, dW_L, db_L = backprop(m, cache_A[depth-1], W[depth-1], derivative_L=None, 
                                              dA_L=None, dZ_L=dZ_L)
            else:
                dA_pre, dW_L, db_L = backprop(m, cache_A[layer-1], W[layer-1], dA_L=dA_pre, 
                                              derivative_L=cache_d[layer-1])
            
            #update parameters
            W[layer-1], B[layer-1] = update_par(W[layer-1], B[layer-1], dW_L, db_L, learning_rate)
    
    assert(len(lossList)>0)
    draw_loss(lossList)

main(0.01, 50) #learning rate and epochs
        