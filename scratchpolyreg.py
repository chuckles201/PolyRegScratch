import numpy as np
import torch
import pandas as pd
import pdb
import matplotlib.pyplot as plt
d = int(input("To what degreee we guessin? :"))

# Generating random polynomial data
n_vars, n_samples = 1, 5000

X = np.random.randint(0,30,size=[n_samples,n_vars]).astype(float) # x of size vars + 1xsamples (adding bias)


#y = 5*(X**3) + 10*(X**2) + 100*X + 1*(X**4) - 0.04*(X**5)
y = 0.5*np.cos(X)
print(X,y)
plt.scatter(X,y)
plt.show()


# gradient descent
'''
Polynomial regression is linear regression, with all possiblepolynomial features appended to the samples 
to the nth degree. More degrees and more variables will lead to exponentially more terms. 
For example, if i wanted to fit a curve that i beleived to be up to the second degre (y= ~x^2) then i would
square the column vector X1 which holds all of this variable, and fit this too a linear model. The model
will be able to then assign a weight to the slope with x^2. Think of a linear graph of y=x^2 against x^2.

Matrix X will be samplesxvariables

Here is a messsily thrown together examples that works on 3 degrees with no intercept ;)
'''

def poly_reg(X,y,L,epochs,degree=2):
    # adding new feature
    x_og = X.copy()
    theta = np.zeros((X.shape[1] * degree) )
    stds = np.zeros(theta.shape[0])
    means = stds.copy()
    
    for i in range(degree):
        if i > 0:
            X = np.concatenate([x_og**(i + 1),X],axis=1)
            
    for col in range(X.shape[1]):
        stds[col] = np.std(X[:,col])
        means[col] = np.mean(X[:,col])
        st = (X[:,col] - np.mean(X[:,col])) / np.std(X[:,col])
        X[:,col] = st
    #X = np.concatenate([x_og**0,X],axis=1)
    #stds = np.append(1,stds)
    for i in range(epochs):
        theta_grad = X.T @ ((X @ theta) - y.reshape(y.shape[0]))
        theta += -L*theta_grad
    print(means)
    theta = theta/stds # re-standardizing...
    
    
    return theta
    

theta_answers = poly_reg(X,y,0.00005,10000,d)

# plotting answers
print(f'Model Answers: {theta_answers}\nReal Answers: (-3000,2,3,0.5)')

#plt.scatter(X,y)
def answer(x):
    a= 0
    for i in range(d):
        a += (x**(d-i)) * (theta_answers[i])
    return a
xs = np.linspace(0,30,1000)
vf = np.vectorize(answer)
ys = vf(xs)
plt.scatter(X,y)
plt.plot(xs,ys,color='red')
plt.show()