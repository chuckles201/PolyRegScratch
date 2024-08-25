# Objective/ Plan
'''We are going to create a polynomial regression from scratch!
Here are the general steps: (using pytorch (GPU)!)

1. Generating good, random polynomial data to test on
2. Creating model that can create a polynomial function for k variables to the nth degree (w/feature combos)
    - compute all possible features
    - normalize / add biases
    - gradient descent algorithm (batch?), some standardiszation techniques with learning schedule/rates
    - train model, return un-standardized weights
    - stichting together funcs into a class

3. Plot the results in lower dimensions, and then test in higher dimensions (real data!)!


'''
import torch


''' COMMENT OUT
# checking device used (fancy)
import tkinter
import tkinter.messagebox

def question():
    response = tkinter.messagebox.askquestion("Get ready...","Would you like to use the gpu?")
    return response

res = question()
if res == 'yes':
    print("Using cuda")
    device = torch.device('cuda')
else:
    print("Using cpu")
    device = torch.device('cpu')
'''
device= torch.device('cpu')
print(torch.cuda.is_available())


# TESTING  STEP: generating polynomial data (create a tensor of data?)
'''
1. Create dataset and of all possible combinations to dth degree (with bools to determine which ones will show)
2. Multiply by random weights and add noise, 
3.then create function to output final func in standard notation.
'''

import numpy as np
import itertools

# creating original X matrix (samples) 
n_samples,n_vars, degree_og = 20000,2,2
X = torch.randn(size=[n_samples,n_vars]) * 10
# generate combinations to degree
combo_holder = []
bool_holder = []
for d in range(degree_og):
    combo_holder.append(np.array(list(itertools.combinations_with_replacement(range(n_vars),degree_og - d))))
    

# choosing which combinations to actually use (1/3 chance)
for i in range(len(combo_holder)):
    for i2 in range(len(combo_holder[i])):
        bool = np.random.randint(0,4)
        if bool == 1:
            bool_holder.append(True)
        else:
            bool_holder.append(False)
# assigning random weights for each features combo thats being used
weights = np.random.randn(len(bool_holder)) * 5
weights_flat = weights.flat.copy()
weights_flat[bool_holder] = 0
weights = weights_flat.reshape(weights.shape)

weights = np.append(-120,weights) # arbitrary bias!

# giving X new features
X_new = torch.ones(X.shape[0],1) # adding bias
for i in range(len(combo_holder)):
    for i2 in range(len(combo_holder[i])):
        colsto_multiply = X[:,combo_holder[i][i2]]
        X_new = torch.concat([X_new,torch.prod(colsto_multiply,dim=1).reshape(-1,1)],dim=1)


# creating output vector (w/noise)!
weights = torch.tensor(weights, dtype=torch.float32).reshape(-1,1)

y = torch.matmul(X_new,weights)
noise = torch.randn(y.shape)
y += (noise * 10)
y = y.to(device=device)
print(f"Feature Combos: \n{combo_holder}\n\nWeights for combos([1:]) : \n{weights[1:]}\n\nBias:{weights[0]}\n\n")



# creating all possible features given 'X' only using torch and itertools for getting the coefeccients...
'''*Explanation of functions handeled by iter tools*:
- Combos
- torch.prod
'''
def poly_features(X,degree,bias=True):
    if isinstance(X,torch.Tensor):
        pass
    else:
        X = torch.tensor(X,device=device)
        
    m,n = X.shape
    holder = []
    
    for i in range(degree): # step 1
        # creating combos for m cols and all degrees up to d
        comb = np.array(list(itertools.combinations_with_replacement(range(n),(degree-i))))
        
        holder.append(comb) 
        
    x_poly = torch.ones(X.shape[0],1,device=device)
    for deg in range(len(holder)): # step 2 (creating polynomial features)
        for combo in range(len(holder[deg])):
            # multiply combos selections of variables by eachother (row-wise for sample)
            multip_col = torch.clone(X[:,holder[deg][combo]]).to(device)# (duplicates allowed in indexing of cols)
            x_poly = torch.cat([x_poly,torch.prod(multip_col,dim=1).reshape(-1,1)],dim=1)

    # step 3 (adding intercept)
        
    return x_poly.to(device)


b = poly_features(X,2,True)



# gradient descent with normalization
'''
Finding path of optimal descent in parameter space (think of convex function)
'''
def grad_descent(X,y,epochs,L):
    m,n = X.shape
    if isinstance(X,torch.Tensor):
        pass
    else:
        X = torch.tensor(X,device=device)
        
    theta = torch.zeros(n,1,device=device)  
    theta_og = torch.zeros(n,1,device=device) 
    
    for e in range(epochs):
        error1 = torch.matmul(X,theta) - y
        theta_grad = torch.matmul(X.T, error1).reshape(-1,1)
        theta = theta - (theta_grad * L)
    return theta, torch.linalg.vector_norm(torch.matmul(X,theta) - y), torch.linalg.vector_norm(torch.matmul(X,theta_og) - y)


# with enough epochs it works! now for normalization...
import time 
y = y.to(device)
stds = []
means = []
for col in range(b.shape[1]):
    if col != 0:
        means.append(torch.mean(b[:,col]))
        print(torch.mean(b[:,col]))
        print(b[:,col])
        stds.append(torch.std(b[:,col]))
        b[:,col] = (b[:,col] - torch.mean(b[:,col])) / torch.std(b[:,col])

t1 = time.time()
answer = grad_descent(b,y,2000,0.00009)
t2 = time.time() - t1

print(f'Time :{t2}')
answer_weights = []
for i in range(len(answer[0])):
    if i != 0:
        answer_weights.append(answer[0][i] / stds[i-1])
        print(f"Theta {i} :{(answer[0][i] / stds[i-1]).item()}")
    else: 
        pass

answer_array = [answer.item() for answer in answer_weights]
means_array = [mean.item() for mean in means]
print(f'Bias : {answer[0][i].item() - np.dot(means_array,answer_array)}')
print(f"\nMSE Loss: {answer[1]}")
print(f"Original Loss: {answer[2]}")




