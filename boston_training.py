#training  a   price  prediction   model  on boston hosuing datset  


import pandas  as  pd 
import  numpy  as  np  
import matplotlib.pyplot as  plt  

X  =  pd.read_csv("boston_X.csv").values
y  =  pd.read_csv("boston_y.csv").values

np.random.seed(42)
indices  =  np.random.permutation(len(X))

X_shuffled = X[indices]
y_shuffled = y[indices]


split = int(0.8*len(X))

X_train = X_shuffled[:split]
X_test = X_shuffled[split:]
y_train = y_shuffled[:split]
y_test = y_shuffled[split:]


print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

mu =   X_train.mean(axis=0)
sigma = X_train.std(axis=0)

X_train =   (X_train-mu)/sigma
X_test = (X_test-mu)/sigma 

print("X_train mean (should be ~0):", X_train.mean(axis=0).round(3))
print("X_train std  (should be ~1):", X_train.std(axis=0).round(3))
print()
print()
print("X_test mean  (close to 0, not exact):", X_test.mean(axis=0).round(3))
print("X_test std   (close to 1, not exact):", X_test.std(axis=0).round(3))


# X_train  =  np.c_[np.ones((len(X_train),1)),X_train]
# X_test   =  np.c_[np.ones((len(X_test),1)),X_test]

# print("X_train with bias:",X_train.shape)
# print("X_test with bias:",X_test.shape )

# Add bias column (column of ones) to both train and test
X_train = np.c_[np.ones((len(X_train), 1)), X_train]
X_test  = np.c_[np.ones((len(X_test),  1)), X_test]

print("X_train with bias:", X_train.shape)
print("X_test  with bias:", X_test.shape)













w = np.zeros((13,1))
def predict(X,w):
    return X @ w
     

def compute_cost(X,y,w):
    m =  len(y)
    cost  =  (1/(2*m))*np.sum(( predict(X,w)-  y)**2)
    return cost  
#print(compute_cost(X,y,w))

def compute_gradient(X,y,w):
    m =  len(y)
    error = ((predict(X,w))- y ) 
    gradient =  (1/m)*(X.T @ error)
    return gradient  
#print(compute_gradient(X,y,w))

#grad descent loop 
# w:= w - α∇J
def grad_descent(X,y,lr,w_init,n_iters):
    w_curr =  w_init.copy()
    cost_history  = []
    cost_history.append(compute_cost(X,y,w_curr))
    for i  in range(n_iters):
        g =  compute_gradient(X,y,w_curr)
        w_curr -= lr*g   
        cost_history.append(compute_cost(X,y,w_curr))
    return w_curr,cost_history
# lr = 0.001
# w_final,costs =  grad_descent(X,y,lr,w_init = w , n_iters=1000)
# print("w_final:",w_final.ravel())
# print("First cost:",costs[0])
# print("Last cost:",costs[-1])


# print("global w :",w.ravel())

# plt.plot(costs)
# plt.xlabel("iteration")
# plt.ylabel("cost")
# plt.title("cost vs iteration")
# plt.show()


def normal_equation(X,y):
    return np.linalg.inv(X.T@X)@X.T @ y 


W_ne  =  normal_equation(X_train,y_train)
print("Normal Equation:",W_ne.ravel())
print()

def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)
def rmse(y_true,y_pred):
    return np.sqrt(mse(y_true,y_pred))
def r2(y_true,y_pred):
    ss_res =  np.sum((y_true - y_pred)**2)
    ss_tot  =  np.sum((y_true-np.mean(y_true))**2)
    return 1  - (ss_res/ss_tot)
# y_pred  = predict(X,W_ne)
# print("="*50)
# print("METRICS(using normmal equation weights)")

# print("MSE:",mse(y,y_pred))
# print("RMSE:",rmse(y,y_pred))
# print("R²   :",r2(y,y_pred))
# print()



w  =  np.zeros((14,1))
w_final ,costs =  grad_descent(X_train,y_train,lr = 0.01,w_init = w ,n_iters = 2000)

print("w_final shape:",w_final.shape)
print("First cost:",costs[0])
print("last cost:",costs[-1])


y_pred_train =  predict(X_train,w_final)
y_pred_test = predict(X_test,w_final)


print("="*40)
print("TRAIN METRICS")
print("="*40)

print("MSE:",mse(y_train, y_pred_train))
print("R²:",r2(y_train,y_pred_train))
print("RMSE:",rmse(y_train,y_pred_train))


print()
print("="*50)
print("TEST METRICS")
print("="* 50)
print("MSE:",mse(y_test,y_pred_test))
print("R²:",r2(y_test,y_pred_test))
print("RMSE:",rmse(y_test,y_pred_test))



#cost  vs  iteration    
plt.figure(figsize=(8,4))
plt.plot(costs)
plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("cost vs iteration")
plt.grid(True)
plt.show()


#Actual  vs  Predicted (test set )
plt.figure(figsize=(6,6))
plt.scatter(y_test,y_pred_test,alpha= 0.5)


plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()], "r--")

plt.xlabel("actual y")
plt.ylabel("predicted y")
plt.title("Actual vs Predicted (test set )")
plt.grid(True)
plt.show()


#residuals vs predicted(test_set)
residuals = y_test -  y_pred_test
plt.figure(figsize=(6,6))
plt.scatter(y_pred_test,residuals,alpha = 0.5,color ="green")
plt.axhline(0,color = "red",linestyle = "--")
plt.xlabel("predicted y ")
plt.ylabel("residual")
plt.title("Residuals vs Predicted (test set)")
plt.grid(True)
plt.show()


    
