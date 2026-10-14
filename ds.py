import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. DATA  (y = 3x + 2, no noise)
# ==============================================================
x = np.array([[1], [2], [3], [4], [5]])       # shape (5, 1)
y = np.array([[5], [8], [11], [14], [17]])    # shape (5, 1)

# Add bias column
X = np.c_[np.ones((5, 1)), x]                 # shape (5, 2)

# Starting weights
w = np.zeros((2, 1))

# ==============================================================
# 2. CORE FUNCTIONS
# ==============================================================
def predict(X, w):
    return X @ w

def compute_cost(X, y, w):
    m = len(y)
    return (1 / (2 * m)) * np.sum((predict(X, w) - y) ** 2)

def compute_gradient(X, y, w):
    m = len(y)
    error = predict(X, w) - y
    return (1 / m) * (X.T @ error)

def grad_descent(X, y, lr, w_init, n_iters):
    w_curr = w_init.copy()
    cost_history = [compute_cost(X, y, w_curr)]
    for i in range(n_iters):
        g = compute_gradient(X, y, w_curr)
        w_curr -= lr * g
        cost_history.append(compute_cost(X, y, w_curr))
    return w_curr, cost_history

def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# ==============================================================
# 3. TRAIN — both methods
# ==============================================================
w_gd, costs = grad_descent(X, y, lr=0.01, w_init=w, n_iters=1000)
w_ne = normal_equation(X, y)

print("=" * 50)
print("WEIGHTS")
print("=" * 50)
print("Gradient descent :", w_gd.ravel())     # expect ~[2, 3]
print("Normal equation  :", w_ne.ravel())     # expect exactly [2, 3]
print()

# ==============================================================
# 4. METRICS
# ==============================================================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

y_pred = predict(X, w_ne)   # use the exact solution

print("=" * 50)
print("METRICS (using normal equation weights)")
print("=" * 50)
print("MSE  :", mse(y, y_pred))
print("RMSE :", rmse(y, y_pred))
print("R²   :", r2(y, y_pred))
print()

# ==============================================================
# 5. PLOT 1 — Cost vs Iteration
# ==============================================================
plt.figure(figsize=(8, 4))
plt.plot(costs)
plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("Cost vs Iteration (gradient descent)")
plt.grid(True)
plt.show()

# ==============================================================
# 6. PLOT 2 — Actual vs Predicted
# ==============================================================
plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, color="blue", label="data points")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="perfect fit (y=x)")
plt.xlabel("actual y")
plt.ylabel("predicted y")
plt.title("Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================
# 7. PLOT 3 — Residuals vs Predicted
# ==============================================================
residuals = y - y_pred
plt.figure(figsize=(6, 6))
plt.scatter(y_pred, residuals, color="green")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("predicted y")
plt.ylabel("residual (actual - predicted)")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.show()

# ==============================================================
# 8. PLOT 4 — The fitted line on top of the data
# ==============================================================
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="blue", label="data")
x_line = np.linspace(0, 6, 100).reshape(-1, 1)
X_line = np.c_[np.ones((100, 1)), x_line]
y_line = predict(X_line, w_ne)
plt.plot(x_line, y_line, "r-", label="fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Data + Fitted Line")
plt.legend()
plt.grid(True)
plt.show()


#baseline  code notes for  machine learning  block 1  #
#==========================================================#