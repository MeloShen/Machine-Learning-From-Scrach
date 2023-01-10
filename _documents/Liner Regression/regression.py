import numpy as np
import matplotlib.pyplot as plt

# Read the traning data
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]
# Draw the graph
#plt.plot(train_x,train_y,'o')
#plt.show()

#------------------------------------------------------------------
# Implemented as a one-time function

# Initialization
theta0 = np.random.rand()
theta1 = np.random.rand()

# Prediction function
def f(x):
    return theta0 + theta1*x

# Objective function
def E(x,y):
    return 0.5 * np.sum((y-f(x))*2)

# Standardization
mu =train_x.mean()
sigma = train_y.std()

def standardize(x):
    return (x-mu)/sigma

train_z = standardize(train_x)

# Draw the graph
#plt.plot(train_z,train_y,'o')
#plt.show()
#------------------------------------------------------------------
# Learning rate
ETA = 1e-3
# The difference of the error
diff = 1
# Number of updates
count = 0
# Repetitive learning
error = E(train_z,train_y)
while diff > 1e-2:
    # The update result is saved to a temporary variable
    tmp0 = theta0 - ETA * np.sum((f(train_z)-train_y))
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    # Update parameter
    theta0 = tmp0
    theta1 = tmp1
    # Calculate the error from last time
    current_error = E(train_z,train_y)
    diff = error - current_error
    error = current_error
    # Output log
    count += 1
    log = 'It is {}: theta0 = {:.3f},theta1 = {:.3f},difference = {:.4f}'
    print(log.format(count,theta0,theta1,diff))
# Draw the graph
x = np.linspace(-3,3,100)
plt.plot(train_z,train_y,'o')
plt.plot(x,f(x))
plt.show()
