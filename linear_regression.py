import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:
    def __init__(self, X_train, y_train):
        self.X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1), dtype=X_train.dtype)])
        self.x_train = X_train
        self.y_train = y_train
        self.w = np.zeros(self.X_train.shape[1])
    
    # Sum of Square Error
    def loss(self):
        return float(np.sum((self.y_train - np.dot(self.X_train,self.w))**2))

    def fit(self):
        # Using Least Square Method by derivertive cost fuction(Sum of Square Error)
        self.w = inv(self.X_train.T@self.X_train) @ (self.X_train.T@self.y_train)
        print("loss={:.2f}".format(self.loss()))
        return self.w
        
    def predict(self, X_test):
        return self.w[0]*X_test + self.w[1]
    
    def plot(self):
        if self.x_train.shape[1] == 1:
            plot_2D(self.x_train, self.y_train, self.w)
        elif self.x_train.shape[1] == 2:
            plot_3D(self.x_train, self.y_train, self.w[0], self.w[1], self.w[2])
        else:
            print("Can be used only 1D or 2D data")

def plot_2D(X, Y, W):
    x = np.linspace(min(X), max(X), 10)
    plt.scatter(X, Y, alpha=0.7)
    plt.plot(x, W[0]*x + W[1], color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_3D(X,Y, slope_x1, slope_x2, intercept):
    X1, X2 = np.array_split(X, 2, axis=1)
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, Y, c='b', marker='o', label='Data Points')

    # Create a meshgrid for the plane
    x1_range = np.linspace(min(X1), max(X1), 100)
    x2_range = np.linspace(min(X2), max(X2), 100)
    x1_plane, x2_plane = np.meshgrid(x1_range, x2_range)
    y_plane = slope_x1 * x1_plane + slope_x2 * x2_plane + intercept

    # Plot the plane
    ax.plot_surface(x1_plane, x2_plane, y_plane, alpha=0.5, color='r', label='Linear Regression Plane')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')

    plt.title('3D Scatter Plot with Linear Regression Plane')
    plt.legend()
    plt.show()

def normalization(X):
    return (X-np.min(X))/(np.max(X)-np.min(X))