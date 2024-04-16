import numpy as np

from global_name_space import ARGS

def generate_mesh(x_min=0, x_max=5, y_min=0, y_max=5, n_points=100):
    x = np.linspace(x_min,x_max,n_points)
    y = np.linspace(y_min,y_max,n_points)
    return np.meshgrid(x, y)

def load_data():
    X, Y = generate_mesh()
    Z = ARGS.compute_loss(X, Y)
    return X, Y, Z