import numpy as np
import matplotlib.pyplot as plt

# learn more data
for d in range(1000):
    # generate np.array([0, 0, ... 0, 0]), d
    pass

D = np.array([
    [np.array([0, 0, 0, 0]), 0],
    [np.array([0, 0, 0, 1]), 1],
    [np.array([0, 0, 1, 0]), 2],
    [np.array([1, 0, 1, 0]), 10],
    [np.array([1, 0, 1, 1]), 5],
    [np.array([0, 1, 1, 1]), 7],
])


D1 = np.array([
    [np.array([0, 1, 0, 0])],
    [np.array([0, 1, 1, 0])],
    [np.array([1, 1, 1, 0])],
])

w = np.zeros((D[0][0].shape[0],1))

β = -0.4
α = 0.1
σ = lambda x: x #(x > 1).astype(int)
 
def f(x):
    s = β + x @ w
    return σ(s)
 
def train():
    global w
    _w = w.copy()
    for x, y in D:
        i = np.where(x>0)
        w[i] += α * (y - f(x))
    return (w != _w).any()
            
while train():
    print(w)

for d in D1:
    print(np.round(f(d[0])))





