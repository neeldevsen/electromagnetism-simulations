import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse

def Jz(x, y, x0, y0, J0, sigma):
    return J0 * np.exp(-((x-x0) ** 2 + (y-y0) ** 2) / (2 * sigma ** 2))

def generate_b(L, N, x0, y0, J0, sigma):
    h = L / (N-1)
    b = np.fromfunction(lambda i,j: Jz(-0.5 * L + h * i, -0.5 * L + h * j, x0, y0, J0, sigma), shape=(N,N))
    b[0,:] = 0
    b[:,0] = 0
    b[N-1,:] = 0
    b[:,N-1] = 0

    return b

def RHS(L, N, x0, y0, J0, sigma):
    b = generate_b(L, N, x0, y0, J0, sigma)
    b_vec = np.empty(N ** 2)
    for i in range(0, N):
        for j in range(0, N):
            b_vec[i*N + j] = constants.mu_0 * b[i,j] 

    return b_vec

def sparse_matrix(N, L):
    h = L / (N-1)
    rows = []
    cols = []
    values = []
    for i in range(0, N):
        for j in range(0, N):
            k = i*N + j

            if i == 0 or i == N - 1 or j == 0 or j == N -1:
                rows.append(k)
                cols.append(k)
                values.append(1)
            else:
                rows.append(k)
                cols.append(k)
                values.append(4/(h**2))

                rows.append(k)
                cols.append(k-1)
                values.append(-1/(h**2))

                rows.append(k)
                cols.append(k+1)
                values.append(-1/(h**2))

                rows.append(k)
                cols.append(k-N)
                values.append(-1/(h**2))

                rows.append(k)
                cols.append(k+N)
                values.append(-1/(h**2))

    A = sparse.coo_matrix((values, (rows, cols)), shape=(N ** 2, N ** 2))

    return A

def solve_system(A,b,N):
    a = sparse.linalg.spsolve(A,b)
    a_matrix = np.empty((N,N))
    for i in range(0, N):
        for j in range(0, N):
            k = N * i + j
            a_matrix[i, j] = a[k]
    return a_matrix

def generate_B(a, N, L):
    h = L / (N-1)
    B_x = np.zeros((N, N))
    B_y = np.zeros((N, N))
    for i in range(1, N-1):
        for j in range(1, N-1):
            B_x[i,j] = (a[i, j+1] - a[i, j-1]) / (2 * h)
            B_y[i,j] = -(a[i+1, j] - a[i-1, j]) / (2 * h)

    return B_x, B_y


N = 20
L = 10
x0 = 0
y0 = 0
J0 = 1
sigma  = 2

A = sparse_matrix(N, L)
b = RHS(L,N,x0,y0,J0,sigma)
b_matrix = generate_b(L,N,x0,y0,J0,sigma)
a = solve_system(A, b, N)

B_x, B_y = generate_B(a, N, L)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')


x , y = np.meshgrid(np.linspace(-L/2,L/2,N), np.linspace(-L/2,L/2,N), indexing="ij")

ax.quiver(x, y, np.zeros_like(x), B_x, B_y, np.zeros_like(B_x), normalize=True, length=0.4, color="black")
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')


ax.plot_surface(x,y,b_matrix, alpha=0.7, cmap="coolwarm")

plt.show()
                


                


    


    


