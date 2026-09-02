import matplotlib.pyplot as plt
import numpy as np
from scipy import constants 

def point_charge_field(q_dipole, r_dipole, x, y):
    const = q_dipole / (4 * constants.pi * constants.epsilon_0)

    x0 = r_dipole[0]
    y0 = r_dipole[1]

    r = np.sqrt((x-x0) ** 2 + (y-y0) ** 2) 
    r[r <= 1] = np.nan
    u = const * (x - x0) / r
    v = const * (y - y0) / r

    return u, v


def vector_add_point_fields(r_dipole, q_vector, x, y):
    u = 0
    v = 0
    w = 0
    for i in range(0 , len(q_vector)):
        ui, vi = point_charge_field(q_vector[i], r_dipole[i], x, y)
        u += ui
        v += vi
    return u, v

def potential_charge_field(q_dipole, r_dipole, x, y):
    const = q_dipole / (4 * constants.pi * constants.epsilon_0)

    x0 = r_dipole[0]
    y0 = r_dipole[1]

    r = np.sqrt((x-x0) ** 2 + (y-y0) ** 2 )
    r[r <= 1] = np.nan
    V = const / r

    return V

def scalar_add_potential(r_dipole, q_vector, x, y):
    V = np.zeros_like(x, dtype=float)
    for i in range(0 , len(q_vector)):
        Vi = point_charge_field(q_vector[i], r_dipole[i], x, y)[0]
        V += Vi

    return V


#chatgpt generated the cases cuz i cba to do that by hand 

q = np.array([
     1, -2,  3, -1,
     2, -3,  1, -2,
     3, -1,  2, -3
])

r = np.array([
    [-4.0, -3.0],
    [-1.5, -4.0],
    [ 2.5, -3.5],
    [ 4.0, -1.0],

    [-3.5,  0.5],
    [-0.8, -0.5],
    [ 1.2,  1.0],
    [ 3.8,  2.0],

    [-3.0,  3.5],
    [-0.5,  4.0],
    [ 2.0,  3.2],
    [ 4.5,  4.0]
])

#done with the chatgpt stuff


xyz_max = 6
x0min = np.min(r[:, 0])
x0max = np.max(r[:, 0])
y0min = np.min(r[:, 1])
y0max = np.max(r[:, 1])


x, y = np.meshgrid(np.arange(-xyz_max + x0min, xyz_max + x0max, 1), np.arange(-xyz_max + y0min, xyz_max + y0max, 1))

u, v = vector_add_point_fields(r, q, x, y)
V = scalar_add_potential(r,q,x,y)
fig = plt.figure()

contour = plt.contour(x, y, V, colors="gray")



for i in range(0, len(q)):
    if q[i] >= 0:
        color = "red"
    else:
        color = "blue"
    plt.scatter(r[i,0], r[i,1], color=color, s=100)


plt.xlim(left=-xyz_max + x0min, right=xyz_max + x0max)
plt.ylim(bottom=-xyz_max + y0min, top=xyz_max + y0max)


    # Plot the vector field
plt.quiver(x, y, u, v,  color="black")

    # Set labels
plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.show()