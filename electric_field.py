import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants 

def point_charge_field(q_dipole, r_dipole, x, y, z):
    const = q_dipole / (4 * constants.pi * constants.epsilon_0)

    x0 = r_dipole[0]
    y0 = r_dipole[1]
    z0 = r_dipole[2]

    r3 = np.sqrt((x-x0) ** 2 + (y-y0) ** 2 + (z-z0) ** 2) ** 3
    r3[r3 == 0] = np.nan
    u = const * (x - x0) / r3
    v = const * (y - y0) / r3
    w = const * (z - z0) / r3

    return u, v, w




def vector_add_point_fields(r_dipole, q_vector, x, y, z):
    u = 0
    v = 0
    w = 0
    for i in range(0 , len(q_vector)):
        ui, vi, wi = point_charge_field(q_vector[i], r_dipole[i], x, y, z)
        u += ui
        v += vi
        w += wi
    return u, v, w

#chatgpt generated the cases cuz i cba to do that by hand 

q = np.array([
     1, -1,  1, -1,
    -1,  1, -1,  1,
     2, -2,  2, -2
])

r = np.array([
    [-3, -3, -3],
    [ 3, -3, -3],
    [-3,  3, -3],
    [ 3,  3, -3],

    [-3, -3,  3],
    [ 3, -3,  3],
    [-3,  3,  3],
    [ 3,  3,  3],

    [-1,  0,  0],
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  1,  0]
])

#done with the chatgpt stuff


xyz_max = 6
x0min = np.min(r[:, 0])
x0max = np.max(r[:, 0])
y0min = np.min(r[:, 1])
y0max = np.max(r[:, 1])
z0min = np.min(r[:, 2])
z0max = np.max(r[:, 2])


x, y, z = np.meshgrid(np.arange(-xyz_max + x0min, xyz_max + x0max, 1), np.arange(-xyz_max + y0min, xyz_max + y0max, 1), np.arange(-xyz_max + z0min, xyz_max + z0max, 1))

u, v, w = vector_add_point_fields(r, q, x, y, z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(0, len(q)):
    if q[i] >= 0:
        color = "red"
    else:
        color = "blue"
    ax.scatter(r[i,0], r[i,1], r[i,2], color=color, s=100)


ax.set_xlim(left=-xyz_max + x0min, right=xyz_max + x0max)
ax.set_ylim(bottom=-xyz_max + y0min, top=xyz_max + y0max)
ax.set_zlim(bottom=-xyz_max + z0min, top=xyz_max + z0max)

    # Plot the vector field
ax.quiver(x, y, z, u, v, w, normalize=True, length=0.4, color="black")

    # Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()