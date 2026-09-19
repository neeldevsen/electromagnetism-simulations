import matplotlib.pyplot as plt
import numpy as np
from scipy import constants 

def point_charge_field(q_dipole, r_dipole, x, y):
    const = q_dipole / (4 * constants.pi * constants.epsilon_0)

    x0 = r_dipole[0]
    y0 = r_dipole[1]

    r = np.sqrt((x-x0) ** 2 + (y-y0) ** 2) 
    r[r <= 1] = np.nan
    u = const * (x - x0) / (r**3)
    v = const * (y - y0) / (r**3)

    return u, v


def vector_add_point_fields(r_dipole, q_vector, x, y):
    u = 0
    v = 0
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
        Vi = potential_charge_field(q_vector[i], r_dipole[i], x, y)
        V += Vi

    return V


#chatgpt generated the cases cuz i cba to do that by hand 

q = np.array([
     1, -1,  1, -1,
    -1,  1, -1,  1,
     2, -2,  2, -2
])

r = np.array([
    [-3, -3],
    [ 3, -3],
    [-3,  3],
    [ 3,  3],

    [-3, -3],
    [ 3, -3],
    [-3,  3],
    [ 3,  3],

    [-1,  0],
    [ 1,  0],
    [ 0, -1],
    [ 0,  1]
])


#done with the chatgpt stuff


xyz_max = 6
x0min = np.min(r[:, 0])
x0max = np.max(r[:, 0])
y0min = np.min(r[:, 1])
y0max = np.max(r[:, 1])


import numpy as np
import matplotlib.pyplot as plt

# grid
x, y = np.meshgrid(
    np.arange(-xyz_max + x0min, xyz_max + x0max, 1),
    np.arange(-xyz_max + y0min, xyz_max + y0max, 1)
)

# field + potential
u, v = vector_add_point_fields(r, q, x, y)
V = scalar_add_potential(r, q, x, y)

# figure
fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
ax.set_facecolor("black")

# equipotential lines (yellow)
ax.contour(
    x, y, V,
    levels=20,
    colors="cyan",
    linewidths=0.8,
    alpha=0.9
)

# electric field lines (white)
ax.streamplot(
    x, y, u, v,
    color="white",
    density=1.2,
    linewidth=0.7,
    arrowsize=0.8
)

# charges
for i in range(len(q)):
    color = "red" if q[i] >= 0 else "blue"
    ax.scatter(
        r[i, 0], r[i, 1],
        color=color,
        s=100,
        edgecolors="white",
        linewidths=0.8,
        zorder=3
    )

# limits
ax.set_xlim(-xyz_max + x0min, xyz_max + x0max)
ax.set_ylim(-xyz_max + y0min, xyz_max + y0max)

# labels
ax.set_xlabel("X axis", color="white")
ax.set_ylabel("Y axis", color="white")

# ticks + spines
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

plt.show()