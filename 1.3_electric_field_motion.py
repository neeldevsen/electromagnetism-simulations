import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants 
from matplotlib.animation import FuncAnimation

def point_charge_field(q_dipole, r_dipole, x, y, z):
    const = q_dipole / (4 * constants.pi * constants.epsilon_0)

    x0 = r_dipole[0]
    y0 = r_dipole[1]
    z0 = r_dipole[2]

    r3 = np.sqrt((x-x0) ** 2 + (y-y0) ** 2 + (z-z0) ** 2) ** 3
    r3 = np.where(r3 == 0, np.nan, r3)
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





def solve_dot_state(initial_state, q, m, r_vector, q_vector, u, v, w):
    dot_state_vector = np.empty(6)

    x = initial_state[0]
    y = initial_state[1]
    z = initial_state[2]

    v_x = initial_state[3]
    v_y = initial_state[4]
    v_z = initial_state[5]


    a_x = q * vector_add_point_fields(r_vector, q_vector, x, y, z)[0] / m
    a_y = q * vector_add_point_fields(r_vector, q_vector, x, y, z)[1] / m
    a_z = q * vector_add_point_fields(r_vector, q_vector, x, y, z)[2] / m

    dot_state_vector[0] = v_x
    dot_state_vector[1] = v_y
    dot_state_vector[2] = v_z

    dot_state_vector[3] = a_x
    dot_state_vector[4] = a_y
    dot_state_vector[5] = a_z

    return dot_state_vector

def next_state_RK4(initial_state, q, m, r_vector, q_vector, h=0.01):
    x = initial_state[0]
    y = initial_state[1]
    z = initial_state[2]

    u, v, w = vector_add_point_fields(r_vector, q_vector, x, y, z)
    k1 = np.array(solve_dot_state(initial_state, q, m, r_vector, q_vector, u, v, w))
    k2 = np.array(solve_dot_state(initial_state + 0.5 * h * k1, q, m, r_vector, q_vector, u, v, w))
    k3 = np.array(solve_dot_state(initial_state + 0.5 * h * k2, q, m, r_vector, q_vector, u, v, w))
    k4 = np.array(solve_dot_state(initial_state + h * k3, q, m, r_vector, q_vector, u, v, w))
    new_state = np.array(initial_state + h/6 * (k1 + 2*k2 + 2*k3 + k4))
    return new_state

def moving_charge(r_initial, q_dipole, m_dipole, r_vector, q_vector, h=0.01, N=10000):
    states = np.empty((6, int(N)))
    states[:, 0] = np.array([r_initial[0], r_initial[1], r_initial[2], 0, 0, 0])
    for i in range(1, N):
        states[:, i] = next_state_RK4(states[:, i-1], q_dipole, m_dipole, r_vector, q_vector, h)

    return states


#chatgpt generated the cases cuz i cba to do that by hand 

q = np.array([
     1, -1,  1, -1,
    -1,  1, -1,  1,
     2, -2,  2, -2
])

q = q * 1e-7

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


xyz_max = 2
x0min = np.min(r[:, 0])
x0max = np.max(r[:, 0])
y0min = np.min(r[:, 1])
y0max = np.max(r[:, 1])
z0min = np.min(r[:, 2])
z0max = np.max(r[:, 2])


x, y, z = np.meshgrid(np.arange(-xyz_max + x0min, xyz_max + x0max, 1), np.arange(-xyz_max + y0min, xyz_max + y0max, 1), np.arange(-xyz_max + z0min, xyz_max + z0max, 1))


u, v, w = vector_add_point_fields(r, q, x, y, z)

r_dip = np.array([0,0,0])
q_dip = 1e-7
m_dip = 1e-6

states = moving_charge(r_dip, q_dip, m_dip, r, q, 0.01, 1000)

import numpy as np
import pyvista as pv
import time


# --------------------------------------------------
# Plotter
# --------------------------------------------------
plotter = pv.Plotter()
plotter.set_background("black")


# --------------------------------------------------
# Moving particle
# --------------------------------------------------
particle = pv.PolyData(
    np.array([[r_dip[0], r_dip[1], r_dip[2]]])
)

plotter.add_mesh(
    particle,
    color="#21FF76",
    point_size=20,
    render_points_as_spheres=True
)


# --------------------------------------------------
# Charges
# --------------------------------------------------
q = np.asarray(q)
r = np.asarray(r)

positive = q >= 0
negative = q < 0

if np.any(positive):
    positive_points = pv.PolyData(r[positive])

    plotter.add_mesh(
        positive_points,
        color="red",
        point_size=12,
        render_points_as_spheres=True
    )

if np.any(negative):
    negative_points = pv.PolyData(r[negative])

    plotter.add_mesh(
        negative_points,
        color="blue",
        point_size=12,
        render_points_as_spheres=True
    )


# --------------------------------------------------
# Vector field
# --------------------------------------------------
grid = pv.StructuredGrid(x, y, z)

vectors = np.column_stack([
    u.ravel(order="F"),
    v.ravel(order="F"),
    w.ravel(order="F")
])

# Normalize field vectors
magnitude = np.linalg.norm(vectors, axis=1)

valid = magnitude > 0
vectors[valid] /= magnitude[valid, None]

grid["field"] = vectors

arrows = grid.glyph(
    orient="field",
    scale=False,
    factor=0.4
)

plotter.add_mesh(
    arrows,
    color="white",
    opacity=0.10
)


# --------------------------------------------------
# Particle trail
# --------------------------------------------------
trail = pv.PolyData()

# Initialise with first position
trail.points = np.array([
    [states[0, 0], states[1, 0], states[2, 0]]
])

plotter.add_mesh(
    trail,
    color="#21FF76",
    line_width=3
)


# --------------------------------------------------
# Bounds / axes
# --------------------------------------------------
bounds = (
    -xyz_max + x0min - 5,
     xyz_max + x0max + 5,

    -xyz_max + y0min - 5,
     xyz_max + y0max + 5,

    -xyz_max + z0min - 5,
     xyz_max + z0max + 5
)

plotter.show_bounds(
    bounds=bounds,
    xtitle="x axis",
    ytitle="y axis",
    ztitle="z axis",
    color="white"
)

plotter.add_axes(color="white")


# --------------------------------------------------
# Start interactive window
# --------------------------------------------------
plotter.show(
    interactive_update=True,
    auto_close=False
)


# --------------------------------------------------
# Animation
# --------------------------------------------------
n_frames = min(6000, states.shape[1])

for frame in range(n_frames):

    # current particle position
    position = np.array([[
        states[0, frame],
        states[1, frame],
        states[2, frame]
    ]])

    particle.points = position

    # trajectory up to current frame
    path = states[:3, :frame + 1].T

    if len(path) >= 2:
        trail.points = path

        # one continuous polyline through all trail points
        trail.lines = np.concatenate([
            [len(path)],
            np.arange(len(path))
        ])

    plotter.update()

    # approximately equivalent to interval=33 ms
    time.sleep(0.033)

