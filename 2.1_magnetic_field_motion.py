import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants 
from matplotlib.animation import FuncAnimation

def mangetic_dipole_field(m_dipole, r_dipole, x, y, z):
 
    x0 = r_dipole[0]
    y0 = r_dipole[1]
    z0 = r_dipole[2]

    r = np.sqrt((x-x0) ** 2 + (y-y0) ** 2 + (z-z0) ** 2) 
    r = np.where(r == 0, np.nan, r)

    const = constants.mu_0 / (4 * constants.pi * r ** 3)

    m_dot_r = ((m_dipole[0] * (x-x0) + m_dipole[1] * (y-y0) + m_dipole[2] * (z-z0))) / r

    u = const * (3 * m_dot_r / r * (x-x0) - m_dipole[0]) 
    v = const * (3 * m_dot_r / r * (y-y0) - m_dipole[1]) 
    w = const * (3 * m_dot_r / r * (z-z0) - m_dipole[2]) 

    return u, v, w

def vector_add_magnets(r_dipole, m_matrix, x, y, z):
    u = 0
    v = 0
    w = 0
    for i in range(0 , m_matrix.shape[0]):
        ui, vi, wi = mangetic_dipole_field(m_matrix[i, :], r_dipole[i], x, y, z)
        u += ui
        v += vi
        w += wi
    return u, v, w

def solve_dot_state(initial_state, q, m, r_vector, m_matrix):
    dot_state_vector = np.empty(6)

    x = initial_state[0]
    y = initial_state[1]
    z = initial_state[2]

    v_x = initial_state[3]
    v_y = initial_state[4]
    v_z = initial_state[5]

    B_x, B_y, B_z = vector_add_magnets(r_vector, m_matrix, x, y, z)


    a_x = q / m * (v_y * B_z - v_z * B_y)
    a_y = q / m * (v_z * B_x - v_x * B_z)
    a_z = q / m * (v_x * B_y - v_y * B_x)

    dot_state_vector[0] = v_x
    dot_state_vector[1] = v_y
    dot_state_vector[2] = v_z

    dot_state_vector[3] = a_x
    dot_state_vector[4] = a_y
    dot_state_vector[5] = a_z

    return dot_state_vector

def next_state_RK4(initial_state, q, m, r_vector, m_matrix, h=0.01):
    x = initial_state[0]
    y = initial_state[1]
    z = initial_state[2]

    k1 = np.array(solve_dot_state(initial_state, q, m, r_vector, m_matrix))
    k2 = np.array(solve_dot_state(initial_state + 0.5 * h * k1, q, m, r_vector, m_matrix))
    k3 = np.array(solve_dot_state(initial_state + 0.5 * h * k2, q, m, r_vector, m_matrix))
    k4 = np.array(solve_dot_state(initial_state + h * k3, q, m, r_vector, m_matrix))
    new_state = np.array(initial_state + h/6 * (k1 + 2*k2 + 2*k3 + k4))
    return new_state

def moving_charge(r_initial, q_dipole, m_dipole, r_vector, m_matrix, v_vector, h=0.01, N=10000):
    states = np.empty((6, int(N)))
    states[:, 0] = np.array([r_initial[0], r_initial[1], r_initial[2], v_vector[0], v_vector[1], v_vector[2]])
    for i in range(1, N):
        states[:, i] = next_state_RK4(states[:, i-1], q_dipole, m_dipole, r_vector, m_matrix, h)

    return states


#chatgpt generated the cases cuz i cba to do that by hand 


r = np.array([
    [ 1.2, -2.5, -8.0],
    [-3.1,  1.8, -6.2],
    [ 0.4,  3.6, -4.5],
    [ 2.8, -0.9, -2.8],
    [-1.5, -3.7, -1.2],
    [ 3.4,  2.2,  0.3],

    [-4.0,  0.3,  1.8],
    [ 1.7, -1.8,  3.3],
    [ 4.3,  1.1,  4.9],
    [-0.8,  2.9,  6.2],
    [ 2.5, -3.3,  7.5],
    [-3.6,  1.4,  9.0]
])

m_matrix = np.array([
 [ 1.7, -3.2,  0.8],
 [-2.4,  1.1,  4.6],
 [ 0.5,  2.8, -1.9],
 [ 3.6, -0.7, -2.3],
 [-1.2, -4.1,  2.5],
 [ 2.9,  3.3,  1.4],
 [-3.8,  0.6, -1.1],
 [ 1.3, -2.7,  3.9],
 [ 4.2,  1.8, -0.4],
 [-0.9,  2.2,  4.7],
 [ 2.1, -3.6, -2.8],
 [-4.4,  1.5,  0.9]
])

m_matrix = m_matrix * 1e4
#done with the chatgpt stuff


xyz_max = 2
x0min = np.min(r[:, 0])
x0max = np.max(r[:, 0])
y0min = np.min(r[:, 1])
y0max = np.max(r[:, 1])
z0min = np.min(r[:, 2])
z0max = np.max(r[:, 2])


x, y, z = np.meshgrid(np.arange(-xyz_max + x0min, xyz_max + x0max, 1), np.arange(-xyz_max + y0min, xyz_max + y0max, 1), np.arange(-xyz_max + z0min, xyz_max + z0max, 1))


u2, v2, w2 = vector_add_magnets(r, m_matrix, x, y, z)


r_dip = np.array([0,0,0])
q_dip = 1e-7
M_dip = 1e-11
v_vector = [4,2,4]

states = moving_charge(r_dip, q_dip, M_dip, r, m_matrix, v_vector, 0.01, 1000)

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
particle = pv.PolyData(np.array([[r_dip[0], r_dip[1], r_dip[2]]]))

plotter.add_mesh(
    particle,
    color="#21FF76",
    point_size=20,
    render_points_as_spheres=True
)

# --------------------------------------------------
# Fixed black points
# --------------------------------------------------
fixed_points = pv.PolyData(r[:, :3])

plotter.add_mesh(
    fixed_points,
    color="black",
    point_size=10,
    render_points_as_spheres=True
)

# --------------------------------------------------
# Vector field
# --------------------------------------------------
grid = pv.StructuredGrid(x, y, z)

vectors = np.column_stack((
    u2.ravel(order="F"),
    v2.ravel(order="F"),
    w2.ravel(order="F")
))

# normalize=True equivalent
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
norms[norms == 0] = 1
vectors_normalized = vectors / norms

grid["vectors"] = vectors_normalized

arrows = grid.glyph(
    orient="vectors",
    scale=False,
    factor=0.4
)

plotter.add_mesh(
    arrows,
    color="blue",
    opacity=0.2
)

# --------------------------------------------------
# Trail
# --------------------------------------------------
trail = pv.PolyData(np.array([[states[0, 0], states[1, 0], states[2, 0]]]))

plotter.add_mesh(
    trail,
    color="orange",
    line_width=3
)

# --------------------------------------------------
# Bounds / axes
# --------------------------------------------------
bounds = (
    -xyz_max + x0min - 5, xyz_max + x0max + 5,
    -xyz_max + y0min - 5, xyz_max + y0max + 5,
    -xyz_max + z0min - 5, xyz_max + z0max + 5
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
# Show window
# --------------------------------------------------
plotter.show(interactive_update=True, auto_close=False)

# --------------------------------------------------
# Animation loop
# --------------------------------------------------
n_frames = min(1000, states.shape[1])

for frame in range(n_frames):
    # update moving particle
    pos = np.array([[states[0, frame], states[1, frame], states[2, frame]]])
    particle.points = pos

    # update trail
    path = states[:3, :frame + 1].T
    trail.points = path

    if len(path) >= 2:
        trail.lines = np.concatenate((
            [len(path)],
            np.arange(len(path))
        ))

    plotter.update()
    time.sleep(0.033)   # about 33 ms per frame

plotter.close()
