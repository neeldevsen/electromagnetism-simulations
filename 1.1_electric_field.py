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
import numpy as np
import pyvista as pv

plotter = pv.Plotter()
plotter.set_background("black")

# -----------------------------
# Plot the charges
# -----------------------------
q = np.asarray(q)
r = np.asarray(r)

pos_mask = q >= 0
neg_mask = q < 0

if np.any(pos_mask):
    pos_points = pv.PolyData(r[pos_mask])
    plotter.add_mesh(
        pos_points,
        color="red",
        point_size=12,
        render_points_as_spheres=True
    )

if np.any(neg_mask):
    neg_points = pv.PolyData(r[neg_mask])
    plotter.add_mesh(
        neg_points,
        color="blue",
        point_size=12,
        render_points_as_spheres=True
    )

# -----------------------------
# Build the vector field grid
# -----------------------------
grid = pv.StructuredGrid(x, y, z)

vectors = np.column_stack((
    u.ravel(order="F"),
    v.ravel(order="F"),
    w.ravel(order="F")
))

# normalize vectors
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
norms[norms == 0] = 1
vectors_normalized = vectors / norms

grid["vectors"] = vectors_normalized

arrows = grid.glyph(
    orient="vectors",
    scale=False,
    factor=0.4
)

plotter.add_mesh(arrows, color="white")

# -----------------------------
# Bounds / axes
# -----------------------------
plotter.show_bounds(
    bounds=(
        -xyz_max + x0min, xyz_max + x0max,
        -xyz_max + y0min, xyz_max + y0max,
        -xyz_max + z0min, xyz_max + z0max
    ),
    xtitle="x axis",
    ytitle="y axis",
    ztitle="z axis",
    color="white"
)

plotter.add_axes(color="white")
plotter.show()