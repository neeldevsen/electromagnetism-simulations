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


def solve_dot_state(initial_state, q, m, r1_vector, r2_vector, q_vector, m_matrix):
    dot_state_vector = np.empty(6)

    x = initial_state[0]
    y = initial_state[1]
    z = initial_state[2]

    v_x = initial_state[3]
    v_y = initial_state[4]
    v_z = initial_state[5]

    B_x, B_y, B_z = vector_add_magnets(r2_vector, m_matrix, x, y, z)
    E_x, E_y, E_z = vector_add_point_fields(r1_vector, q_vector, x, y, z)

    a_x = q / m * (E_x + v_y * B_z - v_z * B_y)
    a_y = q / m * (E_y + v_z * B_x - v_x * B_z)
    a_z = q / m * (E_z + v_x * B_y - v_y * B_x)

    dot_state_vector[0] = v_x
    dot_state_vector[1] = v_y
    dot_state_vector[2] = v_z

    dot_state_vector[3] = a_x
    dot_state_vector[4] = a_y
    dot_state_vector[5] = a_z

    return dot_state_vector

def next_state_RK4(initial_state, q, m, r1_vector, r2_vector, q_vector, m_matrix, h=0.01):
    x = initial_state[0]
    y = initial_state[1]
    z = initial_state[2]

    k1 = np.array(solve_dot_state(initial_state, q, m, r1_vector, r2_vector, q_vector, m_matrix))
    k2 = np.array(solve_dot_state(initial_state + 0.5 * h * k1, q, m, r1_vector, r2_vector, q_vector, m_matrix))
    k3 = np.array(solve_dot_state(initial_state + 0.5 * h * k2, q, m, r1_vector, r2_vector, q_vector, m_matrix))
    k4 = np.array(solve_dot_state(initial_state + h * k3, q, m, r1_vector, r2_vector, q_vector, m_matrix))
    new_state = np.array(initial_state + h/6 * (k1 + 2*k2 + 2*k3 + k4))
    return new_state

def moving_charge(r_initial, q_dipole, m_dipole, r1_vector, r2_vector, q_vector, m_matrix, v_vector, h=0.01, N=10000):
    states = np.empty((6, int(N)))
    states[:, 0] = np.array([r_initial[0], r_initial[1], r_initial[2], v_vector[0], v_vector[1], v_vector[2]])
    for i in range(1, N):
        states[:, i] = next_state_RK4(states[:, i-1], q_dipole, m_dipole, r1_vector, r2_vector, q_vector, m_matrix, h)

    return states


#chatgpt generated the cases cuz i cba to do that by hand 

q = np.array([
     1, -1,  1, -1,
    -1,  1, -1,  1,
     2, -2,  2, -2
])

q = q * 1e-11

r1 = np.array([
    [-3, -3, -9],
    [ 3, -3, -9],
    [-3,  3, -9],
    [ 3,  3, -9],

    [-3, -3,  9],
    [ 3, -3,  9],
    [-3,  3,  9],
    [ 3,  3,  9],

    [-1,  0, -3],
    [ 1,  0,  3],
    [ 0, -1, -6],
    [ 0,  1,  6]
])

r2 = np.array([
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
x0min = np.min(r1[:, 0])
x0max = np.max(r1[:, 0])
y0min = np.min(r1[:, 1])
y0max = np.max(r1[:, 1])
z0min = np.min(r1[:, 2])
z0max = np.max(r1[:, 2])


x, y, z = np.meshgrid(np.arange(-xyz_max + x0min, xyz_max + x0max, 1), np.arange(-xyz_max + y0min, xyz_max + y0max, 1), np.arange(-xyz_max + z0min, xyz_max + z0max, 1))


u1, v1, w1 = vector_add_point_fields(r1, q, x, y, z)
u2, v2, w2 = vector_add_magnets(r2, m_matrix, x, y, z)


r_dip = np.array([0,0,0])
q_dip = 1e-7
M_dip = 1e-11
v_vector = [1.2,1.6,1.8]

states = moving_charge(r_dip, q_dip, M_dip, r1, r2, q, m_matrix, v_vector, 0.01, 1000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pt = ax.scatter(r_dip[0], r_dip[1], r_dip[2], color="#21FF76", s = 200)


for i in range(0, len(q)):
    if q[i] >= 0:
        color = "red"
    else:
        color = "blue"
    ax.scatter(r1[i,0], r1[i,1], r1[i,2], color=color, s=50)

for i in range(0, 12):
    ax.scatter(r2[i,0], r2[i,1], r2[i,2], color="black", s=50)

ax.set_xlim(left=-xyz_max + x0min -5, right=xyz_max + x0max + 5)
ax.set_ylim(bottom=-xyz_max + y0min -5, top=xyz_max + y0max + 5)
ax.set_zlim(bottom=-xyz_max + z0min -5, top=xyz_max + z0max +5)

    # Plot the vector field
ax.quiver(x, y, z, u1, v1, w1, normalize=True, length=0.4, color="red", alpha = 0.1)
ax.quiver(x, y, z, u2, v2, w2, normalize=True, length=0.4, color="blue", alpha = 0.1)

    # Set labels
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')


trail, = ax.plot([], [], [], linewidth=2)


def update(frame):
    x = states[0, frame]
    y = states[1, frame]
    z = states[2, frame]

    trail.set_data(states[0, :frame+1], states[1, :frame+1])
    trail.set_3d_properties(states[2, :frame+1])

    pt._offsets3d = ([x], [y], [z])

    return pt,

ani = FuncAnimation(fig=fig, func=update,frames=1000, interval=33, blit=True)
plt.legend(loc="upper right", fontsize=14)
plt.show()
    

