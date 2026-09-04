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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pt = ax.scatter(r_dip[0], r_dip[1], r_dip[2], color="#21FF76", s = 200)


for i in range(0, len(q)):
    if q[i] >= 0:
        color = "red"
    else:
        color = "blue"
    ax.scatter(r[i,0], r[i,1], r[i,2], color=color, s=50)



ax.set_xlim(left=-xyz_max + x0min -5, right=xyz_max + x0max + 5)
ax.set_ylim(bottom=-xyz_max + y0min -5, top=xyz_max + y0max + 5)
ax.set_zlim(bottom=-xyz_max + z0min -5, top=xyz_max + z0max +5)

    # Plot the vector field
ax.quiver(x, y, z, u, v, w, normalize=True, length=0.4, color="black", alpha = 0.1)

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

ani = FuncAnimation(fig=fig, func=update,frames=6000, interval=33, blit=True)
plt.legend(loc="upper right", fontsize=14)
plt.show()
    
