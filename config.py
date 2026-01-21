import jax.numpy as jnp
from random import randint
import jax
from helper_functions import generate_positions

TIME_ITER = 60              # Robot simulation iterations
SVGD_ITER = 30               # SVGD iterations
NUM_PARTICLE = 100           # number of samples
T = 30                       # horizon steps (time step=0.05)
DIM_U = 2                    # dimension of control input u
DT = 0.05                    # dt (s)
R_col = 0.3                  # collision avoidance
N = 10                       # Number of robots
     

key = jax.random.PRNGKey(randint(0,10000)) 
init_pos = generate_positions(N, R_col, minval=-2.0, maxval=2.0, key = key)
init_vel = jnp.ones((N, 2))  # initial velocity = [1,1] for all
state = jnp.hstack([init_pos, init_vel])
                   
# Goal states: spread around a region far from initial states
goal_pos = generate_positions(N, R_col, minval=9.0, maxval=12.0, key = key)
x_goal = goal_pos


# 7 obstacles with varying sizes spread roughly along the path
obstacles = jnp.array([
    [2.0, 2.0],
    [5.0, 4.0],
    [6.0, 6.3],
    [9.0, 5.0],
    [8.3, 8.5],
    [4.5, 2.5],
    [3.5, 8.0],
])

# Radii for each obstacle
radii = jnp.array([0.5, 0.7, 0.85, 1.5, 0.9, 0.4, 1.4])

# # 10 obstacles with varying sizes spread roughly along the path
# obstacles = jnp.array([
#     [2.0, 2.0],
#     [5.0, 8.0],
#     [6.0, 6.3],
#     [3.0, 7.0],
#     [5.3, 4.5],
#     [6.5, 2.5],
#     [3.2, 1.5],
#     [8.3, 7.5],
#     [6.9, 4.5],
#     [3.2, 9.0],
# ])
# radii = jnp.array([0.5, 0.7, 0.85, 1.5, 0.9, 0.4, 1.4, 1.5, 1.0, 0.8])

padding = 0.3

Q, R, S = 50.0, 10000.0, 1500.0*jnp.ones(2)    # Q: velocity, R: collision, S: Goal
