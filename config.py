import jax.numpy as jnp
from random import randint
import jax

### Generate random positions between minval and maxval that are min_dist away from each other.
def generate_positions(N, min_dist, minval, maxval, key):
    positions = []
    while len(positions) < N:
        key, subkey = jax.random.split(key)
        candidate = jax.random.uniform(subkey, (1, 2), minval=minval, maxval=maxval)
        if positions:
            # compute distance to all existing positions
            dists = jnp.linalg.norm(jnp.vstack(positions) - candidate, axis=1)
            if jnp.all(dists >= min_dist):
                positions.append(candidate[0])
        else:
            positions.append(candidate[0])
    return jnp.array(positions)



TIME_ITER = 100              # Robot simulation iterations
SVGD_ITER = 1000              # SVGD iterations
NUM_PARTICLE = 100           # number of samples
T = 30                       # horizon steps (time step=0.05)
DIM_U = 2                    # dimension of control input u
DT = 0.05                    # dt (s)
R_com = 3.0                  # communication radius (m)
R_col = 0.3                  # collision avoidance
desired_r = 3                # desired robustness
LEADER_NUM = 4               # number of leaders
N = 10                        # Number of robots
     

key = jax.random.PRNGKey(randint(0,10000)) 
init_pos = generate_positions(N, R_col, minval=-2.0, maxval=2.0, key = key)
init_vel = jnp.ones((N, 2))  # initial velocity = [1,1] for all
state = jnp.hstack([init_pos, init_vel])
                   
# Goal states: spread around a region far from initial states
goal_pos = generate_positions(N, R_col, minval=9.0, maxval=12.0, key = key)
x_goal = goal_pos


# 8 obstacles with varying sizes spread roughly along the path
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
padding = 0.3
radii = jnp.array([0.5, 0.7, 0.85, 1.5, 0.9, 0.4, 1.4])

Q, R, S, P = 50.0, 10000.0, 1500.0*jnp.ones(2), 4*1e3     # Q: velocity, R: collision, S: Goal, P: robustness


# import resilient_motion_discrete