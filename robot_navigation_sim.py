import jax.numpy as jnp
import jax
from svgd import *
from helper_functions import *
from config import *
from jax import jit
import time

# double integrator dynamics in 2D
@jax.jit
def double_integrator_rollout(x0,u, dt = DT):

    # jax.lax.scan: takes a function step_fn, initial point x0, and a sequence of u to loop over to 
    # construct a trajectory over a squence of u
    @jax.jit
    def rollout_single(x0_i, u_i):
        @jax.jit
        def step_fn(x, ui):
            px, py, vx, vy = x
            ax, ay = ui
            px_next = px + vx * dt + 1/2*ax*dt**2
            py_next = py + vy * dt + 1/2*ay*dt**2
            vx_next = vx + ax * dt
            vy_next = vy + ay * dt
            x_next = jnp.array([px_next, py_next, vx_next, vy_next])
            return x_next, x_next
        
        _, traj = jax.lax.scan(step_fn, x0_i, u_i)
        return traj

    return jax.vmap(rollout_single)(x0, u)

# Takes positions of all robots at time t, and comptues the collisions.
@jit
def inter_collision_penalty(pos_t):
    diff = pos_t[:, None, :] - pos_t[None, :, :]              # (N, N, 2)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-3)
    dist = dist + jnp.eye(N) * 1e6                     # mask self-distances
    penal = jnp.where(dist < 2 * R_col,
                    R * (2 * R_col - dist),
                    2 * R_col - dist)
    return penal

# Create a cost function with terminal cost.
# Q: (4,) weight vector for state stage cost
# R: (2*T,) weight vector for collision (both obstacle and inter-agent)
# S: (2,) weight vector for terminal state cost
# At the moment, I am only penalizing the norm of u and the final state's deviation to the goal


###### Need to make x0 such that it goes inside the function
def make_cost(Q,R,S, desired_terminal):


    def cost_fn(u, x0):

        # Min u norm
        controls = u.reshape(N, -1, 2)

        traj = double_integrator_rollout(x0, controls)    # shape (N,T,state_dim)

        vel_penal = Q*jnp.sum((traj[:, :, 2:])**2)
        
        reshaped_traj = traj[:, :, :2].reshape(-1,2)    # shape (T*N, state_dim)
        diffs = reshaped_traj[:, None,:] - obstacles[None,:, :] # shape (T*N, num_obstacles, state_dim)
        dists = jnp.linalg.norm(diffs, axis=-1)  # shape (T*N, num_obstacles)

        # compute penalties for obstacle collisions
        obs_col_penal = jnp.where(dists < radii+padding,R*(radii+padding - dists),radii+padding-dists)     # (T*N, num_obstacles)

        # compute penalities for inter-agent collisions:
        inter_col_penal = jnp.sum(jax.vmap(inter_collision_penalty, in_axes = 1)(traj[:, :, :2]))

        # sum over time and obstacles
        obstacle_penal = jnp.sum(obs_col_penal.flatten())

        # final state's deviation to the goal
        terminal_cost = jnp.sum(S*jnp.array(traj[:, -1, :2] - desired_terminal)**2)

        return -(vel_penal + terminal_cost + obstacle_penal + inter_col_penal)

    return cost_fn


if __name__ == "__main__" or __name__ == "resilient_motion_discrete":
    all_samples = []
    best_trajs = []
    time_history =[]

    key = jax.random.PRNGKey(0)
    log_prob = make_cost(Q, R, S, x_goal)
    cost_function = jax.jit(lambda u, v: log_prob(u, v))
    from svgd import run_svgd
    # with jax.log_compiles():
    for t in range(TIME_ITER):
        init_time = time.time()
        # Get a new key
        oldkey, key = jax.random.split(key)

        # Compute the SVGD steps
        theta_final = run_svgd(cost_function, state, key, NUM_PARTICLE, T, DIM_U, n_steps=SVGD_ITER, lr=0.05)

        # Roll out all sampled trajectories
        sample_trajs = jax.vmap(lambda u: double_integrator_rollout(state, u.reshape(N, T, DIM_U)))(theta_final)
        all_samples.append(sample_trajs)

        # Pick best trajectory
        costs = jax.vmap(lambda u: cost_function(u, state))(theta_final)
        best_idx = jnp.argmax(costs)
        best_traj = sample_trajs[best_idx]
        best_trajs.append(best_traj)

        # Update state to the next step
        state = best_traj[:,1]

        time_history.append(time.time()-init_time)
    animate_mpc(all_samples, best_trajs, x_goal)

    print("max comp time", max(time_history[1:]))
    print("average comp time", sum(time_history[1:])/(TIME_ITER-1))
    print("min comp time", min(time_history[1:]))
    plt.show()
