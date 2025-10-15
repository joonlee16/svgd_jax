import jax.numpy as jnp
import jax
from svgd import *
from helper_functions import *
from config import *
from evaluate_topology import *
from jax import lax, jit, jacfwd, jacrev, hessian
from functools import partial
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
            px_next = px + vx * dt
            py_next = py + vy * dt
            vx_next = vx + ax * dt
            vy_next = vy + ay * dt
            x_next = jnp.array([px_next, py_next, vx_next, vy_next])
            return x_next, x_next
        
        _, traj = jax.lax.scan(step_fn, x0_i, u_i)
        return traj

    return jax.vmap(rollout_single)(x0, u)


######################Computes the \bar {\pi}_{\mathcal F}######################
@jit
def barrier_func(x):
    leaders = 4
    q_A, s_A = 0.02, 1.0
    q, s = 0.02, 1.0

    sigmoid_A = lambda x: (1+q_A)/(1+(1/q_A)*jnp.exp(-s_A*x))-q_A
    sigmoid = lambda x: (1+q)/(1+(1/q)*jnp.exp(-s*x))-q

    # Vectorized AA
    diffs = x[:, None, :] - x[None, :, :]
    distsq = jnp.sum(diffs**2, axis=-1)
    A = jnp.where(R_com**2 - distsq >= 0, sigmoid_A((R_com**2 - distsq)**2), 0.0)
    A = A.at[jnp.diag_indices(N)].set(0.0)

    state_vector = jnp.zeros((N-leaders,1))

    def body(state, _):
        temp_x = A @ jnp.vstack([jnp.ones((leaders,1)), state])
        new_state = sigmoid(temp_x[leaders:] - desired_r)
        return new_state, new_state

    x_final, _ = lax.scan(body, state_vector, jnp.arange(2))
    return 1-jnp.sum(jnp.exp(-20*x_final[:,0]))


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
# R: (2*T,) weight vector for control cost
# S: (2,) weight vector for terminal state cost
# At the moment, I am only penalizing the norm of u and the final state's deviation to the goal


###### Need to make x0 such that it goes inside the function
def make_cost(Q,R,S, desired_terminal):


    def cost_fn(u, x0):

        # Takes position trajectories of all robots, and comptues the robustness discrete time HOCBF scores.
        @jit
        def compute_penalties(hs):
            ''' traj.shape = (T,N,dim_x)'''

            alpha1 = 1.5
            alpha2 = 1.5
            def delta_phi_i(t):
                delta_phi_t = hs[t+1] - hs[t] + alpha1*(hs[t])
                delta_phi_t_1 = hs[t+2] - hs[t+1] + alpha1*(hs[t+1])
                return delta_phi_t_1 - (1-alpha2)*delta_phi_t

            hocbf_scores = jax.vmap(delta_phi_i)(jnp.arange(0,T-2))

            penal = -jnp.where(hocbf_scores < 0.0,
                                    P * hocbf_scores,
                                    hocbf_scores)
            return penal

        # Min u norm
        controls = u.reshape(N, -1, 2)

        traj = double_integrator_rollout(x0, controls)    # shape (N,T,state_dim)

        vel_penal = Q*jnp.sum((traj[:, :, 2:])**2)
        
        reshaped_traj = traj[:, :, :2].reshape(-1,2)    # shape (T*N, state_dim)
        diffs = reshaped_traj[:, None,:] - obstacles[None,:, :] # shape (T*N, num_obstacles, state_dim)
        dists = jnp.linalg.norm(diffs, axis=-1)  # shape (T*N, num_obstacles)

        # compute penalties for obstacle collisions
        obs_col_penal = jnp.where(dists < radii+padding,R*(radii+padding - dists),radii+padding-dists)     # (T*N, num_obstacles)


        # strong r robustness maintenance
        # h.shape = (T, number of followers) = values of each follower for each time arr
        # der_ = (T, number of followers, number of all agents, dim_u)
        # double_der_ = (T, number of followers, number of all agents, dim_u, number of all agents, dim_u)

        # compute penalities for inter-agent collisions:
        inter_col_penal = jnp.sum(jax.vmap(inter_collision_penalty, in_axes = 1)(traj[:, :, :2]))

        # compute penalities for robustness CBF violations
        hs = jax.vmap(barrier_func, in_axes=1)(traj[:,:,:2])     # hs.shape = (T,)
        robustness_penal = jnp.sum(compute_penalties(hs))

        # sum over time and obstacles
        obstacle_penal = jnp.sum(obs_col_penal.flatten())

        # final state's deviation to the goal
        terminal_cost = jnp.sum(S*jnp.array(traj[:, -1, :2] - desired_terminal)**2)

        return -(vel_penal + terminal_cost + obstacle_penal + inter_col_penal + robustness_penal)
        # return -(vel_penal + terminal_cost + obstacle_penal + inter_col_penal)

    return cost_fn


if __name__ == "__main__" or __name__ == "resilient_motion_discrete":
    all_samples = []
    best_trajs = []
    time_history =[]
    robustness_computation_history = []

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

        robustness_computation_history.append(compute_strong_robust(state[:,:2]))
        time_history.append(time.time()-init_time)
    animate_mpc(all_samples, best_trajs, x_goal)

    print("max comp time", max(time_history[1:]))
    print("average comp time", sum(time_history[1:])/(TIME_ITER-1))
    print("min comp time", min(time_history[1:]))

    plt.plot(np.arange(TIME_ITER), robustness_computation_history)
    plt.show()
