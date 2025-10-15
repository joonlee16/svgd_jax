import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import *
from matplotlib.patches import Circle


'''
Misc. helper functions, such as visualization and animations
'''

# -------------------------
# Animation Implementation
# -------------------------
def animate_svgd(history, log_prob):
    fig, ax = plt.subplots(figsize=(5, 5))
    scat = ax.scatter([], [], s=10)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("SVGD Particle Evolution")

    # Plot density contour
    xx, yy = np.meshgrid(np.linspace(-5,5,200), np.linspace(-5,5,200))
    grid = np.stack([xx, yy], axis=-1)  # (200,200,2)
    logp_vec = jax.vmap(jax.vmap(log_prob))
    logp_vals = logp_vec(grid)
    ax.contourf(xx, yy, np.exp(np.array(logp_vals)), levels=30, cmap="Blues", alpha=0.4)

    def update(frame):
        scat.set_offsets(history[frame])
        ax.set_title(f"SVGD Step {frame*10}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=history.shape[0], interval=100, blit=True)
    plt.show()


def animate_mpc(all_samples, best_trajs, goals):
    """
    all_samples: list of length TIME_ITER, each (S, N, T, 4)
    best_trajs:  list of length TIME_ITER, each (N, T, 4)
    goals: (N, 2)
    obstacles: (M, 2) array-like or None
    radii: (M,) array-like or None
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 13)

    # Plot goals
    ax.scatter(goals[:, 0], goals[:, 1], c='red', s=60, marker='x', label='Goals')

    # Draw obstacles if provided
    obstacle_patches = []
    obs = np.array(obstacles)
    rad = np.array(radii)
    for (x, y), r in zip(obs, rad):
        circle = Circle((x, y), r, color='black', alpha=0.3)
        ax.add_patch(circle)
        obstacle_patches.append(circle)

    # Get shapes
    S, N, T, _ = all_samples[0].shape
    colors = plt.cm.tab10(np.arange(N) % 10)

    # Lines for best trajectories
    traj_lines = [ax.plot([], [], '-', lw=2, color=colors[i])[0] for i in range(N)]
    robot_dots = [ax.scatter([], [], c=[colors[i]], s=40, zorder=3) for i in range(N)]

    # Sampled trajectories (faint)
    sample_lines = [[ax.plot([], [], '-', lw=1, alpha=0.1, color=colors[i])[0]
                     for _ in range(S)] for i in range(N)]

    def update(frame):
        samples = all_samples[frame]  # (S, N, T, 4)
        best = best_trajs[frame]      # (N, T, 4)

        for i in range(N):
            for s in range(S):
                # traj = samples[s, i]  # (T, 4)
                # sample_lines[i][s].set_data(traj[:, 0], traj[:, 1])
                pass
            traj_lines[i].set_data(best[i, :, 0], best[i, :, 1])
            robot_dots[i].set_offsets(best[i, 0, :2])

        ax.set_title(f"MPC Step {frame}")
        return sum(sample_lines, []) + traj_lines + robot_dots + obstacle_patches

    ani = animation.FuncAnimation(fig, update, frames=len(all_samples),
                                  interval=200, blit=True)
    plt.legend()
    plt.show()