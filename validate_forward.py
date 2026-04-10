# validate_and_plot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from integrator import integrate_interval
from config import K, S, state_dim, control_dim


def load_data(states_csv="ms_states.csv", controls_csv="ms_controls.csv"):
    df_s = pd.read_csv(states_csv)
    df_u = pd.read_csv(controls_csv)

    states = df_s[["x","z","vx","vz","theta","mass"]].values
    controls = df_u[["u1","u2"]].values
    time = df_s["t"].values
    tf = time[-1]

    return states, controls, time, tf


def forward_integrate(states, controls, tf):
    """Replay the trajectory by integrating forward from the first state."""
    dt = tf / K
    s = states[0].astype(float)

    integrated_states = [s.copy()]
    errors = []

    for i in range(K):
        u = controls[i]
        s_end = np.array(integrate_interval(s, u, dt, S), float)
        integrated_states.append(s_end.copy())

        err = np.linalg.norm(s_end - states[i+1])
        errors.append(err)

        s = s_end

    integrated_states = np.vstack(integrated_states)
    return integrated_states, np.array(errors)


def plot_comparison(states_opt, states_int, controls_opt, time):
    """Plot optimized vs integrated states & controls."""
    
    # ========== 1) TRAJECTORY PLOT (X vs Z) ==========
    plt.figure(figsize=(6,6))
    plt.plot(states_opt[:,0], states_opt[:,1], 'b-', label="Optimized")
    plt.plot(states_int[:,0], states_int[:,1], 'r--', label="Integrated")
#   plt.gca().invert_yaxis()
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("Trajectory (X vs Z)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trajectory_comparison.png")

    # ========== 2) STATES ==========
    labels = ["x", "z", "vx", "vz", "theta", "mass"]
    plt.figure(figsize=(12,10))
    for i in range(6):
        plt.subplot(3,2,i+1)
        plt.plot(time, states_opt[:,i], 'b-', label="Optimized")
        plt.plot(time, states_int[:,i], 'r--', label="Integrated")
        plt.xlabel("Time [s]")
        plt.ylabel(labels[i])
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig("states_comparison.png")

    # ========== 3) CONTROLS ==========
    plt.figure(figsize=(10,6))
    mid_time = 0.5*(time[:-1] + time[1:])

    plt.subplot(2,1,1)
    plt.step(mid_time, controls_opt[:,0], 'b-', where='mid', label="u1 optimized")
    plt.xlabel("Time [s]")
    plt.ylabel("u1")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.step(mid_time, controls_opt[:,1], 'b-', where='mid', label="u2 optimized")
    plt.xlabel("Time [s]")
    plt.ylabel("u2")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("controls.png")


def main():
    print("Loading optimized trajectory...")
    states_opt, controls_opt, time, tf = load_data()

    print("Forward integrating...")
    states_int, errors = forward_integrate(states_opt, controls_opt, tf)

    print("Max per-interval error =", np.max(errors))
    print("Mean per-interval error =", np.mean(errors))
    print("Final optimized state =", states_opt[-1])
    print("Final integrated state =", states_int[-1])

    print("Plotting...")
    plot_comparison(states_opt, states_int, controls_opt, time)
    plt.show()


if __name__ == "__main__":
    main()
