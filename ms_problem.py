# ms_problem.py
import jax
import jax.numpy as jnp
from config import *
from integrator import integrate_interval


def unpack_vars(vars):
    N_states = (K+1)*state_dim
    N_controls = K*control_dim

    states_flat = vars[:N_states]
    controls_flat = vars[N_states:N_states+N_controls]
    tf = vars[-1]

    states = states_flat.reshape((K+1, state_dim))
    controls = controls_flat.reshape((K, control_dim))
    return states, controls, tf


@jax.jit
def objective(vars):
    states, controls, tf = unpack_vars(vars)
    return -states[-1, 5]   # maximize final mass


@jax.jit
def constraints(vars):
    states, controls, tf = unpack_vars(vars)
    dt_interval = tf / K

    cons = []

    # Initial state constraints
    s0 = states[0]
    cons.append(s0[0] - x_0)
    cons.append(s0[1] - z_0)
    cons.append(s0[2] - vx_0)
    cons.append(s0[3] - vz_0)
    cons.append(s0[4] - theta_0)
    cons.append(s0[5] - m0)

    # Multiple shooting continuity
    for i in range(K):
        s_i = states[i]
        u_i = controls[i]
        s_end = integrate_interval(s_i, u_i, dt_interval, S)

        cons.extend(list(s_end - states[i+1]))

    # Final state constraints
    sN = states[-1]
    cons.append(sN[0] - x_tf)  # x_tf = 0.0
    cons.append(sN[1] - z_tf)
    cons.append(sN[2] - vx_tf)
    cons.append(sN[3] - vz_tf)
    cons.append(sN[4] - theta_tf)

    return jnp.array(cons)
