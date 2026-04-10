# integrator.py
import jax
import jax.numpy as jnp
from dynamics import f


@jax.jit
def rk4_step(state, control, dt):
    k1 = f(state, control)
    k2 = f(state + 0.5 * dt * k1, control)
    k3 = f(state + 0.5 * dt * k2, control)
    k4 = f(state + dt * k3, control)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


@jax.jit
def integrate_interval(s_i, u_i, dt_interval, substeps):
    dt = dt_interval / substeps

    def body_fun(step, s):
        return rk4_step(s, u_i, dt)

    # fori_loop returns ONLY the final state
    s_end = jax.lax.fori_loop(0, substeps, body_fun, s_i)
    return s_end
