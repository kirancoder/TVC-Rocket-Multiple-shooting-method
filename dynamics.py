# dynamics.py
import jax.numpy as jnp
from config import g, c1, c2, c3

def f(state, control):
    x, z, vx, vz, theta, m = state
    u1, u2 = control

    x_dot = vx
    z_dot = vz
    vx_dot = c1* u1 * jnp.sin(theta) / m
    vz_dot = c1 * u1 * jnp.cos(theta) / m - g
    theta_dot = c3 * u2
    m_dot = - (u1 * c1/ c2)

    return jnp.stack([x_dot, z_dot, vx_dot, vz_dot, theta_dot, m_dot])
