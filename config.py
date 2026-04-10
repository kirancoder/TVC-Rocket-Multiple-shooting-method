# config.py
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Discretization
K = 40        # number of shooting intervals
S = 40         # RK4 sub-steps per interval (increase for accuracy)

# Problem constants
g = 1.62
g0 = 9.81

I_sp = 311.0
m0 = 10000.0

U1_max = 1.0
U1_min = 0.0

U2_max = 1.0
U2_min = -1.0

t_max = 500.0

# initial & final
x_0, z_0, vx_0, vz_0, theta_0_deg = 0.0, 500.0, 10.0, 10.0, 20.0
theta_0 = jnp.deg2rad(theta_0_deg)
x_tf, z_tf, vx_tf, vz_tf, theta_tf = 0.0, 0.0, 0.0, 0.0, 0.0

c1 = 44000.0
c2 = I_sp * g0
c3 = 0.0698

state_dim = 6
control_dim = 2
