# python - << 'EOF'
from initial_guess import build_initial_guess
from ms_problem import unpack_vars, constraints
from integrator import integrate_interval
import numpy as np
from config import K

x0 = build_initial_guess(verbose=False)
states, controls, tf = unpack_vars(x0)
dt = tf / K

# compute ONLY continuity residuals (no final constraints)
errs = []
for i in range(K):
    s_i = states[i]
    u_i = controls[i]
    s_end = integrate_interval(s_i, u_i, dt, 40)
    errs.append(np.linalg.norm(s_end - states[i+1]))

print("max continuity (dynamics only) =", max(errs))

# check final-state residuals explicitly
sN = states[-1]
print("Final state z error:", sN[1])
print("Final vx error:", sN[2])
print("Final vz error:", sN[3])
print("Final theta error:", sN[4])
# EOF
