import pandas as pd
import matplotlib.pyplot as plt

# Load solution
df = pd.read_csv("ms_traj_full.csv")

time = df["t"]

# ----------------------------
# 1. State: X position
# ----------------------------
plt.figure()
plt.plot(time, df["x"])
plt.xlabel("Time (s)")
plt.ylabel("X Position (m)")
plt.title("X Position vs Time")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 2. State: Z position
# ----------------------------
plt.figure()
plt.plot(time, df["z"])
plt.xlabel("Time (s)")
plt.ylabel("Z Position (m)")
plt.title("Z Position vs Time")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 3. Velocities vx
# ----------------------------
plt.figure()
plt.plot(time, df["vx"])
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Velocity vx (m/s)")
plt.title("vx vs Time")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 4. Velocities vz
# ----------------------------
plt.figure()
plt.plot(time, df["vz"])
plt.xlabel("Time (s)")
plt.ylabel("Vertical Velocity vz (m/s)")
plt.title("vz vs Time")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 5. Pitch angle theta
# ----------------------------
plt.figure()
plt.plot(time, df["theta"])
plt.xlabel("Time (s)")
plt.ylabel("Pitch Angle (rad)")
plt.title("Theta vs Time")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 6. Mass trajectory
# ----------------------------
plt.figure()
plt.plot(time, df["mass"])
plt.xlabel("Time (s)")
plt.ylabel("Mass (kg)")
plt.title("Vehicle Mass vs Time")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 7. Control u1
# ----------------------------
plt.figure()
plt.plot(time, df["u1"])
plt.xlabel("Time (s)")
plt.ylabel("Thrust u1 (N)")
plt.title("Thrust Command u1 vs Time")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 8. Control u2
# ----------------------------
plt.figure()
plt.plot(time, df["u2"])
plt.xlabel("Time (s)")
plt.ylabel("Pitch Rate u2 (rad/s)")
plt.title("Pitch Rate Command u2 vs Time")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 9. 2D trajectory (optional)
# ----------------------------
plt.figure()
plt.plot(df["x"], df["z"])
plt.xlabel("X Position (m)")
plt.ylabel("Z Position (m)")
plt.title("2D Trajectory: X vs Z")
plt.grid(True)
plt.tight_layout()

plt.show()
