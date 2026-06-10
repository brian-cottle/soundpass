from scipy.spatial.transform import Rotation as R
import numpy as np

w, x, y, z = 0.98, 0.1, 0.1, 0.1 # Some arbitrary quaternion

# Correct Scipy way
q1 = np.array([x, y, z, w])
q1 = q1 / np.linalg.norm(q1)
r1 = R.from_quat(q1)
e1 = r1.as_euler('xyz', degrees=True)
print(f"Correct: Pitch {-e1[0]:.1f}, Roll {e1[1]:.1f}, Yaw {e1[2]:.1f}")

# Buggy Picoscope way
q2 = np.array([w, x, y, z])
q2 = q2 / np.linalg.norm(q2)
r2 = R.from_quat(q2)
e2 = r2.as_euler('xyz', degrees=True)
print(f"Buggy Picoscope: Pitch {-e2[0]:.1f}, Roll {e2[1]:.1f}, Yaw {e2[2]:.1f}")

