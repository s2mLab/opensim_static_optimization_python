from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from static_optim import kinematic


model_selection = "arm26"  # "arm26" or "gait2392"
muscle_to_evaluate = 3  # idx or "all"
frame_to_evaluate = 80
previous_activation = 0.5

if model_selection == "arm26":
    model = "template/arm26.osim"
    kin = "data/arm26_InverseKinematics.mot"
elif model_selection == "gait2392":
    model = "template/gait2392.osim"
    kin = "data/gait2392_walk1_ik.mot"
else:
    raise RuntimeError("Wrong model selection")

# Load the model
m = kinematic.KinematicModel(model)

# Load a certain kinematics
m.finalize_model(kin, (2, 6))

# Select a random frame
m.upd_model_kinematics(frame_to_evaluate)

# Get some aliases
fs = m.model.getForceSet()
muscles = fs.getMuscles()
if muscle_to_evaluate == "all":
    muscles_idx = range(muscles.getSize())
else:
    muscles_idx = [muscle_to_evaluate]


def generate_force_activation_pattern(x):
    y = np.ndarray((len(x), len(muscles_idx)))
    idx_j = 0
    for j in muscles_idx:
        for idx in range(muscles.getSize()):
            muscles.get(idx).setActivation(m.state, 0)

        for idx, val in enumerate(x):
            muscles.get(j).setActivation(m.state, val)
            try:
                m.model.equilibrateMuscles(m.state)
            except RuntimeError:
                print("Oups..")
            y[idx, idx_j] = muscles.get(j).getTendonForce(m.state)
        idx_j += 1
    return y


# Generate the full force/activation pattern as reference
N_ref = 1000
x_ref = np.linspace(0, 1, N_ref)
y_ref = generate_force_activation_pattern(x_ref)

# Generate the linear force/activation pattern
x_100 = np.array([0., 1.])
y_100 = generate_force_activation_pattern(x_100)

# Generate the previous linear force/activation pattern
x_prev = np.array([0, previous_activation])
y_prev = generate_force_activation_pattern(x_prev)
for i in range(y_prev.shape[1]):  # Project values to 100%
    y_prev[-1, i] = y_prev[-1, i] / previous_activation
x_prev[1] = 1

# Generate the cubic spline force/activation pattern
x_spline = np.linspace(0, 1, 4)
y_spline = generate_force_activation_pattern(x_spline)
# for i range()
spl = interpolate.splrep(x_spline, y_spline[:, 0], k=3)
N_spline = 1000
x_spline_full = np.linspace(0, 1, N_spline)
y_spline_full = interpolate.splev(x_spline_full, spl)

# plot all the data
font = {'family' : 'normal',
        'size'   : 18}
plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.figure("Force against activation")
ax = plt.axes()
# ax.set_title("Force en fonction de l'activation musculaire")
ax.set_xlabel("Muscle activation (\%)")
ax.set_ylabel("Muscle Force (N)")
ax.set_xlim((0, 100))
ax.plot(x_ref * 100, y_ref, '-', color=[0.7, 0.7, 0.7], linewidth=6)
# ax.set_prop_cycle(None)
ax.plot(x_100 * 100, y_100, 'k-.')
# ax.set_prop_cycle(None)
ax.plot(x_prev * 100, y_prev, 'k--')
# ax.set_prop_cycle(None)
ax.plot(x_spline_full * 100, y_spline_full, 'k-')
# ax.set_prop_cycle(None)
plt.legend((r"${SO}^{Trad}$", r"${SO}^{Lin}_{max}$", r"${SO}^{Lin}_{prev}$", r"${SO}^{Spline}$"))
plt.xticks(np.arange(0, 101, step=10))

plt.show()
# plt.savefig('coucou.png', format='png', dpi=300)
