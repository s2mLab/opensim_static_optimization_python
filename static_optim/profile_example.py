import cProfile
import numpy as np
from scipy import stats

from tests.example import static_optimization_example


def run_code(*args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    static_optimization_example(*args, **kwargs)
    pr.disable()
    for val in pr.getstats():
        if isinstance(val.code, str) and "<method 'solve' of 'cyipopt.problem' objects>" == val.code:
            total_time = val[3]
        if hasattr(val.code, 'co_name') and "jacobian" == val.code.co_name:
            jaco_time = val[3]
    return total_time, jaco_time

n_frames = 120
classic_total_time = []
classic_jaco_time = []
lin_max_total_time = []
lin_max_jaco_time = []
lin_prev_total_time = []
lin_prev_jaco_time = []
cubic_total_time = []
cubic_jaco_time = []
for i in range(30):
    print(f"*** Iter = {i} ***")
    print("Classic")
    total, jaco = run_code("shoulder", calc_classic=True, calc_lin_max=False, calc_lin_prev=False, calc_cubic=False,
                           force_classic_recompute=True, show_results=False, verbose=False)
    classic_total_time.append(total)
    classic_jaco_time.append(jaco)

    print("LinMax")
    total, jaco = run_code("shoulder", calc_classic=False, calc_lin_max=True, calc_lin_prev=False, calc_cubic=False,
                           force_classic_recompute=False, show_results=False, verbose=False)
    lin_max_total_time.append(total)
    lin_max_jaco_time.append(jaco)

    print("LinPrev")
    total, jaco = run_code("shoulder", calc_classic=False, calc_lin_max=False, calc_lin_prev=True, calc_cubic=False,
                           force_classic_recompute=False, show_results=False, verbose=False)
    lin_prev_total_time.append(total)
    lin_prev_jaco_time.append(jaco)

    print("Cubic")
    total, jaco = run_code("shoulder", calc_classic=False, calc_lin_max=False, calc_lin_prev=False, calc_cubic=True,
                           force_classic_recompute=False, show_results=False, verbose=False)
    cubic_total_time.append(total)
    cubic_jaco_time.append(jaco)

# Print results
print(f"classic_total_time = {classic_total_time}")
print(f"classic_jaco_time = {classic_jaco_time}")
print(f"lin_max_total_time = {lin_max_total_time}")
print(f"lin_max_jaco_time = {lin_max_jaco_time}")
print(f"lin_prev_total_time = {lin_prev_total_time}")
print(f"lin_prev_jaco_time = {lin_prev_jaco_time}")
print(f"cubic_total_time = {cubic_total_time}")
print(f"cubic_jaco_time = {cubic_jaco_time}")

# Analyse them
classic_total_time = np.array(classic_total_time)
classic_jaco_time = np.array(classic_jaco_time)
lin_max_total_time = np.array(lin_max_total_time)
lin_max_jaco_time = np.array(lin_max_jaco_time)
lin_prev_total_time = np.array(lin_prev_total_time)
lin_prev_jaco_time = np.array(lin_prev_jaco_time)
cubic_total_time = np.array(cubic_total_time)
cubic_jaco_time = np.array(cubic_jaco_time)

print("Mean time per frame")
print(f"classic = {classic_total_time.mean()/n_frames} ± {np.std(classic_total_time)/n_frames}")
print(f"lin_max = {lin_max_total_time.mean()/n_frames} ± {np.std(lin_max_total_time)/n_frames}")
print(f"lin_prev = {lin_prev_total_time.mean()/n_frames} ± {np.std(lin_prev_total_time)/n_frames}")
print(f"cubic = {cubic_total_time.mean()/n_frames} ± {np.std(cubic_total_time)/n_frames}")
print("")
print("Mean time jacobian per frame")
print(f"classic = {classic_jaco_time.mean()/n_frames} ± {np.std(classic_jaco_time)/n_frames}")
print(f"lin_max = {lin_max_jaco_time.mean()/n_frames} ± {np.std(lin_max_jaco_time)/n_frames}")
print(f"lin_prev = {lin_prev_jaco_time.mean()/n_frames} ± {np.std(lin_prev_jaco_time)/n_frames}")
print(f"cubic = {cubic_jaco_time.mean()/n_frames} ± {np.std(cubic_jaco_time)/n_frames}")
print("")

# Statictics
f_value, p_value = stats.f_oneway(classic_total_time, lin_max_total_time, lin_prev_total_time, cubic_total_time)
all_p = []
if p_value < 0.05:
    _, p = stats.ttest_ind(classic_total_time, lin_max_total_time)
    all_p.append(p)
    _, p = stats.ttest_ind(classic_total_time, lin_prev_total_time)
    all_p.append(p)
    _, p = stats.ttest_ind(classic_total_time, cubic_total_time)
    all_p.append(p)
    _, p = stats.ttest_ind(lin_max_total_time, lin_prev_total_time)
    all_p.append(p)
    _, p = stats.ttest_ind(lin_max_total_time, cubic_total_time)
    all_p.append(p)
    _, p = stats.ttest_ind(lin_prev_total_time, cubic_total_time)
    all_p.append(p)
print(f"All p = {all_p}")
print("")

# # Results from 30 runs
# classic_total_time = np.array([83.11248499999999, 83.682013, 83.568028, 84.205794, 83.808047, 84.078915, 83.779837, 84.30575499999999, 84.076386, 84.22316599999999, 83.907457, 84.143832, 84.225459, 84.397093, 83.776425, 84.296764, 84.01164, 84.32203299999999, 84.523635, 84.591757, 84.24759999999999, 84.912116, 84.877738, 84.73124, 84.848832, 85.01159, 84.770671, 84.878255, 85.28107299999999, 85.34855])*1000
# classic_jaco_time = np.array([56.374393999999995, 56.758523, 56.679351999999994, 57.152947999999995, 56.873798, 57.082525999999994, 56.843191999999995, 57.287119, 57.098546999999996, 57.227971, 56.974267, 57.157336, 57.238008, 57.369284, 56.920035, 57.258016, 57.049932, 57.265195, 57.356165, 57.40562, 57.129649, 57.561493999999996, 57.537313999999995, 57.414502999999996, 57.481480999999995, 57.591432999999995, 57.41569, 57.545362999999995, 57.783350999999996, 57.826823])*1000
# lin_max_total_time = np.array([0.561412, 0.557449, 0.557352, 0.5567219999999999, 0.55409, 0.5577529999999999, 0.553493, 0.558167, 0.55519, 0.5558339999999999, 0.556466, 0.557914, 0.556234, 0.555145, 0.55916, 0.555337, 0.5565559999999999, 0.56009, 0.556334, 0.559443, 0.561817, 0.5639219999999999, 0.561979, 0.564253, 0.5620339999999999, 0.560944, 0.561793, 0.5629259999999999, 0.5716359999999999, 0.562528])*1000
# lin_max_jaco_time = np.array([0.000441, 0.00044899999999999996, 0.000491, 0.000472, 0.00044899999999999996, 0.000481, 0.00048199999999999995, 0.00045799999999999997, 0.000486, 0.000464, 0.0005099999999999999, 0.000489, 0.00045799999999999997, 0.00047, 0.00047099999999999996, 0.0004969999999999999, 0.000461, 0.000457, 0.00047799999999999996, 0.000464, 0.00045599999999999997, 0.000486, 0.0004919999999999999, 0.000477, 0.00047799999999999996, 0.000455, 0.00046499999999999997, 0.0005, 0.000491, 0.000493])*1000
# lin_prev_total_time = np.array([0.6008279999999999, 0.589464, 0.588375, 0.588616, 0.585216, 0.5879789999999999, 0.5824, 0.588963, 0.584105, 0.596061, 0.586266, 0.584996, 0.588684, 0.585263, 0.589104, 0.583979, 0.586, 0.592382, 0.588546, 0.5926049999999999, 0.5912189999999999, 0.593631, 0.593958, 0.595756, 0.595077, 0.594111, 0.594804, 0.597996, 0.598765, 0.597418])*1000
# lin_prev_jaco_time = np.array([0.00047599999999999997, 0.000463, 0.000491, 0.0005009999999999999, 0.00044899999999999996, 0.000474, 0.000459, 0.000505, 0.00046499999999999997, 0.00046899999999999996, 0.000493, 0.000457, 0.00045799999999999997, 0.000459, 0.000507, 0.000454, 0.000445, 0.00046699999999999997, 0.00047799999999999996, 0.00048699999999999997, 0.00045799999999999997, 0.000463, 0.000488, 0.000502, 0.000473, 0.000491, 0.000489, 0.0004919999999999999, 0.000498, 0.00048499999999999997])*1000
# cubic_total_time = np.array([1.2054129999999998, 1.181895, 1.175359, 1.177573, 1.181163, 1.181719, 1.174396, 1.181058, 1.180505, 1.172558, 1.175112, 1.181726, 1.177916, 1.176218, 1.174803, 1.176096, 1.18227, 1.1875369999999998, 1.185296, 1.188105, 1.181468, 1.189533, 1.1920549999999999, 1.19275, 1.19281, 1.195606, 1.1953289999999999, 1.197168, 1.199775, 1.194623])*1000
# cubic_jaco_time = np.array([0.139341, 0.136344, 0.135256, 0.13536499999999999, 0.135461, 0.136073, 0.13544799999999999, 0.135756, 0.134848, 0.134062, 0.13372799999999999, 0.13477999999999998, 0.134879, 0.135217, 0.13511099999999998, 0.13411399999999998, 0.135077, 0.135654, 0.134804, 0.13508699999999998, 0.134967, 0.135331, 0.135884, 0.134952, 0.135205, 0.13621999999999998, 0.135965, 0.135599, 0.136323, 0.135438])*1000
