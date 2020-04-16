# This code is designed for those who wants a deeper control on Static Optimization via Python. Otherwise, you should
# call the proper AnalyzeTool as described in this minimal example:
#
# import opensim as osim
# tool = osim.AnalyzeTool("data/gait2392_Setup_StaticOptimization.xml")
# tool.run()

import os
import time
import matplotlib.pyplot as plt
from cycler import cycler
import ipopt
import numpy as np
from pyomeca import Analogs3d

from static_optim.dynamic_models import StaticOptimization, StaticOptimizationLinearPrevConstraints as LinPrev, \
                                        StaticOptimizationLinearMaxImplementation as LinMax, \
                                        StaticOptimizationCubicSplineConstraints


def get_peak_differences(data_ref, data_to_compare):
    d = data_ref - data_to_compare
    m = d.max()[0]
    activation_at_max = []
    for i in range(d.shape[1]):
        activation_at_max.append(data_ref[:, i, (d[:, i, :] == m[i]).squeeze()][0][0])
    return m, activation_at_max


def static_optimization_example(model_type,
                                calc_classic=True,
                                calc_lin_max=True,
                                calc_lin_prev=True,
                                calc_cubic=True,
                                force_classic_recompute=False,
                                show_results=True,
                                verbose=True,
                                ipopt_print_level=0,
                                number_of_running=1,
                                time_functions=False):
    if model_type == "shoulder":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        setup = {
            "model": f"{dir_path}/template/arm26.osim",
            "mot": f"{dir_path}/data/arm26_InverseKinematics.mot",
            "filter_param": (2, 4),
            "muscle_physiology": True,
            "external_load_xml": None,
            "residual_actuator_xml": None
        }
    elif model_type == "walking":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        setup = {
            "model": f"{dir_path}template/gait2392.osim",
            "mot": f"{dir_path}data/gait2392_walk1_ik.mot",
            "filter_param": (2, 6),
            "muscle_physiology": True,
            # "external_load_xml": f"{dir_path}data/gait2392_walk1_grf.xml",
            "residual_actuator_xml": f"{dir_path}data/gait2392_actuators.xml"
        }
    else:
        raise RuntimeError("Wrong model type selected")

    model_for_frame = None
    # Classic static optimization model
    if calc_classic:
        model_classic = StaticOptimization(setup["model"], setup["mot"], setup["filter_param"],
                                           residual_actuator_xml=setup["residual_actuator_xml"])
        model_for_frame = model_classic
        if not force_classic_recompute and os.path.isfile(f'{dir_path}/{model_type}_data_classic_all.npy'):
            # If possible load the data since classic is very slow
            x0_classic_all = np.load(f'{dir_path}/{model_type}_data_classic_all.npy')
            info_classic = dict()
            info_classic["obj_val"] = 0
            info_classic["g"] = 0

        else:
            x0_classic_all = []
            force_classic_recompute = True

            # optimization options
            lb_classic, ub_classic = model_classic.get_bounds()

            # problem
            problem_classic = ipopt.problem(
                n=model_classic.n_actuators,  # Nb of variables
                lb=lb_classic,  # Variables lower bounds
                ub=ub_classic,  # Variables upper bounds
                m=model_classic.n_dof,  # Nb of constraints
                cl=np.zeros(model_classic.n_dof),  # Lower bound constraints
                cu=np.zeros(model_classic.n_dof),  # Upper bound constraints
                problem_obj=model_classic,  # Class that defines the problem
            )
            problem_classic.addOption("tol", 1e-7)
            problem_classic.addOption("print_level", ipopt_print_level)
            n_frames = int(model_classic.n_frame)

    # linPrev static optimization model
    if calc_lin_prev:
        model_lin_prev = LinPrev(
            setup["model"],
            setup["mot"],
            setup["filter_param"],
            muscle_physiology=setup["muscle_physiology"],
            residual_actuator_xml=setup["residual_actuator_xml"]
        )

        # optimization options
        lb_lin_prev, ub_lin_prev = model_lin_prev.get_bounds()

        # problem
        problem_lin_prev = ipopt.problem(
            n=model_lin_prev.n_actuators,  # Nb of variables
            lb=lb_lin_prev,  # Variables lower bounds
            ub=ub_lin_prev,  # Variables upper bounds
            m=model_lin_prev.n_dof,  # Nb of constraints
            cl=np.zeros(model_lin_prev.n_dof),  # Lower bound constraints
            cu=np.zeros(model_lin_prev.n_dof),  # Upper bound constraints
            problem_obj=model_lin_prev,  # Class that defines the problem
        )
        problem_lin_prev.addOption("tol", 1e-7)
        problem_lin_prev.addOption("print_level", ipopt_print_level)
        model_for_frame = model_lin_prev

    # LinMax static optimization model
    if calc_lin_max:
        model_lin_max = LinMax(
            setup["model"],
            setup["mot"],
            setup["filter_param"],
            muscle_physiology=setup["muscle_physiology"],
            residual_actuator_xml=setup["residual_actuator_xml"]
        )

        # optimization options
        lb_lin_max, ub_lin_max = model_lin_max.get_bounds()

        # problem
        problem_lin_max = ipopt.problem(
            n=model_lin_max.n_actuators,  # Nb of variables
            lb=lb_lin_max,  # Variables lower bounds
            ub=ub_lin_max,  # Variables upper bounds
            m=model_lin_max.n_dof,  # Nb of constraints
            cl=np.zeros(model_lin_max.n_dof),  # Lower bound constraints
            cu=np.zeros(model_lin_max.n_dof),  # Upper bound constraints
            problem_obj=model_lin_max,  # Class that defines the problem
        )
        problem_lin_max.addOption("tol", 1e-7)
        problem_lin_max.addOption("print_level", ipopt_print_level)
        model_for_frame = model_lin_max

    if calc_cubic:
        # Cubic static optimization model
        model_cubic = StaticOptimizationCubicSplineConstraints(
            setup["model"],
            setup["mot"],
            setup["filter_param"],
            residual_actuator_xml=setup["residual_actuator_xml"]
        )

        # optimization options
        lb_cubic, ub_cubic = model_cubic.get_bounds()

        # problem
        problem_cubic = ipopt.problem(
            n=model_cubic.n_actuators,  # Nb of variables
            lb=lb_cubic,  # Variables lower bounds
            ub=ub_cubic,  # Variables upper bounds
            m=model_cubic.n_dof,  # Nb of constraints
            cl=np.zeros(model_cubic.n_dof),  # Lower bound constraints
            cu=np.zeros(model_cubic.n_dof),  # Upper bound constraints
            problem_obj=model_cubic,  # Class that defines the problem
        )
        problem_cubic.addOption("tol", 1e-7)
        problem_cubic.addOption("print_level", ipopt_print_level)
        model_for_frame = model_cubic

    if model_for_frame is None:
        raise RuntimeError('No algorithm was selected')

    # Prepare running time
    if time_functions:
        running_time_classic = np.zeros((number_of_running,))
        running_time_lin_max = np.zeros((number_of_running,))
        running_time_lin_prev = np.zeros((number_of_running,))
        running_time_cubic = np.zeros((number_of_running,))

    for n in range(number_of_running):
        if verbose:
            print(f"** Passage number {n} **")

        # set initial guesses
        if calc_classic:
            activation_initial_guess_classic = np.zeros([model_classic.n_actuators])
            activations_classic = []
        if calc_lin_max:
            activation_initial_guess_lin_max = np.zeros([model_lin_max.n_actuators])
            activations_lin_max = []
        if calc_lin_prev:
            activation_initial_guess_lin_prev = np.zeros([model_lin_prev.n_actuators]) + 1
            activations_lin_prev = []
        if calc_cubic:
            activation_initial_guess_cubic = np.zeros([model_cubic.n_actuators]) + 1
            activations_cubic = []

        for iframe in range(0, int(model_for_frame.n_frame)):
            if time_functions:
                start_time = time.time()
            if verbose:
                print(f'frame: {iframe} | time: {model_for_frame.get_time(iframe)}')

            # Reference
            if calc_classic:
                if force_classic_recompute:
                    try:
                        if time_functions:
                            t = time.time()
                        model_classic.upd_model_kinematics(iframe)
                        x_classic, info_classic = problem_classic.solve(activation_initial_guess_classic)
                        if time_functions:
                            running_time_classic[n] += time.time() - t
                    except RuntimeError:
                        print(f"Error while computing the frame #{iframe}")
                    x0_classic_all.append(x_classic)
                else:
                    x_classic = x0_classic_all[iframe]

            # lin_prev optim
            if calc_lin_prev:
                try:
                    if time_functions:
                        t = time.time()
                    model_lin_prev.set_previous_activation(activation_initial_guess_lin_prev)
                    model_lin_prev.upd_model_kinematics(iframe)
                    x_lin_prev, info_lin_prev = problem_lin_prev.solve(activation_initial_guess_lin_prev)
                    if time_functions:
                        running_time_lin_prev[n] += time.time() - t
                except RuntimeError:
                    print(f"Error while computing the frame #{iframe}")

            # LinMax optim
            if calc_lin_max:
                try:
                    if time_functions:
                        t = time.time()
                    model_lin_max.set_previous_activation(activation_initial_guess_lin_max)
                    model_lin_max.upd_model_kinematics(iframe)
                    x_lin_max, info_lin_max = problem_lin_max.solve(activation_initial_guess_lin_max)
                    if time_functions:
                        running_time_lin_max[n] += time.time() - t
                except RuntimeError:
                    print(f"Error while computing the frame #{iframe}")

            # cubic optim
            if calc_cubic:
                try:
                    if time_functions:
                        t = time.time()
                    model_cubic.upd_model_kinematics(iframe)
                    x_cubic, info_cubic = problem_cubic.solve(activation_initial_guess_cubic)
                    if time_functions:
                        running_time_cubic[n] += time.time() - t
                except RuntimeError:
                    print(f"Error while computing the frame #{iframe}")

            # the output is the initial guess for next frame
            if calc_classic:
                activation_initial_guess_classic = x_classic
                activations_classic.append(x_classic)
                if verbose:
                    print(f'x_classic = {x_classic}')
            if calc_lin_prev:
                activation_initial_guess_lin_prev = x_lin_prev
                activations_lin_prev.append(x_lin_prev)
                if verbose:
                    print(f'x_lin_prev       = {x_lin_prev}')
            if calc_lin_max:
                activation_initial_guess_lin_max = x_lin_max
                activations_lin_max.append(x_lin_max)
                if verbose:
                    print(f'x_lin_max       = {x_lin_max}')
            if calc_cubic:
                activation_initial_guess_cubic = x_cubic
                activations_cubic.append(x_cubic)
                if verbose:
                    print(f'x_cubic    = {x_cubic}')
            if verbose and calc_classic and calc_lin_prev and calc_lin_max and calc_cubic:
                print(f'Diff lin_prev    = {x_classic - x_lin_prev}')
                print(f'Diff lin_max    = {x_classic - x_lin_max}')
                print(f'Diff cubic = {x_classic - x_cubic}')
            if verbose:
                print('')
        if verbose:
            print('')
            print('')

    # Save the classic data
    if calc_classic and force_classic_recompute:
        np.save(f'{dir_path}/{model_type}_data_classic_all', x0_classic_all)

    data = {}
    if calc_classic:
        data['classic'] = Analogs3d(np.array(activations_classic)) * 100
    if calc_lin_max:
        data['lin_max'] = Analogs3d(np.array(activations_lin_max)) * 100
    if calc_lin_prev:
        data['lin_prev'] = Analogs3d(np.array(activations_lin_prev)) * 100
    if calc_cubic:
        data['cubic'] = Analogs3d(np.array(activations_cubic)) * 100

    # Analyses
    if calc_classic and calc_lin_prev and calc_lin_max and calc_cubic:
        if time_functions:
            # Show total time
            print("")
            print("Total time")
            print(f"classic = {running_time_classic}")
            print(f"lin_prev = {running_time_lin_prev}")
            print(f"lin_max = {running_time_lin_max}")
            print(f"cubic = {running_time_cubic}")
            print("")
            print("Mean time per frame")
            print(f"classic = {running_time_classic.mean()/model_lin_prev.n_frame} ± {np.std(running_time_classic)/model_lin_prev.n_frame}")
            print(f"lin_prev = {running_time_lin_prev.mean()/model_lin_prev.n_frame} ± {np.std(running_time_lin_prev)/model_lin_prev.n_frame}")
            print(f"lin_max = {running_time_lin_max.mean()/model_lin_prev.n_frame} ± {np.std(running_time_lin_max)/model_lin_prev.n_frame}")
            print(f"cubic = {running_time_cubic.mean()/model_lin_prev.n_frame} ± {np.std(running_time_cubic)/model_lin_prev.n_frame}")
            print("")

        # Compute RMSE
        if calc_classic:
            print("")
            print("RMSE")
            print(f"SO_lin_max = \t{(data['classic'] - data['lin_max']).rms().squeeze()}")
            print(f"SO_lin_prev = \t{(data['classic'] - data['lin_prev']).rms().squeeze()}")
            print(f"SO_cubic = \t\t{(data['classic'] - data['cubic']).rms().squeeze()}")
            print("")

        # Peak differences
        print("")
        print("Peak differences")
        max_lin_max, activation_at_max_lin_max = get_peak_differences(data['classic'], data['lin_max'])
        max_lin_prev, activation_at_max_lin_prev = get_peak_differences(data['classic'], data['lin_prev'])
        max_cubic, activation_at_max_cubic = get_peak_differences(data['classic'], data['cubic'])
        print(f"SO_lin_max = \t{max_lin_max} at {activation_at_max_lin_max}")
        print(f"SO_lin_prev = \t{max_lin_prev} at {activation_at_max_lin_prev}")
        print(f"SO_cubic = \t{max_cubic} at {activation_at_max_cubic}")
        print("")

    # Show the data
    if show_results:
        new_prop_cycle = cycler('color', ['r', 'g', 'b', 'k', 'c', 'm'])

        # Plot 1
        # Prepare axes
        plt.figure()
        ax = plt.axes()
        ax.set_title("Muscle activations generated by the \ndifferent static optimization algorithms")
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Muscle activation (%)")

        # Prepare legend
        if calc_classic:
            data["classic"][:, 0:1, :].plot('k-', ax=ax)
        if calc_lin_max:
            data["lin_max"][:, 0:1, :].plot('k--', ax=ax)
        if calc_lin_prev:
            data["lin_prev"][:, 0:1, :].plot('k-.', ax=ax)
        if calc_cubic:
            data["cubic"][:, 0:1, :].plot('k.-', ax=ax)
        ax.set_prop_cycle(new_prop_cycle)

        # Plot actual data
        legend_name = []
        if calc_classic:
            data["classic"].plot('-', ax=ax)
            legend_name.append("SO^{Ref}")
        if calc_lin_max:
            data["lin_max"].plot('--', ax=ax)
            legend_name.append("SO^{Lin}_{max}")
        if calc_lin_prev:
            data["lin_prev"].plot('-.', ax=ax)
            legend_name.append("SO^{Lin}_{prev}")
        if calc_cubic:
            data["cubic"].plot('.-', ax=ax)
            legend_name.append("SO^{Spline}")
        for m in ("Muscle1", "Muscle2", "Muscle3", "Muscle4", "Muscle5", "Muscle6"):
            legend_name.append(m)
        ax.legend(legend_name)

        # Plot 2
        if calc_classic:
            # Prepare axes
            plt.figure()
            ax2 = plt.axes()
            ax2.set_title("Absolute difference between algorithms and SO_ref")
            ax2.set_xlabel("Time (frame)")
            ax2.set_ylabel("Absolute difference (%)")

            # Prepare for legend
            if calc_lin_max:
                (data["classic"] - data["lin_max"])[:, 0:1, :].plot('k--', ax=ax2)
            if calc_lin_prev:
                (data["classic"] - data["lin_prev"])[:, 0:1, :].plot('k-.', ax=ax2)
            if calc_cubic:
                (data["classic"] - data["cubic"])[:, 0:1, :].plot('k-', ax=ax2)
            ax2.set_prop_cycle(new_prop_cycle)
            vide = Analogs3d(np.zeros((1, 6, 2))*np.nan)
            vide.plot('-', ax=ax2)
            ax2.set_prop_cycle(new_prop_cycle)

            # Plot actual data
            if calc_lin_max:
                (data["classic"] - data["lin_max"]).plot('--', ax=ax2)
            if calc_lin_prev:
                (data["classic"] - data["lin_prev"]).plot('-.', ax=ax2)
            if calc_cubic:
                (data["classic"] - data["cubic"]).plot('-', ax=ax2)
            ax2.legend(legend_name[1:])

        # Plot paper (saved at dpi=300)
        # Prepare axes
        text_size = 20
        plt.figure()
        ax = plt.axes()
        ax.set_title("Muscle activation of the $\it{Biceps'\ long\ head}$\noptimized by the different SO algorithms")
        ax.set_xlabel("Time (second)")
        ax.set_ylabel("$\it{Biceps'\ long\ head}$ activation (%)")
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(text_size)

        # Prepare legend
        t = [model_for_frame.get_time(i) for i in range(int(model_for_frame.n_frame))]
        if calc_classic:
            data["classic"][:, 3:4, :].plot('-', x=t, ax=ax, color=[0.7, 0.7, 0.7], linewidth=6)
        if calc_lin_max:
            data["lin_max"][:, 3:4, :].plot('k-', x=t, ax=ax)
        if calc_lin_prev:
            data["lin_prev"][:, 3:4, :].plot('k--', x=t, ax=ax)
        if calc_cubic:
            data["cubic"][:, 3:4, :].plot('k.-', x=t, ax=ax, markersize=5)
        ax.set_prop_cycle(new_prop_cycle)
        ax.legend(legend_name[:len(ax.get_lines())])
        ax.legend([r'$SO^{Ref}$', r'$SO^{Lin}_{max}$', r'$SO^{Lin}_{prev}$', r'$SO^{Spline}$'], prop={'size': text_size})
        # plt.savefig('coucou.png', format='png', dpi=300)
        for item in (ax.get_legend().get_texts()):
            item.set_fontsize(text_size)

        # Prepare legend
        t = [model_for_frame.get_time(i) for i in range(int(model_for_frame.n_frame))]
        if calc_classic:
            data["classic"][:, [0], :].plot('-', x=t, ax=ax, color=[0.7, 0.7, 0.7], linewidth=6)
        if calc_lin_max:
            data["lin_max"][:, [0], :].plot('k-', x=t, ax=ax)
        if calc_lin_prev:
            data["lin_prev"][:, [0], :].plot('k--', x=t, ax=ax)
        if calc_cubic:
            data["cubic"][:, [0], :].plot('k.-', x=t, ax=ax, markersize=5)

        # Show
        plt.show()


if __name__ == "__main__":
    static_optimization_example('shoulder')
