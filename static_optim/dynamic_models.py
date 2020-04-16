"""
Dynamic models.

These are the actual classes to send to IPOPT.
"""

import numpy as np
import opensim as osim
from scipy import interpolate

from static_optim.constraints import ConstraintAccelerationTarget
from static_optim.forces import ResidualForces, ExternalForces
from static_optim.kinematic import KinematicModel
from static_optim.objective import ObjMinimizeActivation


class StaticOptimization(
    KinematicModel,
    ObjMinimizeActivation,
    ConstraintAccelerationTarget,
    ResidualForces,
    ExternalForces,
):
    """
    Computes the muscle activations in order to minimize them while targeting the acceleration from inverse kinematics.
    This is the most classical approach to Static Opimization.
    """

    # TODO: write tests
    def __init__(
        self,
        model_path,
        mot,
        filter_param=None,
        activation_exponent=2,
        residual_actuator_xml=None,
        external_load_xml=None,
    ):
        KinematicModel.__init__(self, model_path)
        ObjMinimizeActivation.__init__(self, activation_exponent)
        ConstraintAccelerationTarget.__init__(self)
        ResidualForces.__init__(self, residual_actuator_xml)
        ExternalForces.__init__(self, external_load_xml)
        self.finalize_model(mot, filter_param)

    def forward_dynamics(self, x, n_times=0):
        # set residual forces
        fs = self.model.getForceSet()
        for i in range(self.n_muscles, fs.getSize()):
            coord = osim.CoordinateActuator.safeDownCast(fs.get(i))
            if coord:
                coord.setOverrideActuation(self.state, x[i] * x[i])

        # update muscles
        muscle_activation = x[: self.n_muscles]
        for m in range(self.n_muscles):
            self.muscle_actuators.get(m).setActivation(self.state, muscle_activation[m])
        try:
            self.model.equilibrateMuscles(self.state)
            self.model.realizeAcceleration(self.state)
            return self.state.getUDot()
        except RuntimeError:
            if n_times > 10:
                raise RuntimeError("equilibrateMuscles failed too much times")
            return self.forward_dynamics(x+0.001, n_times+1)


class StaticOptimizationLinearPrevConstraints(
    KinematicModel,
    ObjMinimizeActivation,
    ConstraintAccelerationTarget,
    ResidualForces,
    ExternalForces,
):
    """
    Intends to mimic the classical approach but with the constraints linearized.
    It makes the assumption that muscle length is constant at a particular position and velocity, whatever the muscle
    activation.
    """

    # TODO: write tests
    def __init__(
        self,
        model_path,
        mot,
        filter_param=None,
        activation_exponent=2,
        residual_actuator_xml=None,
        external_load_xml=None,
        muscle_physiology=True,
    ):
        self.muscle_physiology = muscle_physiology
        self.previous_activation = np.array([])

        KinematicModel.__init__(self, model_path)
        ObjMinimizeActivation.__init__(self, activation_exponent)
        ConstraintAccelerationTarget.__init__(self)
        ResidualForces.__init__(self, residual_actuator_xml)
        ExternalForces.__init__(self, external_load_xml)
        self.finalize_model(mot, filter_param)

        # prepare linear constraints variables
        self.previous_activation = np.ones(self.n_muscles)
        self.qddot_from_nl = []
        self.constraint_vector = []
        self.optimal_forces = []
        self.passive_forces = []
        self.constraint_matrix = []
        self.jacobian_matrix = []  # Precomputed jacobian
        self._prepare_constraints()

    def forward_dynamics(self, x):
        fs = self.model.getForceSet()
        for i in range(fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                act.setOverrideActuation(self.state, x[i] * self.optimal_forces[i])

        self.model.realizeAcceleration(self.state)
        return self.state.getUDot()

    def passive_forward_dynamics(self):
        fs = self.model.getForceSet()
        imus = 0
        for i in range(fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                if osim.Muscle.safeDownCast(fs.get(i)):
                    act.setOverrideActuation(self.state, self.passive_forces[imus])
                    imus += 1
                else:
                    act.setOverrideActuation(self.state, 0)

        self.model.realizeAcceleration(self.state)
        qddot = self.state.getUDot()
        return [qddot.get(idx_q) for idx_q in range(qddot.size())]

    def upd_model_kinematics(self, frame):
        super().upd_model_kinematics(frame)
        if self.previous_activation.size > 0:
            self._prepare_constraints()

    def _prepare_constraints(self):
        fs = self.model.getForceSet()
        for i in range(fs.getSize()):
            m = osim.Muscle.safeDownCast(fs.get(i))
            if m:
                m.overrideActuation(self.state, False)

        self.passive_forces = []
        redo = True
        modifier = 0.001
        n_time = 0
        while redo:
            redo = False
            for m in range(self.n_muscles):
                self.muscle_actuators.get(m).setActivation(self.state, self.previous_activation[m] + modifier)
            try:
                self.model.equilibrateMuscles(self.state)
            except RuntimeError:
                redo = True
                modifier += 0.001
                n_time += 1
            if n_time > 10:
                raise RuntimeError("equilibrateMuscles failed too much times")

        self.model.realizeVelocity(self.state)
        for m in range(self.n_muscles):
            self.passive_forces.append(self.muscle_actuators.get(m).getPassiveFiberForceAlongTendon(self.state))

        forces = self.model.getForceSet()
        self.optimal_forces = []
        imus = 0
        for i in range(forces.getSize()):
            muscle = osim.Muscle.safeDownCast(forces.get(i))
            if muscle:
                if self.muscle_physiology:
                    self.model.setAllControllersEnabled(True)
                    self.optimal_forces.append(self.compute_fiber_force(muscle, i))
                    self.model.setAllControllersEnabled(False)
                else:
                    self.optimal_forces.append(muscle.getMaxIsometricForce())
                imus += 1
            coordinate = osim.CoordinateActuator.safeDownCast(forces.get(i))
            if coordinate:
                self.optimal_forces.append(coordinate.getOptimalForce())

        self.approximate_constraints()

    def compute_fiber_force(self, muscle, i):
        return muscle.getActiveFiberForceAlongTendon(self.state) / self.previous_activation[i]

    def constraints(self, x, idx=None):
        x_tp = x.reshape((x.shape[0], 1))
        jacobian = self.jacobian(x)
        x_mul = np.ndarray((jacobian.shape[0], x_tp.shape[1]))
        np.matmul(jacobian, x_tp, x_mul)
        x_constraint = x_mul.ravel()

        # That is what really happening
        # const = self.actual_qddot - (x_constraint + self.passive_forward_dynamics())
        const = x_constraint - self.constraint_vector

        if idx is not None:
            return const[idx]
        else:
            return const

    def approximate_constraints(self):
        fs = self.model.getForceSet()
        for i in range(fs.getSize()):
            act = osim.ScalarActuator.safeDownCast(fs.get(i))
            if act:
                act.overrideActuation(self.state, True)
        p_vector = np.zeros(self.n_actuators)

        qddot_from_nl_tp = self.forward_dynamics(p_vector)
        self.qddot_from_nl = np.array(
            [qddot_from_nl_tp.get(idx_q) for idx_q in range(qddot_from_nl_tp.size())])

        self.constraint_matrix = np.zeros((self.n_dof, self.n_actuators))

        for p in range(self.n_actuators):
            p_vector[p] = 1
            qddot_from_muscles_tp = self.forward_dynamics(p_vector)
            qddot_from_muscles = np.array([qddot_from_muscles_tp.get(idx_q) for idx_q in range(qddot_from_muscles_tp.size())])
            self.constraint_matrix[:, p] = qddot_from_muscles - self.qddot_from_nl
            p_vector[p] = 0

        self.constraint_vector = np.array(self.actual_qddot) - np.array(self.passive_forward_dynamics())

    def jacobian(self, x):
        return self.constraint_matrix

    def set_previous_activation(self, x):
        x[x <= 0] = 0.01
        x[x > 1] = 1
        self.previous_activation = np.array(x)


class StaticOptimizationLinearMaxImplementation(
    StaticOptimizationLinearPrevConstraints
):
    def __init__(
            self,
            model_path,
            mot,
            filter_param=None,
            activation_exponent=2,
            residual_actuator_xml=None,
            external_load_xml=None,
            muscle_physiology=True,):
        super().__init__(model_path, mot, filter_param, activation_exponent, residual_actuator_xml, external_load_xml, muscle_physiology)

    def compute_fiber_force(self, muscle, i):
        return muscle.calcInextensibleTendonActiveFiberForce(self.state, 1.0)


class StaticOptimizationCubicSplineConstraints(StaticOptimization):
    def __init__(
            self,
            model_path,
            mot,
            filter_param=None,
            activation_exponent=2,
            residual_actuator_xml=None,
            external_load_xml=None):
        super().__init__(model_path, mot, filter_param, activation_exponent, residual_actuator_xml, external_load_xml)
        self.qddot_interp = np.zeros((self.n_dof, self.n_actuators)).tolist()

    def upd_model_kinematics(self, frame):
        super().upd_model_kinematics(frame)
        self.approximate_constraints()

    def approximate_constraints(self):
        fs = self.model.getForceSet()
        for i in range(fs.getSize()):
            act = osim.CoordinateActuator.safeDownCast(fs.get(i))
            if act:
                act.overrideActuation(self.state, True)

        p_vector = np.zeros(self.n_actuators)
        qddot_from_zero_tp = self.forward_dynamics(p_vector)
        qddot_from_zero = np.array([qddot_from_zero_tp.get(idx_q) for idx_q in range(qddot_from_zero_tp.size())])

        x_pts = np.linspace(0, 1, 4)
        self.qddot_interp = np.zeros((self.n_dof, self.n_actuators)).tolist()
        qddot_from_muscles = np.ndarray((self.n_dof, len(x_pts)))
        qddot_from_muscles[:, 0] = qddot_from_zero
        for p in range(self.n_actuators):
            # At each percentage of evaluation
            for i, eval_pt in enumerate(x_pts[1:]):  # Drop the already computed zero
                p_vector[p] = eval_pt
                # Call the real forward dynamics
                qddot_from_muscles_tp = self.forward_dynamics(p_vector)
                qddot_from_muscles[:, i+1] = \
                    np.array([qddot_from_muscles_tp.get(idx_q) for idx_q in range(qddot_from_muscles_tp.size())])

            # Compute cubic spline
            for i in range(self.n_dof):
                self.qddot_interp[i][p] = interpolate.splrep(x_pts, -qddot_from_muscles[i, :])
            p_vector[p] = 0

    def jacobian(self, x):
        jacobian = np.ndarray((self.n_dof, self.n_actuators))
        for p in range(self.n_actuators):
            for i in range(self.n_dof):
                jacobian[i, p] = interpolate.splev(x[p], self.qddot_interp[i][p], der=1)
        return jacobian
