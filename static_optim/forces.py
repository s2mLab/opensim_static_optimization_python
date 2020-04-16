"""
Forces method classes.

The dynamic model should inherit from these classes in order to get the proper forces.
"""
import opensim as osim


class ResidualForces:
    def __init__(self, residual_actuator_xml=None):
        if residual_actuator_xml:
            res_force_set = osim.ForceSet(residual_actuator_xml, True)
            n_forces = res_force_set.getSize()
            model_force_set = self.model.getForceSet()
            for i in range(n_forces):
                force = res_force_set.get(i)
                model_force_set.cloneAndAppend(force)


class ExternalForces:
    def __init__(self, external_load_xml=None):
        if external_load_xml:
            force_set = osim.ForceSet(external_load_xml)
            n_forces = force_set.getSize()
            for i in range(n_forces):
                self.model.getForceSet().append(force_set.get(i))

