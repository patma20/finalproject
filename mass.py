# --- Python 3.8 ---
"""
@File : mass.py
@Time : 2021/04/07
@Author : Peter Atma
@Desc : None
"""

# --- Standard Python modules ---
# --- External Python modules ---
import numpy as np
import openmdao.api as om
import constants as c

# --- Extension modules ---


class StageMass(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("L", units="m")
        self.add_input("R", val=c.R, units="m")
        self.add_input("t", units="m")
        self.add_input("rho_o", val=c.rho_02, units="kg/m ** 3")
        self.add_input("rho_f", val=c.rho_rp1, units="kg/m ** 3")
        self.add_input("rho_s", val=c.rho_ss, units="kg/m ** 3")
        self.add_input("OF", val=c.OF_rp1_o)
        self.add_input("m_L", val=c.m_L, units="kg")

        # --- Outputs ---
        self.add_output("m_s", units="m**3", lower=1e-6, ref=1e1)
        # self.add_output("v_p", units="m**3")
        # self.add_output("v", units="m**3")
        # self.add_output("v_f", units="m**3")
        # self.add_output("v_o", units="m**3")

    def setup_partials(self):
        # --- Derivatives ---
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        L = inputs["L"]
        R = inputs["R"]
        t = inputs["t"]
        # rho_o = inputs["rho_o"]
        # rho_f = inputs["rho_f"]
        rho_s = inputs["rho_s"]
        # OF = inputs["OF"]

        v = np.pi * (R ** 2 * L + 4 / 3 * R ** 3)
        v_p = np.pi * (L * (R - t) ** 2 + 4 / 3 * (R - t) ** 3)
        v_s = v - v_p
        # v_f = v_p * rho_o / (OF * rho_f + rho_o)
        # v_o = v_p - v_f

        m_s = v_s * rho_s

        outputs["m_s"] = m_s  # to minimize structural mass


class Con1(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("p", val=c.p, units="Pa")
        self.add_input("t", units="m")
        self.add_input("R", val=c.R, units="m")
        self.add_input("s_t", val=c.st_ss, units="Pa")

        # --- Outputs ---
        self.add_output("con1", units="Pa")

    def setup_partials(self):
        # --- Derivatives ---
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        p = inputs["p"]
        t = inputs["t"]
        R = inputs["R"]
        s_t = inputs["s_t"]

        outputs["con1"] = s_t / (p * R) - t  # to ensure t is large enough to withstand internal pressures p * R / t -


class Con2(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("p", val=c.p, units="Pa")
        self.add_input("t", units="m")
        self.add_input("R", val=c.R, units="m")
        self.add_input("s_y", val=c.sy_ss, units="Pa")
        self.add_input("g", val=c.g0, units="m/s**2")
        self.add_input("m_L", val=c.m_L, units="kg")

        # --- Outputs ---
        self.add_output("con2", units="Pa")

    def setup_partials(self):
        # --- Derivatives ---
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        p = inputs["p"]
        t = inputs["t"]
        R = inputs["R"]
        s_y = inputs["s_y"]
        g = inputs["g"]
        mL = inputs["m_L"]

        outputs["con2"] = (g * mL) / (np.pi * (2 * R * t - t ** 2)) - p * R / (2 * t) - s_y
        # to ensure t is large enough to withstand stresses


class Con3(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("L", units="m")
        self.add_input("R", val=c.R, units="m")

        # --- Outputs ---
        self.add_output("con3")

    def setup_partials(self):
        # --- Derivatives ---
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        L = inputs["L"]
        R = inputs["R"]

        outputs["con3"] = 1 - L / R  # to ensure L is greater than zero until final constraints are added


class OneStage(om.Group):  # "v_p", "v", "v_f", "v_o"
    def setup(self):
        self.add_subsystem("obj_cmp", StageMass(), promotes_inputs=["t", "L"], promotes_outputs=["m_s"])
        self.add_subsystem("con1_cmp", Con1(), promotes_inputs=["t"], promotes_outputs=["con1"])
        # self.add_subsystem("con2_cmp", Con2(), promotes_inputs=["t"], promotes_outputs=["con2"])
        # self.add_subsystem("con3_cmp", Con3(), promotes_inputs=["L"], promotes_outputs=["con3"])

        self.set_input_defaults("t", 1e-3, units="m")
        self.set_input_defaults("L", 0.5, units="m")

        self.linear_solver = om.DirectSolver(assemble_jac=True)


if __name__ == "__main__":
    prob = om.Problem()
    prob.model = OneStage()

    prob.driver = om.pyOptSparseDriver()
    # prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    prob.model.add_design_var("L", lower=0.5)
    prob.model.add_design_var("t", lower=1e-6, upper=0.49)
    prob.model.add_objective("m_s")
    prob.model.add_constraint("con1", upper=0.0)
    # prob.model.add_constraint("con2", upper=0.0)
    # prob.model.add_constraint("con3", upper=0.0)

    prob.model.approx_totals(method="cs")

    prob.setup(check=True)
    prob.set_solver_print(level=0)

    prob.run_driver()

    prob.model.list_inputs()
    prob.model.list_outputs()

    print("minimum found at")
    print(prob.get_val("L")[0])
    print(prob.get_val("t")[0])

    print("minumum objective")
    # print("Total", prob.get_val("v")[0])
    print("Structure", prob.get_val("m_s")[0])
    # print("Propellent", prob.get_val("v_p")[0])
    # print("Fuel", prob.get_val("v_f")[0])
    # print("Oxidizer", prob.get_val("v_o")[0])
