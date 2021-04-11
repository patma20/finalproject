# --- Python 3.8 ---
"""
@File : one_stage.py
@Time : 2021/04/07
@Author : Peter Atma
@Desc : None
"""

# --- Standard Python modules ---
# --- External Python modules ---
import numpy as np
import openmdao.api as om

# --- Extension modules ---


class StageMass(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("L", val=5.0, units="m")
        self.add_input("R", val=0.5, units="m")
        self.add_input("t", val=1.0, units="m")
        self.add_input("rho_s", val=8000, units="kg/m ** 3")
        self.add_input("rho_o", val=1000, units="kg/m ** 3")
        self.add_input("rho_f", val=1021, units="kg/m ** 3")
        self.add_input("OF", val=2.56)
        self.add_input("m_L", val=100, units="kg")

        # --- Outputs ---
        # self.add_output("mR", val=100, units="kg")
        self.add_output("m01", val=1000, units="kg")

    def compute(self, inputs, outputs):
        L = inputs["L"]
        R = inputs["R"]
        t = inputs["t"]
        rho_s = inputs["rho_s"]
        rho_o = inputs["rho_o"]
        rho_f = inputs["rho_f"]
        OF = inputs["OF"]
        m_L = inputs["m_L"]

        v = np.pi * R ** 2 * L + 4 / 3 * np.pi * R ** 3
        v_p = np.pi * (R - t) ** 2 * L + 4 / 3 * np.pi * (R - t) ** 3
        v_s = v - v_p
        v_f = v_p * rho_o / (OF * rho_f + rho_o)
        v_o = v_p - v_f

        m_s = v_s * rho_s
        m_p = v_o * rho_o + v_f * rho_f

        outputs["m01"] = m_s + m_p + m_L
        # outputs["mR"] = (m_s + m_p + m_L) / m_L


class Con1(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("p", val=0.36e6, units="Pa")
        self.add_input("t", val=1.0, units="m")
        self.add_input("R", val=0.5, units="m")
        self.add_input("s_t", val=515e6, units="Pa")

        # --- Outputs ---
        self.add_output("con1", val=0.0, units="Pa")

    def compute(self, inputs, outputs):
        p = inputs["p"]
        t = inputs["t"]
        R = inputs["R"]
        s_t = inputs["s_t"]

        outputs["con1"] = p * R / t - s_t


class Con2(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("p", val=0.36e6, units="Pa")
        self.add_input("t", val=1.0, units="m")
        self.add_input("R", val=0.5, units="m")
        self.add_input("s_y", val=332e6, units="Pa")
        self.add_input("g", val=9.81, units="m/s**2")
        self.add_input("m_L", val=100, units="kg")

        # --- Outputs ---
        self.add_output("con2", val=0.0, units="Pa")

    def compute(self, inputs, outputs):
        p = inputs["p"]
        t = inputs["t"]
        R = inputs["R"]
        s_y = inputs["s_y"]
        g = inputs["g"]
        m_L = inputs["m_L"]

        outputs["con2"] = (g * m_L) / (np.pi * (2 * R * t - t ** 2)) - p * R / (2 * t) - s_y


class Con3(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("L", val=5.0, units="m")
        self.add_input("R", val=0.5, units="m")

        # --- Outputs ---
        self.add_output("con3", val=0.0)

    def compute(self, inputs, outputs):
        L = inputs["L"]
        R = inputs["R"]

        outputs["con3"] = 1 - L / R


class OneStage(om.Group):
    def setup(self):
        self.add_subsystem("obj_cmp", StageMass())
        self.add_subsystem("con1_cmp", Con1())
        # self.add_subsystem("con2_cmp", Con2())
        self.add_subsystem("con3_cmp", Con3())

        # self.add_subsystem("obj_cmp", StageMass(), promotes=["t", "L"])
        # self.add_subsystem("con1_cmp", Con1(), promotes_inputs=["t"])
        # self.add_subsystem("con2_cmp", Con2(), promotes_inputs=["t", "m01"])
        # self.add_subsystem("con3_cmp", Con3(), promotes_inputs=["L"])

    def configure(self):
        self.promotes("obj_cmp", any=["t", "L", "m01"])
        self.promotes("con1_cmp", any=["t", "con1"])
        # self.promotes("con2_cmp", any=["t", "con2"])
        self.promotes("con3_cmp", any=["L", "con3"])
        # self.connect("obj_cmp.m01", "con2_cmp.m01")
        # self.connect("obj_cmp.L", "con3_cmp.L")
        # self.connect("obj_cmp.t", ["con1_cmp.t", "con2_cmp.t"])


if __name__ == "__main__":
    prob = om.Problem()
    prob.model = OneStage()

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    prob.model.add_design_var("L", lower=0.0)
    prob.model.add_design_var("t", lower=0.0, upper=0.4)
    prob.model.add_objective("m01")
    prob.model.add_constraint("con1", upper=0.0)
    # prob.model.add_constraint("con2", upper=0.0)
    prob.model.add_constraint("con3", upper=0.0)

    prob.model.approx_totals(method="fd")

    prob.setup()
    prob.set_solver_print(level=0)

    prob.run_driver()

    print("minimum found at")
    print(prob.get_val("L")[0])
    print(prob.get_val("t")[0])

    print("minumum objective")
    print(prob.get_val("m01")[0])
