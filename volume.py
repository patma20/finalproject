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
        self.add_input("L", units="m")
        self.add_input("R", units="m")
        self.add_input("t", units="m")
        self.add_input("rho_o", units="kg/m ** 3")
        self.add_input("rho_f", units="kg/m ** 3")
        self.add_input("OF")

        # --- Outputs ---
        self.add_output("v_s", units="m**3")
        self.add_output("v_p", units="m**3")
        self.add_output("v", units="m**3")
        self.add_output("v_f", units="m**3")
        self.add_output("v_o", units="m**3")

    def compute(self, inputs, outputs):
        L = inputs["L"]
        R = inputs["R"]
        t = inputs["t"]
        rho_o = inputs["rho_o"]
        rho_f = inputs["rho_f"]
        OF = inputs["OF"]

        v = np.pi * R ** 2 * L + 4 / 3 * np.pi * R ** 3
        v_p = np.pi * (R - t) ** 2 * L + 4 / 3 * np.pi * (R - t) ** 3
        v_s = v - v_p
        v_f = v_p * rho_o / (OF * rho_f + rho_o)
        v_o = v_p - v_f

        outputs["v_s"] = v_s
        outputs["v_p"] = v_p
        outputs["v"] = v
        outputs["v_f"] = v_f
        outputs["v_o"] = v_o


class Con1(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("p", units="Pa")
        self.add_input("t", units="m")
        self.add_input("R", units="m")
        self.add_input("s_t", units="Pa")

        # --- Outputs ---
        self.add_output("con1", units="Pa")

    def compute(self, inputs, outputs):
        p = inputs["p"]
        t = inputs["t"]
        R = inputs["R"]
        s_t = inputs["s_t"]

        outputs["con1"] = p * R / t - s_t


class Con2(om.ExplicitComponent):
    def setup(self):
        # --- Inputs ---
        self.add_input("p", units="Pa")
        self.add_input("t", units="m")
        self.add_input("R", units="m")
        self.add_input("s_y", units="Pa")
        self.add_input("g", units="m/s**2")
        self.add_input("m_L", units="kg")

        # --- Outputs ---
        self.add_output("con2", units="Pa")

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
        self.add_input("L", units="m")
        self.add_input("R", units="m")

        # --- Outputs ---
        self.add_output("con3")

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

        self.add_subsystem("iR", om.IndepVarComp("R", 0.5, units="m"))
        self.add_subsystem("iro", om.IndepVarComp("rho_o", 1000, units="kg/m ** 3"))
        self.add_subsystem("irf", om.IndepVarComp("rho_f", 1021, units="kg/m ** 3"))
        self.add_subsystem("iOF", om.IndepVarComp("OF", 2.56))
        self.add_subsystem("ip", om.IndepVarComp("p", 0.36e6, units="Pa"))
        self.add_subsystem("isy", om.IndepVarComp("s_y", 200e6, units="Pa"))
        self.add_subsystem("ig", om.IndepVarComp("g", 9.81, units="m/s**2"))
        self.add_subsystem("iml", om.IndepVarComp("m_L", 100, units="kg"))
        self.add_subsystem("ist", om.IndepVarComp("s_t", 515e6, units="Pa"))

    def configure(self):
        self.promotes("obj_cmp", any=["t", "L", "v_s", "v_p", "v", "v_f", "v_o"])
        self.promotes("con1_cmp", any=["t", "con1"])
        # self.promotes("con2_cmp", any=["t", "con2"])
        self.promotes("con3_cmp", any=["L", "con3"])


if __name__ == "__main__":
    prob = om.Problem()
    prob.model = OneStage()

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    prob.model.add_design_var("L", lower=0.0)
    prob.model.add_design_var("t", lower=1e-6, upper=0.49)
    prob.model.add_objective("v_s")
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
    # print("Total", prob.get_val("v")[0])
    print("Structure", prob.get_val("v_s")[0])
    # print("Propellent", prob.get_val("v_p")[0])
    # print("Fuel", prob.get_val("v_f")[0])
    # print("Oxidizer", prob.get_val("v_o")[0])
