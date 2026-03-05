"""
Thevenin Equivalent Circuit Models (1st and 2nd order) as Baselines
====================================================================
This module implements standard ECM baselines for comparison with our
Shepherd-based model.

1st-order Thevenin:
    V = OCV(SOC) - R0*I - V_RC1
    dV_RC1/dt = I/C1 - V_RC1/(R1*C1)

2nd-order Thevenin:
    V = OCV(SOC) - R0*I - V_RC1 - V_RC2
    dV_RC1/dt = I/C1 - V_RC1/(R1*C1)
    dV_RC2/dt = I/C2 - V_RC2/(R2*C2)

OCV-SOC: 6th-order polynomial fit from XJTU discharge data.

References:
    [1] Hu et al., J. Power Sources, 198, 2012, pp.359-367
    [2] Plett, J. Power Sources, 134, 2004, pp.252-261
    [3] He et al., Energies, 4, 2011, pp.582-598
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize
from scipy.io import loadmat
import os


# ==================== OCV-SOC Polynomial ====================
# Default: 6th-order polynomial coefficients fitted from XJTU Batch-1
# Will be overridden by fit_ocv_soc() if data is available
DEFAULT_OCV_COEFFS = [
    -25.132, 87.047, -120.810, 87.117, -34.869, 8.274, 3.237
]  # p6..p0 for np.polyval


def ocv_polynomial(soc, coeffs=None):
    """OCV(SOC) via polynomial. soc in [0,1], returns voltage."""
    if coeffs is None:
        coeffs = DEFAULT_OCV_COEFFS
    soc = np.clip(soc, 0.005, 0.995)
    return np.polyval(coeffs, soc)


def fit_ocv_soc(soc_data, voltage_data, degree=6):
    """Fit OCV-SOC polynomial from low-current discharge data."""
    mask = (soc_data > 0.005) & (soc_data < 0.995)
    coeffs = np.polyfit(soc_data[mask], voltage_data[mask], degree)
    return coeffs


# ==================== 1st-Order Thevenin ====================
class TheveninFirstOrder:
    """
    1st-order Thevenin ECM.
    States: [V_RC1, SOC]
    V_terminal = OCV(SOC) - R0*I - V_RC1
    """

    def __init__(self, Q0=2.0, R0=0.05, R1=0.02, C1=1000.0,
                 V_cutoff=2.5, ocv_coeffs=None):
        self.Q0 = Q0
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.V_cutoff = V_cutoff
        self.ocv_coeffs = ocv_coeffs
        self.tau1 = R1 * C1

    def terminal_voltage(self, soc, current, v_rc1):
        ocv = ocv_polynomial(soc, self.ocv_coeffs)
        return ocv - self.R0 * current - v_rc1

    def dynamics(self, t, state, I_func):
        v_rc1, soc = state
        I = I_func(t) if callable(I_func) else I_func
        soc = np.clip(soc, 0.001, 1.0)

        dv_rc1 = I / self.C1 - v_rc1 / (self.R1 * self.C1)
        dsoc = -I / (self.Q0 * 3600)

        return [dv_rc1, dsoc]

    def simulate_cc_discharge(self, I_load, soc_init=1.0, dt_eval=1.0):
        """Simulate constant-current discharge."""
        t_max = self.Q0 / I_load * 3600 * 1.3

        def stop_voltage(t, state):
            v_rc1, soc = state
            soc = np.clip(soc, 0.001, 1.0)
            V = self.terminal_voltage(soc, I_load, v_rc1)
            return V - self.V_cutoff

        def stop_soc(t, state):
            return state[1] - 0.05

        stop_voltage.terminal = True
        stop_voltage.direction = -1
        stop_soc.terminal = True
        stop_soc.direction = -1

        sol = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y, I_load),
            t_span=[0, t_max],
            y0=[0.0, soc_init],
            method='RK45',
            events=[stop_voltage, stop_soc],
            max_step=dt_eval,
            rtol=1e-6, atol=1e-8,
        )

        soc = np.clip(sol.y[1], 0.001, 1.0)
        V = np.array([
            self.terminal_voltage(s, I_load, vrc)
            for s, vrc in zip(soc, sol.y[0])
        ])

        return {
            'time_s': sol.t,
            'time_h': sol.t / 3600,
            'soc': soc,
            'voltage': V,
            'current': np.full_like(sol.t, I_load),
            'discharge_time_h': sol.t[-1] / 3600,
        }

    def simulate_cp_discharge(self, P_load, soc_init=1.0, dt=1.0):
        """Simulate constant-power discharge (for TTE comparison)."""
        soc = soc_init
        v_rc1 = 0.0
        I_avg = P_load / 3.7
        t = 0
        t_max = 30 * 3600

        times, socs, voltages = [0], [soc], []
        V = self.terminal_voltage(soc, I_avg, v_rc1)
        voltages.append(V)

        while soc > 0.05 and t < t_max:
            V = self.terminal_voltage(soc, I_avg, v_rc1)
            if V < self.V_cutoff:
                break
            I_new = P_load / V if V > 0 else I_avg
            I_avg = 0.9 * I_avg + 0.1 * I_new

            dv_rc1 = I_avg / self.C1 - v_rc1 / (self.R1 * self.C1)
            v_rc1 += dv_rc1 * dt
            soc -= I_avg * dt / (self.Q0 * 3600)
            t += dt

            times.append(t / 3600)
            socs.append(soc)
            voltages.append(V)

        return {
            'time_h': np.array(times),
            'soc': np.array(socs),
            'voltage': np.array(voltages),
            'discharge_time_h': times[-1],
        }


# ==================== 2nd-Order Thevenin ====================
class TheveninSecondOrder:
    """
    2nd-order Thevenin ECM.
    States: [V_RC1, V_RC2, SOC]
    V_terminal = OCV(SOC) - R0*I - V_RC1 - V_RC2
    """

    def __init__(self, Q0=2.0, R0=0.05, R1=0.015, C1=1000.0,
                 R2=0.025, C2=5000.0, V_cutoff=2.5, ocv_coeffs=None):
        self.Q0 = Q0
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.R2 = R2
        self.C2 = C2
        self.V_cutoff = V_cutoff
        self.ocv_coeffs = ocv_coeffs

    def terminal_voltage(self, soc, current, v_rc1, v_rc2):
        ocv = ocv_polynomial(soc, self.ocv_coeffs)
        return ocv - self.R0 * current - v_rc1 - v_rc2

    def dynamics(self, t, state, I_func):
        v_rc1, v_rc2, soc = state
        I = I_func(t) if callable(I_func) else I_func
        soc = np.clip(soc, 0.001, 1.0)

        dv_rc1 = I / self.C1 - v_rc1 / (self.R1 * self.C1)
        dv_rc2 = I / self.C2 - v_rc2 / (self.R2 * self.C2)
        dsoc = -I / (self.Q0 * 3600)

        return [dv_rc1, dv_rc2, dsoc]

    def simulate_cc_discharge(self, I_load, soc_init=1.0, dt_eval=1.0):
        """Simulate constant-current discharge."""
        t_max = self.Q0 / I_load * 3600 * 1.3

        def stop_voltage(t, state):
            v_rc1, v_rc2, soc = state
            soc = np.clip(soc, 0.001, 1.0)
            V = self.terminal_voltage(soc, I_load, v_rc1, v_rc2)
            return V - self.V_cutoff

        def stop_soc(t, state):
            return state[2] - 0.05

        stop_voltage.terminal = True
        stop_voltage.direction = -1
        stop_soc.terminal = True
        stop_soc.direction = -1

        sol = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y, I_load),
            t_span=[0, t_max],
            y0=[0.0, 0.0, soc_init],
            method='RK45',
            events=[stop_voltage, stop_soc],
            max_step=dt_eval,
            rtol=1e-6, atol=1e-8,
        )

        soc = np.clip(sol.y[2], 0.001, 1.0)
        V = np.array([
            self.terminal_voltage(s, I_load, vrc1, vrc2)
            for s, vrc1, vrc2 in zip(soc, sol.y[0], sol.y[1])
        ])

        return {
            'time_s': sol.t,
            'time_h': sol.t / 3600,
            'soc': soc,
            'voltage': V,
            'current': np.full_like(sol.t, I_load),
            'discharge_time_h': sol.t[-1] / 3600,
        }

    def simulate_cp_discharge(self, P_load, soc_init=1.0, dt=1.0):
        """Simulate constant-power discharge (for TTE comparison)."""
        soc = soc_init
        v_rc1, v_rc2 = 0.0, 0.0
        I_avg = P_load / 3.7
        t = 0
        t_max = 30 * 3600

        times, socs, voltages = [0], [soc], []
        V = self.terminal_voltage(soc, I_avg, v_rc1, v_rc2)
        voltages.append(V)

        while soc > 0.05 and t < t_max:
            V = self.terminal_voltage(soc, I_avg, v_rc1, v_rc2)
            if V < self.V_cutoff:
                break
            I_new = P_load / V if V > 0 else I_avg
            I_avg = 0.9 * I_avg + 0.1 * I_new

            dv_rc1 = I_avg / self.C1 - v_rc1 / (self.R1 * self.C1)
            dv_rc2 = I_avg / self.C2 - v_rc2 / (self.R2 * self.C2)
            v_rc1 += dv_rc1 * dt
            v_rc2 += dv_rc2 * dt
            soc -= I_avg * dt / (self.Q0 * 3600)
            t += dt

            times.append(t / 3600)
            socs.append(soc)
            voltages.append(V)

        return {
            'time_h': np.array(times),
            'soc': np.array(socs),
            'voltage': np.array(voltages),
            'discharge_time_h': times[-1],
        }


# ==================== Parameter Identification ====================
def identify_thevenin_params(time_s, voltage, current, Q0, order=1):
    """
    Identify Thevenin ECM parameters from discharge data.

    Uses least-squares optimization to fit R0, R1, C1 (and R2, C2 for 2nd order)
    by minimizing voltage prediction error.
    """
    # Compute SOC from current integration
    dt = np.diff(time_s, prepend=time_s[0])
    capacity = np.cumsum(np.abs(current) * dt) / 3600
    Q_total = capacity[-1]
    soc = 1 - capacity / Q_total

    # Fit OCV polynomial from data (approximate: use voltage at very low current)
    ocv_coeffs = fit_ocv_soc(soc, voltage, degree=6)

    if order == 1:
        def cost(params):
            R0, R1, C1 = params
            if R0 < 0 or R1 < 0 or C1 < 10:
                return 1e10
            model = TheveninFirstOrder(Q0=Q0, R0=R0, R1=R1, C1=C1,
                                       ocv_coeffs=ocv_coeffs)
            # Simulate with actual current profile
            v_rc1 = 0.0
            errors = []
            for i in range(len(time_s)):
                I = np.abs(current[i])
                V_pred = model.terminal_voltage(soc[i], I, v_rc1)
                errors.append((V_pred - voltage[i]) ** 2)
                if i < len(time_s) - 1:
                    dt_i = time_s[i + 1] - time_s[i]
                    dv = I / C1 - v_rc1 / (R1 * C1)
                    v_rc1 += dv * dt_i
            return np.mean(errors)

        from scipy.optimize import differential_evolution
        bounds = [(0.01, 0.3), (0.005, 0.2), (100, 50000)]
        result = differential_evolution(cost, bounds, maxiter=200, seed=42, tol=1e-6)
        R0, R1, C1 = result.x
        return {
            'R0': R0, 'R1': R1, 'C1': C1,
            'ocv_coeffs': ocv_coeffs, 'Q0': Q0,
            'rmse_fit': np.sqrt(result.fun),
        }

    elif order == 2:
        def cost(params):
            R0, R1, C1, R2, C2 = params
            if R0 < 0 or R1 < 0 or R2 < 0 or C1 < 10 or C2 < 10:
                return 1e10
            v_rc1, v_rc2 = 0.0, 0.0
            errors = []
            for i in range(len(time_s)):
                I = np.abs(current[i])
                ocv = ocv_polynomial(soc[i], ocv_coeffs)
                V_pred = ocv - R0 * I - v_rc1 - v_rc2
                errors.append((V_pred - voltage[i]) ** 2)
                if i < len(time_s) - 1:
                    dt_i = time_s[i + 1] - time_s[i]
                    v_rc1 += (I / C1 - v_rc1 / (R1 * C1)) * dt_i
                    v_rc2 += (I / C2 - v_rc2 / (R2 * C2)) * dt_i
            return np.mean(errors)

        from scipy.optimize import differential_evolution
        bounds = [(0.01, 0.3), (0.005, 0.15), (100, 20000),
                  (0.005, 0.15), (1000, 100000)]
        result = differential_evolution(cost, bounds, maxiter=300, seed=42, tol=1e-6)
        R0, R1, C1, R2, C2 = result.x
        return {
            'R0': R0, 'R1': R1, 'C1': C1, 'R2': R2, 'C2': C2,
            'ocv_coeffs': ocv_coeffs, 'Q0': Q0,
            'rmse_fit': np.sqrt(result.fun),
        }


# ==================== Rint (simplest baseline) ====================
class RintModel:
    """
    Simplest baseline: Rint model (internal resistance only).
    V = OCV(SOC) - R_int * I
    """

    def __init__(self, Q0=2.0, R_int=0.05, V_cutoff=2.5, ocv_coeffs=None):
        self.Q0 = Q0
        self.R_int = R_int
        self.V_cutoff = V_cutoff
        self.ocv_coeffs = ocv_coeffs

    def terminal_voltage(self, soc, current):
        ocv = ocv_polynomial(soc, self.ocv_coeffs)
        return ocv - self.R_int * current

    def simulate_cc_discharge(self, I_load, soc_init=1.0, dt=1.0):
        soc = soc_init
        t = 0
        t_max = self.Q0 / I_load * 3600 * 1.3
        times, socs, voltages = [0], [soc], []
        V = self.terminal_voltage(soc, I_load)
        voltages.append(V)

        while soc > 0.05 and t < t_max:
            V = self.terminal_voltage(soc, I_load)
            if V < self.V_cutoff:
                break
            soc -= I_load * dt / (self.Q0 * 3600)
            t += dt
            times.append(t / 3600)
            socs.append(soc)
            voltages.append(V)

        return {
            'time_h': np.array(times),
            'soc': np.array(socs),
            'voltage': np.array(voltages),
            'discharge_time_h': times[-1],
        }


if __name__ == '__main__':
    print("Thevenin ECM Baselines loaded successfully.")
    print("Models: RintModel, TheveninFirstOrder, TheveninSecondOrder")
    print("Use identify_thevenin_params() to fit from data.")
