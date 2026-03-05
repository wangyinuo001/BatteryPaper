"""
Main Model: Shepherd + Arrhenius Temperature Correction
========================================================
Based on hand-written formulas (NO aging effects):

Voltage (Shepherd):
    V = E₀ - R(T)·I - k·(I/SOC) + A·exp(-B·(1-SOC))

Temperature Corrections:
    R(T) = R₀·exp(-Er/(kB·T))           [Arrhenius]
    Q₀(T) = Q₀₀/(1+exp(a+b/T))          [Logistic]

Current-Power:
    I = -Q₀(T)·d(SOC)/dt
    P = V·I

All parameters fitted from XJTU Batch-1 Battery-1 Cycle 0.
"""

import numpy as np

# ==================== Fitted Shepherd Parameters ====================
# From fit_shepherd_params.py using XJTU data
SHEPHERD_E0 = 3.3843  # Open-circuit voltage (V)
SHEPHERD_K = 0.0175  # Polarization constant
SHEPHERD_A = 0.8096  # Exponential zone amplitude (V)
SHEPHERD_B = 1.0062  # Exponential zone rate
SHEPHERD_R0 = 0.0350  # Internal resistance (Ω)

# ==================== Temperature Parameters ====================
TEMP_REF = 298.15  # Reference temperature (25°C = 298.15K)
TEMP_Q_A = -10.05  # Logistic Q₀(T) parameter a
TEMP_Q_B = 3000.0  # Logistic Q₀(T) parameter b (K)
TEMP_R_ER = 0.005  # Arrhenius activation energy (eV)
TEMP_KB = 8.617e-5  # Boltzmann constant (eV/K)


class MainBatteryModel:
    """
    Main Model: Shepherd voltage + Arrhenius/Logistic temperature corrections

    This is the PHYSICS-BASED model for fresh batteries.
    NO aging effects (cycle count independent).
    """

    def __init__(
        self,
        Q0=5.0,
        V_cutoff=2.5,
        E0=SHEPHERD_E0,
        K=SHEPHERD_K,
        A=SHEPHERD_A,
        B=SHEPHERD_B,
        R0=SHEPHERD_R0,
    ):
        """
        Parameters:
        -----------
        Q0 : float
            Nominal capacity at 25°C (Ah) - Samsung Galaxy S25 Ultra: 5.0Ah (5000mAh)
        V_cutoff : float
            Discharge cutoff voltage (V)
        """
        self.Q0 = Q0
        self.V_cutoff = V_cutoff
        self.E0 = E0
        self.K = K
        self.A = A
        self.B = B
        self.R0 = R0

    def get_capacity_at_temp(self, temp_k):
        """
        Temperature-dependent capacity: Q₀(T) = Q₀₀/(1+exp(a+b/T))
        Normalized so Q₀(T_ref) = Q₀₀
        """
        logistic_ref = 1.0 / (1.0 + np.exp(TEMP_Q_A + TEMP_Q_B / TEMP_REF))
        logistic_t = 1.0 / (1.0 + np.exp(TEMP_Q_A + TEMP_Q_B / temp_k))
        return self.Q0 * (logistic_t / logistic_ref)

    def get_resistance_at_temp(self, temp_k):
        """
        Temperature-dependent resistance: R(T) = R₀·exp(-Er/(kB·T))
        Normalized so R(T_ref) = R₀
        """
        arrhenius_ref = np.exp(-TEMP_R_ER / (TEMP_KB * TEMP_REF))
        arrhenius_t = np.exp(-TEMP_R_ER / (TEMP_KB * temp_k))
        return self.R0 * (arrhenius_t / arrhenius_ref)

    def terminal_voltage(self, soc, current, temp_k):
        """
        Shepherd voltage model:
        V = E₀ - R(T)·I - k·(I/SOC) + A·exp(-B·Q₀·(1-SOC))

        Parameters:
        -----------
        soc : float
            State of charge [0, 1]
        current : float
            Discharge current (A)
        temp_k : float
            Temperature (K)
        """
        soc = np.clip(soc, 0.01, 1.0)
        R_eff = self.get_resistance_at_temp(temp_k)
        Q_eff = self.get_capacity_at_temp(temp_k)

        V = (
            self.E0
            - R_eff * current
            - self.K * (current / soc)
            + self.A * np.exp(-self.B * Q_eff * (1 - soc))
        )
        return V

    def predict_discharge(self, P_load, temp_k=298.15, dt=1.0, soc_initial=1.0):
        """
        Predict discharge for constant power load at given temperature

        *** TIME TO EMPTY (TTE) CALCULATION ***
        TTE is computed by numerically integrating the coupled ODE system:

            dSOC/dt = -I(t) / Q₀(T)                    [1]
            V(t) = f_Shepherd(SOC(t), I(t), T)         [2]
            P(t) = V(t) · I(t)  =>  I(t) = P(t)/V(t)   [3]

        Integration continues until:
            - SOC reaches cutoff threshold (typically 5%)
            - Terminal voltage V(t) < V_cutoff
            - Maximum time limit reached

        The discharge time (TTE) is:
            t_empty = ∫₀^{t_cutoff} dt

        where t_cutoff is determined by the stopping conditions above.

        Method: Euler forward integration with adaptive current smoothing

        Parameters:
        -----------
        P_load : float
            Constant power load (W)
        temp_k : float
            Temperature (K), default 298.15 = 25°C
        dt : float
            Time step (seconds)

        Returns:
        --------
        results : dict
            'discharge_time', 'time', 'soc', 'voltage', 'current', 'capacity'
        """
        Q_eff = self.get_capacity_at_temp(temp_k)
        soc0 = float(np.clip(soc_initial, 0.01, 1.0))
        t_max = 30.0  # Maximum 30 hours

        time_points = [0]
        soc_points = [soc0]
        voltage_points = [self.terminal_voltage(soc0, 0, temp_k)]
        current_points = [0]

        t = 0
        soc = soc0
        I_avg = P_load / 3.7  # Initial current guess

        # ===== Numerical Integration Loop =====
        # Integrate ODE system: dSOC/dt = -I(t)/Q₀(T)
        # where I(t) = P_load / V(SOC, I, T)
        while soc > 0.05 and t < t_max * 3600:
            # Compute terminal voltage from Shepherd model
            V_term = self.terminal_voltage(soc, I_avg, temp_k)

            # Check voltage cutoff condition
            if V_term < self.V_cutoff:
                break

            # Update current: I = P / V (from P = V·I)
            I_new = P_load / V_term if V_term > 0 else I_avg
            I_avg = 0.9 * I_avg + 0.1 * I_new  # Smooth to prevent oscillations

            # Euler step: SOC(t+Δt) = SOC(t) + (dSOC/dt)·Δt
            # where dSOC/dt = -I(t) / [Q₀(T)·3600]  (convert Ah to As)
            dsoc = -I_avg * dt / (Q_eff * 3600)
            soc += dsoc
            t += dt

            time_points.append(t / 3600)
            soc_points.append(soc)
            voltage_points.append(V_term)
            current_points.append(I_avg)

        return {
            "time": np.array(time_points),
            "soc": np.array(soc_points),
            "dod": 1 - np.array(soc_points),
            "voltage": np.array(voltage_points),
            "current": np.array(current_points),
            "discharge_time": time_points[-1],
            "temp_k": temp_k,
            "capacity": Q_eff,
        }


if __name__ == "__main__":
    # Test Main Model
    print("=" * 70)
    print("Main Model: Shepherd + Arrhenius Temperature Correction")
    print("=" * 70)

    model = MainBatteryModel(Q0=5.0)

    print(f"\nShepherd Parameters (fitted from XJTU):")
    print(f"  E₀ = {SHEPHERD_E0:.4f} V")
    print(f"  K  = {SHEPHERD_K:.4f}")
    print(f"  A  = {SHEPHERD_A:.4f} V")
    print(f"  B  = {SHEPHERD_B:.4f}")
    print(f"  R₀ = {SHEPHERD_R0:.4f} Ω")

    print(f"\nTemperature Effects:")
    for temp_c in [-10, 0, 10, 25, 40, 50]:
        temp_k = temp_c + 273.15
        Q_eff = model.get_capacity_at_temp(temp_k)
        print(f"  {temp_c:3d}°C: Q={Q_eff:.3f}Ah ({Q_eff/model.Q0*100:.1f}%)")

    print(f"\nDischarge Test (P=1.478W, T=25°C):")
    result = model.predict_discharge(1.478, temp_k=298.15)
    print(f"  Discharge Time: {result['discharge_time']:.3f} hours")
    print(f"  Final SOC: {result['soc'][-1]:.3f}")
    print(f"  Final Voltage: {result['voltage'][-1]:.3f} V")

    print("\n" + "=" * 70)
