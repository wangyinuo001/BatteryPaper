import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ==================== Battery Parameters ====================
Q_NOMINAL_MAH = 4000.0  # Nominal Capacity (mAh)
VOLTAGE_NOMINAL = 3.8  # Nominal Voltage (V)
R_INTERNAL_NOMINAL = 0.15  # Internal Resistance (Ohms)

# Shepherd Model Parameters (Fitted from XJTU Batch-1 Battery-1 Cycle 0)
SHEPHERD_E0 = 3.3843  # Open-circuit voltage (V)
SHEPHERD_K = 0.0175  # Polarization constant
SHEPHERD_A = 0.8096  # Exponential zone amplitude (V)
SHEPHERD_B = 1.0062  # Exponential zone rate (with Q0 factor)

# Temperature Coefficients
TEMP_REF = 298.15  # 25 degrees C in Kelvin
# Logistic temperature model for capacity: Q0(T) = Q00 / (1 + exp(a + b/T))
# At 25°C (298.15K), we want Q0(T) ≈ Q00, so a + b/T_ref ≈ 0
# Using a = -b/T_ref makes the factor ≈1 at reference temperature
TEMP_Q_A = -10.05  # Adjusted so that at 298.15K, factor ≈ 1
TEMP_Q_B = 3000.0  # Temperature sensitivity (K)
# Arrhenius model for resistance: R(T) = R0 * exp(-Er/(kB*T))
# At 25°C, we want R(T) ≈ R0, which requires Er/kB/T_ref to be small
TEMP_R_ER = 0.005  # Reduced activation energy (eV) for realistic behavior
TEMP_KB = 8.617e-5  # Boltzmann constant (eV/K)

# Component Power Coefficients
# Screen: P_screen = η(γ_ls*A_screen*B + λ_s*A_screen) where γ_ls*A_screen=0.4991, λ_s*A_screen=0.1113
# Units: W (converted to mW in calculation)
GAMMA_LS_A_SCREEN = 0.4991  # γ_ls * A_screen
LAMBDA_S_A_SCREEN = 0.1113  # λ_s * A_screen

# SoC/CPU power
P_SOC_MAX = 2040.0  # mW (2.04 W max)
P_SOC_IDLE = 0.0  # mW (idle)

# Radio power (4G)
P_4G_IDLE = 0.0
P_4G_ACTIVE = 1173.0  # mW (1.173 W max)

# WiFi power
P_WIFI_IDLE = 0.0
P_WIFI_ACTIVE = 200.0  # mW (estimated)

# 5G power
P_5G_IDLE = 0.0
P_5G_ACTIVE = 1200.0  # mW (estimated)

# GPS power
P_GPS_ON = 300.0  # mW

# Baseline idle power
P_BASE_IDLE = 120.0  # mW

# Temperature-dependent leakage power coefficients
LEAKAGE_C1 = 0.0
LEAKAGE_C2 = 0.0


class BatteryModel:
    def __init__(self, q_nom_mah=Q_NOMINAL_MAH, cycle_count=0, calendar_time_hours=0):
        """
        Battery model with data-driven aging from XJTU dataset

        Parameters:
        - q_nom_mah: Nominal capacity in mAh (default 4000)
        - cycle_count: Number of charge-discharge cycles completed
        - calendar_time_hours: Cumulative usage time in hours
        """
        self.q_nom_mah = q_nom_mah
        self.cycle_count = cycle_count
        self.calendar_time_hours = calendar_time_hours

        # Fitted aging parameters from XJTU Batch-1 data (averaged across 8 batteries)
        # Model forms consistent with paper Equations 24-26:
        #   R(N) = R0_nom * (1 + beta_R * N^gamma)        [Eq. 24]
        #   Q(N) = Q0_nom * (1 - alpha_Q * N^gamma)        [Eq. 25]
        #   Full: Q(N,ts,T,SOC) = Q0*(1 - alpha_Q*N^gamma - kappa*sqrt(ts)*exp(-Ea/(Rg*T))*exp(lambda*SOC))  [Eq. 26]

        # 1. Resistance growth (Power Law with offset)
        self.R0_nom_aging = 0.0584  # Mean nominal resistance (Ohm) from XJTU fitting
        self.beta_R = 0.001229  # Resistance growth coefficient (from fitting)
        self.gamma_aging = 1.085  # Power-law exponent (from fitting)

        # 2. Capacity fade (cycle-dependent, power-law)
        self.Q0_ref = 2.006  # Reference capacity (Ah) from XJTU dataset
        self.alpha_Q = 0.000256  # Capacity fade coefficient (from fitting)
        # gamma_aging reused from resistance model (consistent exponent)

        # 3. Calendar aging parameters
        self.kappa_cal = 1e-5  # Calendar aging coefficient (1/sqrt(day))
        self.Ea_aging = 50000.0  # Activation energy (J/mol)
        self.Rg_aging = 8.314  # Gas constant (J/(mol*K))
        self.lambda_soc = 1.5  # SOC acceleration coefficient

        # 4. Capacity-Resistance linear correlation: Q = Q0_nom - kQ*(R - R0_nom)
        self.kQ = 7.7231  # Ah/Ohm (from fitting)

        # 5. Standard voltage E0 (nearly constant, CV < 1%)
        self.E0_nominal = 3.5673  # Mean value from data

    def get_aging_factors(self):
        """
        Calculate aging factors based on fitted models from XJTU data.

        Models (consistent with paper Eqs. 24-26):
          R(N) = R0_nom * (1 + beta_R * N^gamma)   → aging_factor_r = 1 + beta_R * N^gamma
          Q(N,ts) = Q0 * (1 - alpha_Q * N^gamma - calendar_term)
                                                    → aging_factor_q = 1 - total_loss

        Returns:
        - aging_factor_q: Capacity retention factor (0-1)
        - aging_factor_r: Resistance growth factor (≥1)
        """
        N = max(0, self.cycle_count)

        # --- Cycle aging ---
        if N > 0:
            cycle_capacity_loss = self.alpha_Q * np.power(N, self.gamma_aging)
            resistance_growth = self.beta_R * np.power(N, self.gamma_aging)
        else:
            cycle_capacity_loss = 0.0
            resistance_growth = 0.0

        # --- Calendar aging (Eq. 26): kappa * sqrt(ts_days) * Arrhenius * SOC factor ---
        t_days = max(0, self.calendar_time_hours) / 24.0
        if t_days > 0:
            temp_k = 298.15  # Default 25°C; overridden by get_effective_parameters
            arrhenius = np.exp(-self.Ea_aging / (self.Rg_aging * temp_k))
            soc_factor = np.exp(self.lambda_soc * 0.5)  # Average SOC ~ 0.5
            cal_loss = self.kappa_cal * np.sqrt(t_days) * arrhenius * soc_factor
        else:
            cal_loss = 0.0

        # Total capacity retention
        total_loss = cycle_capacity_loss + cal_loss
        aging_factor_q = max(0.30, min(1.0, 1.0 - total_loss))  # Floor at 30%

        # Resistance growth factor: R = R_nom * aging_factor_r
        aging_factor_r = 1.0 + resistance_growth
        aging_factor_r = max(1.0, min(3.0, aging_factor_r))  # Cap at 3x

        return aging_factor_q, aging_factor_r

    def get_effective_parameters(self, temp_k):
        """
        Temperature-dependent capacity and resistance from hand-written formulas:
        Q0(T) = Q00 / (1 + exp(a + b/T))  (Logistic)
        R(T) = R0 * exp(-Er/(kB*T))       (Arrhenius)

        Normalized so that at T_ref=298.15K (25°C), factors ≈ 1
        Returns (Q_eff_mAh, R_eff_ohm)
        """
        aging_factor_q, aging_factor_r = self.get_aging_factors()

        # Temperature effects using hand-written formulas
        # Q0(T) = Q00 / (1 + exp(a + b/T))
        # Normalize to 1 at reference temperature
        logistic_ref = 1.0 / (1.0 + np.exp(TEMP_Q_A + TEMP_Q_B / TEMP_REF))
        logistic_t = 1.0 / (1.0 + np.exp(TEMP_Q_A + TEMP_Q_B / temp_k))
        temp_factor_q = logistic_t / logistic_ref

        # R(T) = R0 * exp(-Er/(kB*T))
        # Normalize to 1 at reference temperature
        arrhenius_ref = np.exp(-TEMP_R_ER / (TEMP_KB * TEMP_REF))
        arrhenius_t = np.exp(-TEMP_R_ER / (TEMP_KB * temp_k))
        temp_factor_r = arrhenius_t / arrhenius_ref

        q_eff_mah = self.q_nom_mah * aging_factor_q * temp_factor_q
        r_eff = R_INTERNAL_NOMINAL * aging_factor_r * temp_factor_r
        return q_eff_mah, r_eff

    def calculate_terminal_voltage(self, soc, current_amps, temp_k):
        """
        Shepherd model (paper equation):
        V = E0 - R*i - k*(1/SOC)*i + A*exp(-B*Q0*(1-SOC))
        """
        soc = np.clip(soc, 0.01, 1.0)
        q_eff_mah, r_eff = self.get_effective_parameters(temp_k)
        q_eff_ah = q_eff_mah / 1000.0

        e0 = SHEPHERD_E0
        k = SHEPHERD_K
        a = SHEPHERD_A
        b = SHEPHERD_B

        exponential = a * np.exp(-b * q_eff_ah * (1 - soc))
        polarization = k * (current_amps / soc)
        v_term = e0 - r_eff * current_amps - polarization + exponential
        return v_term

    def get_current_from_power(self, power_mw, soc, temp_k):
        power_w = power_mw / 1000.0
        # Use VOLTAGE_NOMINAL from global constants
        i_guess = max(0.01, power_w / VOLTAGE_NOMINAL)

        def func(i):
            v = self.calculate_terminal_voltage(soc, i, temp_k)
            return v * i - power_w

        try:
            i_sol = fsolve(func, i_guess)[0]
        except Exception:
            i_sol = i_guess

        return max(i_sol, 0.0)

    def get_power_consumption(self, t, inputs):
        """
        Calculate total power consumption based on component usage.

        Power model formulas (from paper):
        - Screen: P_screen = η(γ_ls*A_screen*B + λ_s*A_screen) where B∈[0,1], η is refresh multiplier
        - SoC: P_SoC = ξ₁ * P_SoC^max where ξ₁∈[0,1] is CPU load
        - Radio: P_radio = ξ₂ * P_radio^max where ξ₂∈[0,1] is radio activity
        - GPS: P_GPS = 300mW when active
        - Baseline: P_baseline = 120mW (always on)

        Returns: Total power in mW
        """

        def get_val(key, t_val):
            val = inputs.get(key)
            if callable(val):
                return val(t_val)
            return val

        # Get input parameters
        # Check if brightness_B and brightness_eta are provided separately (new format)
        if "brightness_B" in inputs and "brightness_eta" in inputs:
            B = get_val("brightness_B", t)  # B ∈ [0,1]
            eta = get_val("brightness_eta", t)  # η refresh multiplier
        else:
            # Backward compatibility: bright = B*η combined
            bright = get_val("brightness", t)
            # Estimate: if bright > 1.0, assume high refresh η=1.3
            if bright > 1.0:
                eta = 1.3
                B = bright / eta
            else:
                eta = 1.0
                B = bright

        cpu = get_val("cpu_load", t)  # ξ₁ (0-1)
        net_mode = inputs.get("network_mode", "off")
        net_act = get_val("network_activity", t)  # ξ₂ (0-1)
        gps_on = get_val("gps", t)

        # Screen power: P_screen = η(γ_ls*A_screen*B + λ_s*A_screen)
        # Correct formula: P_screen (W) = η * (GAMMA_LS_A_SCREEN * B + LAMBDA_S_A_SCREEN)
        # When η=0, the entire screen power is 0
        p_screen_w = eta * (GAMMA_LS_A_SCREEN * B + LAMBDA_S_A_SCREEN)
        p_screen = p_screen_w * 1000.0  # Convert W to mW

        # SoC/CPU power: P_SoC = ξ₁ * P_SoC^max
        p_cpu = cpu * P_SOC_MAX  # mW

        # Radio power: P_radio = ξ₂ * P_radio^max
        p_net = 0.0
        if net_mode == "wifi":
            p_net = P_WIFI_IDLE + (P_WIFI_ACTIVE - P_WIFI_IDLE) * net_act
        elif net_mode == "4g":
            p_net = P_4G_IDLE + (P_4G_ACTIVE - P_4G_IDLE) * net_act
        elif net_mode == "5g":
            p_net = P_5G_IDLE + (P_5G_ACTIVE - P_5G_IDLE) * net_act

        # GPS power
        p_gps = P_GPS_ON if gps_on else 0.0

        # Baseline power
        p_base = P_BASE_IDLE

        # Temperature-dependent leakage power: V_nominal * c1 * T^2 * exp(c2 / T)
        # Calibrated so that leakage is zero at 300 K: P_leak(T) - P_leak(300)
        temp_val = inputs.get("temperature")
        temp_k = temp_val(t) if callable(temp_val) else temp_val
        leak_c1 = inputs.get("leakage_c1", LEAKAGE_C1)
        leak_c2 = inputs.get("leakage_c2", LEAKAGE_C2)
        if temp_k is None:
            p_leak = 0.0
        else:
            v_term = VOLTAGE_NOMINAL
            p_leak_t = v_term * leak_c1 * (temp_k**2) * np.exp(leak_c2 / temp_k)
            p_leak_ref = v_term * leak_c1 * (300.0**2) * np.exp(leak_c2 / 300.0)
            p_leak = p_leak_t - p_leak_ref

        return p_screen + p_cpu + p_net + p_gps + p_base + p_leak

    def state_derivative(self, t, y, inputs):
        soc = y[0]
        if soc <= 0.01:
            return [0.0]

        temp_val = inputs.get("temperature")
        temp_k = temp_val(t) if callable(temp_val) else temp_val

        # Constant Power Mode
        p_total_mw = self.get_power_consumption(t, inputs)
        p_total_w = p_total_mw / 1000.0  # Convert mW to W

        # Get terminal voltage at current SOC
        current_amps = self.get_current_from_power(p_total_mw, soc, temp_k)
        v_term = self.calculate_terminal_voltage(soc, current_amps, temp_k)

        # Effective capacity
        q_eff_mah, _ = self.get_effective_parameters(temp_k)
        q_eff_ah = q_eff_mah / 1000.0

        # dSOC/dt = -P / (V * Q) where P in watts, V in volts, Q in Ah
        # This ensures constant power consumption
        if v_term > 0:
            dsoc_dt = -p_total_w / (v_term * q_eff_ah)
        else:
            dsoc_dt = 0.0

        return [dsoc_dt]

    def solve(self, duration_hours, inputs, soc_init=1.0, dt_eval=0.1):
        t_span = (0, duration_hours)
        t_eval = np.arange(0, duration_hours, dt_eval)

        fun = lambda t, y: self.state_derivative(t, y, inputs)

        def battery_empty(t, y):
            return y[0] - 0.01

        battery_empty.terminal = True
        battery_empty.direction = -1

        sol = solve_ivp(fun, t_span, [soc_init], t_eval=t_eval, events=battery_empty)
        return sol
