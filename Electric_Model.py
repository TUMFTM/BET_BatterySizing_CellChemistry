"""
Created on Mar 1 16:47:54 2023

@author: schneiderFTM
"""

#Calculation of new electric states and charging intelligence

# Load modules
import numpy as np

def electric_model(cell, T_Cell, SOC, p_value, z_eol_ri, z_eol_c, dt):
    # Calculate new electric state according to power value,  previous states and cell properties

    # Determine internal resistance
    R_i_ch = cell.R_i_ch_func(T_Cell, SOC) * (1+z_eol_ri)
    R_i_dch = cell.R_i_dch_func(T_Cell, SOC) * (1+z_eol_ri)
    # Determine cell volatage
    Uocv = cell.ocv_func(SOC)

    # Check if charging / discharging
    if p_value > 0:
        R_i = R_i_ch
    else:
        R_i = R_i_dch

    # Value Calculation
    Ibat = np.real((-Uocv + np.sqrt((Uocv ** 2) + (4 * R_i * p_value))) / (2 * R_i))
    C_rate = Ibat / (cell.Qnom * z_eol_c)
    SOC_new = SOC + (C_rate * dt / 3600)
    P_Loss = (Ibat ** 2) * R_i
    Q_throughput = Ibat * (dt/3600) # Ah


    return  SOC_new, C_rate,  P_Loss, Q_throughput



def electric_control(cell, p_value, SOC, T_Cell, Crate,z_eol_ri, z_eol_c,target_soc, dt):
    # Implementation of the CCCV charging protocol and overcharge prevention

    # CCCV - Charging
    if p_value > 0: # Check if vehicle is getting charged / Is current reduction needed ?
        # Calculation of actual voltage, which has to be lower than maximal cell voltage
        R_i_ch = cell.R_i_ch_func(T_Cell, SOC) * (1+z_eol_ri)
        Uocv = cell.ocv_func(SOC)
        Ibat = np.real((-Uocv + np.sqrt((Uocv ** 2) + (4 * R_i_ch * p_value))) / (2 * R_i_ch))
        U_act = Uocv + Ibat * R_i_ch  # Actual Voltage

        # Degradation of current in order to achieve a lower voltage than maximal cell voltage // Constant Voltage
        if U_act >= cell.Umax:
            for i in range(1000):
                if Uocv + Ibat*(1-0.001*i) * R_i_ch < cell.Umax:
                    p_value = Ibat*(1-0.001*i) * U_act
                    break

    # Prevent Overcharge and Implementation of Charging Strategy
    if p_value > 0:
        # Check if in the next step the SOC is higher as the maximal cell SOC
        if (SOC + ((Ibat/cell.Qnom * z_eol_c) * dt / 3600)) > cell.SOC_max:
             p_value = 0  # Stop Charging due to Overcharge

        if (SOC + ((Ibat/cell.Qnom * z_eol_c) * dt / 3600)) > target_soc:  # if needed Energy is recharged at public charger -> stop charging
            p_value = 0  # Stop Charging due no more energy is needed to complete mission

    # Check if Battery Capacity is not Feasible

    check_SOC = 0  # Battery Capacity is feasible // default
    if p_value < 0:
        if (SOC + (Crate * dt / 3600)) < cell.SOC_min:
            check_SOC = 1  # Battery Capacity is not feasible
    else:
        pass

    return p_value, check_SOC