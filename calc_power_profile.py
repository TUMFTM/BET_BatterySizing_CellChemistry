"""
Created on Mar 12 13:42:35 2023

@author: schneiderFTM
"""

# Calculate a power profile, which serves as input for the thermo electric model
# Result is shown in Firure 3b)

import numpy as np

def generatePowerProfiles(scenario_vec, p_charge_mcs, p_charge_home, p_dem, t_vec, gvw, bat_weight, m_wo_bat):

    # Used Timestep
    dt = 1  # in sec

    # Intepolating Vecto-Power-Profile to constant time-step

    # Initialization
    p_profile = np.zeros((1))
    change_step = 0
    scenario_vec.append(0)

    for idx_step, step in enumerate(scenario_vec[:-1]):

        if scenario_vec[idx_step + 1] != step:
            # Insert according to state, depending on corresponding payload
            if step >= 0:  # Driving with corresponding payload
                duration = idx_step - change_step

                # Select power demand according to payload and battery weight
                gvw_state = step + m_wo_bat + bat_weight  # in kg
                # Find closest corresponding power profile
                idx_closest = (np.abs(gvw - gvw_state)).argmin()
                # Interpolate to time step
                t_vec_inter = np.arange(0, t_vec[idx_closest][-1], dt)
                p_bat_vec = np.interp(t_vec_inter, t_vec[idx_closest], p_dem[idx_closest] * (-1))  # W
                # Stack full cycles power demand
                full_cycles = np.floor(duration/t_vec[idx_closest][-1])
                p_profile = np.concatenate((p_profile, np.tile(p_bat_vec, int(full_cycles))))
                # Add remaining time
                p_profile = np.concatenate((p_profile, p_bat_vec[0:int((duration-full_cycles*t_vec_inter[-1]))]))
                # Set Change State
                change_step = idx_step

            if step == -2:  # Insert defined Charging Power en route with MCS
                p_profile = np.concatenate((p_profile, np.tile(p_charge_mcs * 1000, int(idx_step - change_step))))
                change_step = idx_step
            if step == -3:  # Insert defined Charging Power at Home Depot
                p_profile = np.concatenate((p_profile, np.tile(p_charge_home * 1000, int(idx_step - change_step))))
                change_step = idx_step
            if step == -1:  # Insert zero, representing no power demand during plug/pay time sequence
                p_profile = np.concatenate((p_profile, np.tile(0, int(idx_step - change_step))))
                change_step = idx_step

    # Clean Vectors
    p_profile = np.delete(p_profile, 0)  # Remove initial 0
    scenario_vec.pop(-1)  # Remove additional 0

    return p_profile