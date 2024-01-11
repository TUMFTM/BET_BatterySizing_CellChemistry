"""
Created on Mar 13 12:12:16 2023

@author: schneiderFTM
"""

# Calculation of new Temperature State // Control of Heating/Cooling

def thermal_model(par, cell, n_cells, T_Cell, T_Housing, T_amb, P_Loss, Pcool, Pheat, dt, gravimetric_energy_density, c2p_grav):

    #Mass calculation of battery pack without cells, based on battery size
    m_battery_wo_cells = ((n_cells * cell.Qnom * cell.Unom) / gravimetric_energy_density) * (1-c2p_grav) # Mass of Pack without Cells
    # Thermal Capacity Cell
    Cth_Battery = cell.mass * par.cp_cell # J/(kgK)
    # Thermal Capacity Pack without Cells
    Cth_Housing = par.c_aluminium * m_battery_wo_cells # J/K


    # New Housing Temperature
    T_Housing_new = T_Housing + dt * (((Pcool * par.COPcool) + (Pheat * par.COPheat) + par.k_bh * n_cells * (T_Cell - T_Housing) + par.k_out * (
                T_amb - T_Housing)) / Cth_Housing)
    # New Cell Temperature
    T_Cell_new = T_Cell + dt * ((P_Loss + par.k_bh * (T_Housing - T_Cell)) / Cth_Battery)

    return T_Cell_new, T_Housing_new


def thermal_control(par, T_Cell, p_cool_prev, p_value_control, p_value, n_cells):
   # Control Algorithm for active Cooling/Heating

   # Check if cell temperature is below heating threshold
    if T_Cell < par.T_Heat:
        P_Heat = par.Pheater
        P_Cool = 0

    # Cool if Cooling-Threshold is exeeded or
    # (if Cooling was active in the previous time step and Off-Threshold is not reached yet)
    elif T_Cell > par.T_Cool_on or (T_Cell > par.T_Cool_off and p_cool_prev > 0):
        P_Heat = 0
        P_Cool = par.Pcooler
    else:
        P_Heat = 0
        P_Cool = 0

    # Impact of Cooling on Power
    # Case Driving // Cooling Power added to Power demand of driving task
    if p_value <= 0:
        p_value_control = p_value_control + (P_Cool+P_Heat)/n_cells
    # Case Charging // Cooling Power from Infrastructure -> If Cell is limiting no further power demand from cooling
    else:
        if p_value > p_value_control:
            p_value_control = p_value_control
        if p_value >= p_value_control:
            p_value_control = p_value_control - (P_Cool + P_Heat) / n_cells

    return p_value_control, P_Cool, P_Heat
