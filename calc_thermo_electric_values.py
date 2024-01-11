"""
Created on Mar 10 10:22:57 2023

@author: schneiderFTM
"""

# Calculate the thermo-electric values bases on a input power profile

# Initialize functions and models
import numpy as np
from Electric_Model import electric_control
from Electric_Model import electric_model
from Thermal_Model import thermal_control
from Thermal_Model import thermal_model



def calc_thermo_electric_values(par, cell, p_profile, bat_size, dt, gravimetric_energy_density, c2p_grav, eol, scenario_vec):

    #  Scale Power profile on vehicle model to cell level
    n_cells = (bat_size * 1000) / (cell.Qnom * cell.Unom)
    p_cell = p_profile/n_cells

    #  Determine Start of Fast Charging Event
    scenario_vec = np.asarray(scenario_vec)
    idx_start_charge = np.where(scenario_vec == -2)[0][0]
    idx_stop_charge = np.where(scenario_vec == -2)[0][-1]+2

    #  Definition, if cell at end of life or at 100% SOH should be used
    if eol:
        z_eol_c = 0.8
        z_eol_ri = 0.5
        soc_puffer = 0
        SOC_RA = 0.05
    else:
        z_eol_c = 1
        z_eol_ri = 0
        soc_puffer = 0.2
        SOC_RA = 0.05

    #  Initialize Result-Arrays
    p_value_control = np.zeros((len(p_profile)))
    SOC = np.zeros((len(p_profile)+1))
    P_Loss = np.zeros((len(p_profile)))
    T_Cell = np.zeros((len(p_profile)+1))
    Crate = np.zeros((len(p_profile)))
    p_cool = np.zeros((len(p_profile)))
    p_heat = np.zeros((len(p_profile)))
    T_Housing = np.zeros((len(p_profile)+1))

    # Initialize Start Values
    T_Cell[0] = 20  # Start Temperature of cells
    T_Housing[0] = 20  # Start Temperature of Housing
    SOC[0] = cell.SOC_max  # Start SOC of cells
    T_amb = 20  # Start Ambient Temperature
    P_Cool = 0  # Initial Cooling Power
    target_soc = 1  # Initial Chariging Target
    Q_throughput_veh = 0  # Initial chariging throuput



    # Loop over vector of power-Values
    for idx_p_value, p_value in enumerate(p_cell[1:]):

        # Check if vehicle is charging en-route and calculation of target soc in order to achieve daily route
        if idx_p_value == idx_start_charge:
            delta_soc = cell.SOC_max - SOC[idx_p_value]  # Consumed Energy from Start to Charging Event

            if delta_soc <= (SOC[idx_p_value] - soc_puffer - SOC_RA - cell.SOC_min):  # Check if charging is  needed
                target_soc = SOC[idx_p_value]  # Target SOC = Current SOC
            if delta_soc > (SOC[idx_p_value] - soc_puffer - SOC_RA - cell.SOC_min):  #  Check if charging is  needed
                target_soc = (SOC[idx_p_value]) + (delta_soc - (SOC[idx_p_value] - soc_puffer)) + SOC_RA + cell.SOC_min
                # Target SOC = Needed Energy in order to fullfill driving task

        if idx_p_value > idx_stop_charge: # Check if FC is done and adapt Target SOC to the maximum chargeable SOC for SC
            target_soc = cell.SOC_max


        # Electric Control // CCCV Charging / Prevent Overcharge
        p_value_control[idx_p_value], check_SOC = electric_control(cell, p_value, SOC[idx_p_value], T_Cell[idx_p_value], Crate[idx_p_value],z_eol_ri, z_eol_c, target_soc, dt)

        # Feasibility Check
        if check_SOC == 1: #  If check SOC == 1 : Parameter Variation is not feasible and thermo-electric calclulation is aborted
            return p_value_control, SOC, Crate, T_Cell, check_SOC, Q_throughput_veh
            break

        # Thermal Control // Activation Cooling / Heating
        p_value_control[idx_p_value], p_cool[idx_p_value], p_heat[idx_p_value] = thermal_control(par, T_Cell[idx_p_value], p_cool[idx_p_value-1], p_value_control[idx_p_value], p_value, n_cells)

        # Electric Model // Calculation of new SOC / Crate / Power Loss
        SOC[idx_p_value+1], Crate[idx_p_value], P_Loss[idx_p_value], Q_throughput = electric_model(cell, T_Cell[idx_p_value], SOC[idx_p_value], p_value_control[idx_p_value],z_eol_ri, z_eol_c, dt)

        # Thermal Model // Calculation of new Temperature State
        T_Cell[idx_p_value+1], T_Housing[idx_p_value+1] = thermal_model(par, cell, n_cells, T_Cell[idx_p_value],T_Housing[idx_p_value] , T_amb, P_Loss[idx_p_value], p_cool[idx_p_value], p_heat[idx_p_value], dt, gravimetric_energy_density, c2p_grav)

        # Calculation of Charing Throuput
        Q_throughput_veh = Q_throughput_veh + abs(Q_throughput*n_cells)

    return p_value_control, SOC, Crate, T_Cell, check_SOC, Q_throughput_veh