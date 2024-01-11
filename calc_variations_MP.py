"""
Created on Fri Feb 3 8:42:33 2023

@author: schneiderFTM
"""

# Calculation of all defined parameter variations
# Results: Thermo-Electctric Values and stress factor values for aging Calculcation

#Import functions and Libaries
from calc_thermo_electric_values import calc_thermo_electric_values
from feasibility_check import check_feasibility
from calc_power_profile import generatePowerProfiles
from loadCell import load_cell_data
import numpy as np
from Ageing_Model import calc_ageing_naumann
from Ageing_Model import calc_ageing_schmalstieg


def calc_variations(array_parameters_row,par,v_gvw_inter,energy_gvw_inter, cell_vec, vec_payload, vec_chemistry,
                    vec_bat_size, vec_daily_range, vec_p_charge_mcs, vec_p_charge_home,
                    vec_temperature, p_dem, t_vec, gvw, idx_row, feasibility_only):

    #Refactor variation index
    idx_payload = array_parameters_row[0]
    idx_chemistry = array_parameters_row[1]
    idx_bat_size = array_parameters_row[2]
    idx_daily_range = array_parameters_row[3]
    idx_p_charge_mcs = array_parameters_row[4]
    idx_p_charge_home = array_parameters_row[5]
    idx_temperature = array_parameters_row[6]

    ########################################################################################################
    # General Feasibility Checks
    ########################################################################################################
    # Check if Battery Size is enough for defined UseCase and Scenario
    res, time_for_range, val_bat_weight = check_feasibility(par, vec_payload[idx_payload],
                                                            vec_daily_range[idx_daily_range],
                                                            vec_bat_size[idx_bat_size], vec_chemistry[idx_chemistry],
                                                            v_gvw_inter)

    res_check_gvw = res[0] # Is GVW feasible ?
    res_check_volume = res[1] # Is Battery Volume feasible ?
    res_check_travel_time = res[3] # Is the travel time conflicting to national law ?

    ########################################################################################################
    # Scenario Definition
    ########################################################################################################
    # Scenario Definition: Driving Stint // Charging en route // Driving Stint  see Figure 3

    #Defintion of duration of different states
    scenario_time = [(time_for_range / 2) * 60, par.connection_time / 2, 45 - par.connection_time,
                     par.connection_time / 2, (time_for_range / 2) * 60, par.connection_time]

    #Definition of state category
    # >=0=drving with payload // -1 = no action // -2=charging@MCS // -3=charging@Home // -4=charging@Customer
    scenario_states = [vec_payload[idx_payload], -1, -2, -1, vec_payload[idx_payload], -1]

    # Scenario Resulution
    resulution = 1  # in s

    #Refactor to final scenario vector, which contains which state is occuring at every time step
    scenario_vec = []
    for idx_time_span, time_span in enumerate(scenario_time):
        scenario_vec.extend([scenario_states[idx_time_span] for i in range(int(time_span / (resulution / 60)))])


    ########################################################################################################
    # Thermo-Electric Calculation
    ########################################################################################################

    ### Generate Power Profile -------------------------------------------
    # Generate Power Profile according to defined scenario vector
    p_profile = generatePowerProfiles(scenario_vec, vec_p_charge_mcs[idx_p_charge_mcs],
                                      vec_p_charge_home[idx_p_charge_home], p_dem, t_vec, gvw, val_bat_weight,
                                      par.m_wo_bat)

    ### Select Chemistry/Models ------------------------------------------
    # Chemistry // 0:Schmalstieg et. al. // 1:Naumann et.al.
    cell = cell_vec[idx_chemistry]

    ### Electro-Thermal-Ageing Model--------------------------------------------

    ### Calculation 1 // Is the parameter variation feasible at EndofLife( 80% SOH // 50% IR ) ?

    eol = True
    # eol == True: Modify Cell Data in order to simulate Cell at End of Life // Reduced Capacity + Increased IR
    # eol == False: Use initial cell data

    p_value_control, EoL_SOC, Crate, T_Cell, check_SOC, Q_throughput_veh = calc_thermo_electric_values(par, cell, p_profile,
                                                                                 vec_bat_size[idx_bat_size],
                                                                                 resulution,
                                                                                 par.gravimetric_energy_density[
                                                                                     vec_chemistry[idx_chemistry]],
                                                                                 par.c2p_grav[
                                                                                     vec_chemistry[idx_chemistry]], eol, scenario_vec)
    res_min_SOC = min(EoL_SOC[0:-2])
    res_check_SOC = check_SOC
    res_min_p_home = ((1-EoL_SOC[-2]) * vec_bat_size[idx_bat_size])/((24) - (sum(scenario_time)/60))


    ### Calculation 2 // When parameter variation is feasible at end of life
    #                 // Calculate aging stress factors with cell at 100% SOH
    if check_SOC == 0 and feasibility_only == False:

        # Adapt Scenario definiton with addition of charging at the home depot and rest time for full 24h
        scenario_time_ageing= scenario_time.copy()
        scenario_time_ageing.append((24 * 60) - (sum(scenario_time)))
        scenario_states_ageing = [vec_payload[idx_payload], -1, -2, -1, vec_payload[idx_payload], -1, -3]
        scenario_vec_ageing = []
        for idx_time_span, time_span in enumerate(scenario_time_ageing):
            scenario_vec_ageing.extend([scenario_states_ageing[idx_time_span] for i in range(int(time_span / (resulution / 60)))])


        #  Generate Power Profile according to GVW and defined scenario
        p_profile_ageing = generatePowerProfiles(scenario_vec_ageing, vec_p_charge_mcs[idx_p_charge_mcs],
                                          vec_p_charge_home[idx_p_charge_home], p_dem, t_vec, gvw, val_bat_weight,
                                          par.m_wo_bat)

        #  Calculate Thermo-Electric and aging stress factors with 100% SOH cell
        eol = False
        BoL_p_value_control, BoL_SOC, BoL_Crate, BoL_T_Cell, BoL_check_SOC, BoL_Q_throughput_veh = calc_thermo_electric_values(par, cell, p_profile_ageing,
                                                                                     vec_bat_size[idx_bat_size],
                                                                                     resulution,
                                                                                     par.gravimetric_energy_density[
                                                                                         vec_chemistry[idx_chemistry]],
                                                                                     par.c2p_grav[
                                                                                         vec_chemistry[idx_chemistry]],
                                                                                     eol, scenario_vec)

        # Calculate aging stress factors for given parameter variation
        if vec_chemistry[idx_chemistry] == 0:
            # NMC
            BoL_SOC[-1] = BoL_SOC[0]
            res_k_factors_operation_classic = calc_ageing_schmalstieg(cell, BoL_T_Cell, BoL_SOC, BoL_Crate, calendaric=0)

        if vec_chemistry[idx_chemistry] == 1:
            # LFP
            BoL_SOC[-1] = BoL_SOC[0]
            res_k_factors_operation_classic = calc_ageing_naumann(cell, BoL_T_Cell, BoL_SOC, BoL_Crate, calendaric=0)



        #Calculate the share of fast-charged Energy for given Variation
        res_share_energy = calc_share_fc(BoL_SOC, resulution)


    else:
        BoL_SOC = np.zeros(len(EoL_SOC))
        BoL_Crate = np.zeros(len(Crate))
        BoL_T_Cell = np.zeros(len(T_Cell))
        res_k_factors_operation_classic = 0
        BoL_Q_throughput_veh = 0
        res_share_energy = [0, 0]

    print(idx_row) # Log which parameter variation is calculated

    return [array_parameters_row, res_check_gvw, res_check_volume, res_check_travel_time, BoL_SOC, BoL_T_Cell,
            BoL_Crate, res_min_SOC, res_check_SOC, res_min_p_home, EoL_SOC,
            res_k_factors_operation_classic, BoL_Q_throughput_veh, res_share_energy]






def calc_variations_postprocessing(results, vec_payload, vec_chemistry, vec_bat_size, vec_daily_range, vec_p_charge_mcs, vec_p_charge_home, vec_temperature):
    # Refator fullfactorial parameter variation back to the corresponding result array

    #Initialize result arrays
    # Array ( Payload // Chemistry // Bat-Size // Range //  P_MCS // P_HomeDepot // Temperature )
    res_check_gvw = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), 1))
    res_check_volume = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), 1))
    res_check_travel_time = np.zeros(
        (len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range), 1))
    res_BoL_SOC = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
                            len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature),
                            len(results[0][4])))
    res_EoL_SOC = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
                        len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), len(results[0][4])))
    res_BoL_T_Cell = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
                               len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature),
                               len(results[0][5])))
    res_BoL_c_rate = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
                               len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature),
                               len(results[0][6])))
    res_check_SOC = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
                              len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), 1))
    res_min_SOC = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
                            len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), 1))
    res_check_home_SOC = np.zeros((len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
                                   len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), 1))
    res_k_factors_operation_classic = np.zeros(
        (len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
         len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), 5))

    res_Q_throuput_veh = np.zeros(
        (len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
         len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), 1))

    res_share_energy = np.zeros(
        (len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
         len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), 2))

    # Refactor to result arrays
    for res in results:

        # Adapt Length of SOC/T_Cell to size of result array:
        if len(res[4]) > len(results[0][4]):
            res_SOC_value = res[4][:len(results[0][4])]
        elif len(res[4]) < len(res_BoL_SOC[0, 0, 0, 0, 0, 0, 0, :]):
            res_SOC_value = res[4]
            while len(res_SOC_value) < (len(res_BoL_SOC[0, 0, 0, 0, 0, 0, 0, :])):
                res_SOC_value = np.append(res_SOC_value, results[0][4][-2])
        else:
            res_SOC_value = res[4]

        if len(res[5]) > len(results[0][5]):
            res_T_Cell_value = res[5][:len(results[0][5])]
        elif len(res[5]) < len(res_BoL_T_Cell[0, 0, 0, 0, 0, 0, 0, :]):
            for _ in range(len(results[0][5]) - len(res_BoL_T_Cell[0, 0, 0, 0, 0, 0, 0, :])):
                res_T_Cell_value = np.append(res_T_Cell_value, results[0][5][-2])
        else:
            res_T_Cell_value = res[5]

        if len(res[6]) > len(results[0][6]):
            res_c_rate_value = res[6][:len(results[0][6])]
        elif len(res[6]) < len(res_BoL_c_rate[0, 0, 0, 0, 0, 0, 0, :]):
            for _ in range(len(res[6]) - len(res_BoL_c_rate[0, 0, 0, 0, 0, 0, 0, :])):
                res_c_rate_value = np.append(res_c_rate_value, results[0][6][-2])
        else:
            res_c_rate_value = res[6]

        if len(res[10]) > len(res_EoL_SOC[0, 0, 0, 0, 0, 0, 0, :]):
            res_EoL_SOC_value = res[10][:len(results[0][10])]
        elif len(res[10]) < len(res_EoL_SOC[0, 0, 0, 0, 0, 0, 0, :]):
            res_EoL_SOC_value = res[10]
            for _ in range(len(res_EoL_SOC[0, 0, 0, 0, 0, 0, 0, :])- len(res[10])):
                res_EoL_SOC_value = np.append(res_EoL_SOC_value, np.nan)
        else:
            res_EoL_SOC_value = res[10]


        res_check_gvw[res[0][0], res[0][1], res[0][2]] = res[1]
        res_check_volume[res[0][0], res[0][1], res[0][2]] = res[2]
        res_check_travel_time[res[0][0], res[0][1], res[0][2]] = res[3]
        res_BoL_SOC[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res_SOC_value
        res_BoL_T_Cell[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res_T_Cell_value
        res_BoL_c_rate[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res_c_rate_value
        res_min_SOC[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res[7]
        res_check_SOC[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res[8]
        res_check_home_SOC[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res[9]
        res_EoL_SOC[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res_EoL_SOC_value
        res_k_factors_operation_classic[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res[11]
        res_Q_throuput_veh[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res[12]
        res_share_energy[res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5], res[0][6]] = res[13]

    return res_check_gvw, res_check_volume, res_check_travel_time, res_BoL_SOC, res_BoL_T_Cell,res_BoL_c_rate, res_min_SOC, res_check_SOC, res_check_home_SOC, res_k_factors_operation_classic, res_EoL_SOC, res_Q_throuput_veh, res_share_energy



def calc_variations_preprocessing( vec_payload, vec_chemistry, vec_bat_size, vec_daily_range, vec_p_charge_mcs, vec_p_charge_home, vec_temperature):

    # Create Cell-Vector with loaded Cell-Parameters and an array with the fullfactorial cominations of parameters

    #Load Chemistry/Models
    # Chemistry // 0:Schmalstieg et. al. // 1:Naumann et.al.
    cell_vec = []
    for i in range(len(vec_chemistry)):
        cell_vec.append(load_cell_data(vec_chemistry[i])) # Load Cell Data

    # Index Combinations
    vec_payload_idxcomb = np.arange(0, len(vec_payload), 1)
    vec_chemistry_idxcomb = np.arange(0, len(vec_chemistry), 1)
    vec_bat_size_idxcomb = np.arange(0, len(vec_bat_size), 1)
    vec_daily_range_idxcomb = np.arange(0, len(vec_daily_range), 1)
    vec_p_charge_mcs_idxcomb = np.arange(0, len(vec_p_charge_mcs), 1)
    vec_p_charge_home_idxcomb = np.arange(0, len(vec_p_charge_home), 1)
    vec_temperature_idxcomb = np.arange(0, len(vec_temperature), 1)
    array_parameters = np.array(
        np.meshgrid(vec_payload_idxcomb, vec_chemistry_idxcomb, vec_bat_size_idxcomb, vec_daily_range_idxcomb,
                    vec_p_charge_mcs_idxcomb, vec_p_charge_home_idxcomb, vec_temperature_idxcomb)).T.reshape(-1, 7)

    return cell_vec, array_parameters



def calc_share_fc(SOC, dt):
    # Extract share of fast charged and slow charged Energy for the given parameter Variation
    # dt in s

    #Iteration through calculated SOC vector
    start_index_FC = None
    start_index_SC = None

    for i in range(len(SOC) - 15 * int((60/dt))):
        # Extract the subarray consisting of the next 15*60 entries (15min charging)
        subarray = SOC[i:i + 15 * int((60/dt))]
        # Calc diffs
        differences = np.diff(subarray)
        # Check if all complete subarray is a charging stint
        decission = np.all(differences > 0)
        # If charging event is detected - calculate share of fast charged energy
        if decission == True:
            start_index_FC = i
            DoD_FC = SOC[start_index_FC + (40 * int((60/dt)))] - SOC[start_index_FC]
            break

    for i in range(len(SOC) - 60 * int((60/dt))):
        # Extract the subarray consisting of the next 60*60 entries (60min charging -> SlowCharging)
        subarray = SOC[i:i + 60 * int((60/dt))]
        # Calc diffs
        differences = np.diff(subarray)
        # Check if all complete subarray is a charging stint
        decission = np.all(differences > 0)
        # If charging event is detected - calculate share of slow charged energy
        if decission == True:
            start_index_SC = i
            DoD_SC = max(SOC) - SOC[start_index_SC]
            break

    share_FC = DoD_FC / (DoD_FC + DoD_SC)
    share_SC = DoD_SC / (DoD_FC + DoD_SC)

    return [share_FC, share_SC]