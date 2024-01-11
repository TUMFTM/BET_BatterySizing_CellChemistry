"""
Created on Fri Mar 5 8:42:33 2023

@author: schneiderFTM
"""

# Spoilt for Choice: User-Centric Choice of Battery Size and Chemistry for Battery-Electric Long-Haul Trucks
#---------------------------------------------------------------------------------------
#https: // doi.org / 10.3390 / en17010158
#---------------------------------------------------------------------------------------
# Designed by: Jakob Schneider (FTM, Technical University of Munich)
#-------------
# ------------
# Version: Python 3.9
# -------------

# Description:
# This script is used to generate the results used in 'Spoilt for Choice: User-Centric Choice of Battery Size and
# Chemistry for Battery-Electric Long-Haul Trucks'

# The script is divided in 5 parts:
#   1. Parameter Initialization
#   2. Energy Consumtion Calculation
#   3. Scenario Calculation (inkl. Thermo-Electric + Aging Coefficient Calculation)
#   4. TCO Calculation
#   5. Plots
# ------------
# Input:
#   Input data (pickle files) with power profiles gernerated in CityMoS (Input/PowerProfile)
    # Cell Data (inputs/Cells)
    # Mission Profiles (inputs/MissionProfiles)
    # BET Market Overview (inputs/Market_Overview
# ------------
# Output:
#   Plots of the results
# ------------


# Import Libraries and functions
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import multiprocessing as mp
import time
from parameterclass import ParameterClass
from calc_energy_consumption import calc_energy_consumption_depending_gvw
from feasibility_check import calc_delta_weight
from calc_variations_MP import calc_variations
from calc_variations_MP import calc_variations_postprocessing
from calc_variations_MP import calc_variations_preprocessing
from calc_tco import calculate_tco
from calc_tco import calc_sensitivity_preprocessing
from feasibility_check import feasibility_lines
from Ageing_Model import calc_ageing
from Ageing_Model import calc_ageing_schmalstieg
from Ageing_Model import calc_ageing_naumann
from Plots import plot_feasibility_stacked
from Plots import plot_tco_km_sensitivity
from Plots import plot_market_overview
from Plots import plot_tco_km_paper
from Plots import plot_chem_advantage_paper
from Plots import plot_power_profile
from Plots import plot_chem_advantage_sensitivity



def exception_wrapper(args):
    try:
        return calc_variations(*args)
    except Exception as e:
        print(e)


if __name__ == "__main__":

    # ----------------------------------------------------------------------------------------
    ### 1. Model Parametrizion
    # ----------------------------------------------------------------------------------------
    par = ParameterClass()

    # ----------------------------------------------------------------------------------------
    ### 2. Energy Consumption depending on GVW
    # ----------------------------------------------------------------------------------------

    Energy_Calculation_Recalculate = False
    # True = Recalculation of Energy Consumption Interpolation Function
    # False = Reload precalculated result

    gvw = np.arange(20000, 42000, 1000)# Gross Vehicle Weight vector, for which energy consumption should be calculated

    if Energy_Calculation_Recalculate:
        # LDS-Simulation for each GVW-Value
        energy_gvw, v_gvw, p_dem, t_vec = calc_energy_consumption_depending_gvw(par, gvw)
        # Save calculated values
        np.save("saved_variables\\Energy_Consumption\\p_dem.npy", p_dem)
        np.save("saved_variables\\Energy_Consumption\\t_vec.npy", t_vec)
        np.save("saved_variables\\Energy_Consumption\\energy_gvw.npy", energy_gvw)
        np.save("saved_variables\\Energy_Consumption\\v_gvw.npy", v_gvw)
        np.save("saved_variables\\Energy_Consumption\\vec_gvw.npy", gvw)

    else :
        #Load previous calculated values
        p_dem = np.load("saved_variables\\Energy_Consumption\\p_dem.npy")
        t_vec = np.load("saved_variables\\Energy_Consumption\\t_vec.npy")
        energy_gvw = np.load("saved_variables\\Energy_Consumption\\energy_gvw.npy")
        v_gvw = np.load("saved_variables\\Energy_Consumption\\v_gvw.npy")

    # Calculate Interpolation-Functions for EnergyConsumption(GVW)
    energy_gvw_inter = interp1d(gvw, energy_gvw, kind='linear', fill_value='extrapolate')
    v_gvw_inter = interp1d(gvw, v_gvw, kind='linear', fill_value='extrapolate')

    # ----------------------------------------------------------------------------------------
    ### 3. Scenario Calculation (inkl. Thermo-Electric + Aging Coefficient Calculation)
    # ----------------------------------------------------------------------------------------

    Variation_Recalculate = False
    # True = Recalculation of Scenario Calculation
    # False = Reload precalculated results

    if Variation_Recalculate == True:

        ### Scenario Defnitions -----------------------------------------
        # Define which Parameter-Ranges should be calculated
        # Fullfactorial Combination of Parameters will be calculated in the following Code

        # Daily Range
        vec_daily_range = np.arange(600, 720, 200)  # Daily range a truck has to drive

        #Payload
        vec_payload = np.arange(0, 30000, 15000)  # Payload in kg

        # P_Charge @MCS
        vec_p_charge_mcs = np.arange(300, 1000, 800)  # Available Charging Power En-Route in kW

        # P_Charge @Home
        vec_p_charge_home = np.arange(50, 110, 70)  # Available Charging Power at the home depot in kW

        # Cell Chemistry
        vec_chemistry = [0, 1]  # 0 = NMC // 1 = LFP

        # Battery Size
        vec_bat_size = np.arange(500, 1050, 500)  # Battery Size in kWh

        # Outside Temperautre
        vec_temperature = np.arange(25, 30, 15)  # Temperature in

        ### -----------------------------------------

        #Calculate static feasibility based on maximal gross vehicle weight and maximal vehicle package
        max_payload, max_bat_size = feasibility_lines(par, vec_bat_size, vec_chemistry)


        # PREPROCESSING // Create Array with fullfactorial combination of parameters
        cell_vec, array_parameters = calc_variations_preprocessing(vec_payload, vec_chemistry, vec_bat_size, vec_daily_range, vec_p_charge_mcs,
                                      vec_p_charge_home, vec_temperature)


        feasibility_only = False
        # True = Only Calculate if Parameter Combinations are feasible and disable Thermo-Electric Calculation
        # False = Calculate Feasibility AND Thermo-Electric Values

        multiprocessing = True
        # True = Calculate Values using multiprocessing
        # False = Calculate Values using serial calculation (Debugging-Mode)

        if multiprocessing:
            tic = time.perf_counter()
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(exception_wrapper, [(row, par, v_gvw_inter, energy_gvw_inter, cell_vec, vec_payload, vec_chemistry,
                        vec_bat_size, vec_daily_range, vec_p_charge_mcs, vec_p_charge_home, vec_temperature, p_dem, t_vec, gvw, idx_row, feasibility_only) for idx_row, row in enumerate(array_parameters)])

            toc = time.perf_counter()
            print(f"Time: {toc - tic:0.4f} seconds")


        if multiprocessing == False:
            tic = time.perf_counter()
            results = []
            for idx_variation in tqdm(range(len(array_parameters))):
                results.append(calc_variations(array_parameters[idx_variation], par, v_gvw_inter, energy_gvw_inter, cell_vec, vec_payload, vec_chemistry,
                                 vec_bat_size, vec_daily_range, vec_p_charge_mcs, vec_p_charge_home, vec_temperature, p_dem, t_vec, gvw, idx_variation, feasibility_only))
            toc = time.perf_counter()
            print(f"Time: {toc - tic:0.4f} seconds")



        # POSTPROCESSING // Refactor results to result vectors
        res_check_gvw, res_check_volume, res_check_travel_time, res_BoL_SOC, res_BoL_T_Cell, res_BoL_c_rate, res_min_SOC, \
            res_check_SOC, res_min_p_home, res_k_factors_operation_classic, res_EoL_SOC, res_Q_throuput_veh, res_share_energy =\
            calc_variations_postprocessing(results, vec_payload, vec_chemistry, vec_bat_size, vec_daily_range,
                                       vec_p_charge_mcs, vec_p_charge_home, vec_temperature)



        # Save Results
        np.save("saved_variables\\ElectroThermal\\max_payload.npy", max_payload)
        np.save("saved_variables\\ElectroThermal\\max_bat_size.npy", max_bat_size)
        np.save("saved_variables\\ElectroThermal\\res_check_gvw.npy", res_check_gvw)
        np.save("saved_variables\\ElectroThermal\\res_check_volume.npy", res_check_volume)
        np.save("saved_variables\\ElectroThermal\\res_check_travel_time.npy", res_check_travel_time)
        np.save("saved_variables\\ElectroThermal\\res_check_SOC.npy", res_check_SOC)
        np.save("saved_variables\\ElectroThermal\\res_min_p_home.npy", res_min_p_home)
        np.save("saved_variables\\ElectroThermal\\res_min_SOC.npy", res_min_SOC)
        np.save("saved_variables\\ElectroThermal\\res_k_factors_operation_classic.npy", res_k_factors_operation_classic)
        np.save("saved_variables\\ElectroThermal\\res_share_energy.npy", res_share_energy)

        # Save parameterization
        np.save("saved_variables\\ElectroThermal\\vec_payload.npy", vec_payload)
        np.save("saved_variables\\ElectroThermal\\vec_chemistry.npy", vec_chemistry)
        np.save("saved_variables\\ElectroThermal\\vec_bat_size.npy", vec_bat_size)
        np.save("saved_variables\\ElectroThermal\\vec_daily_range.npy", vec_daily_range)
        np.save("saved_variables\\ElectroThermal\\vec_p_charge_mcs.npy", vec_p_charge_mcs)
        np.save("saved_variables\\ElectroThermal\\vec_p_charge_home.npy", vec_p_charge_home)
        np.save("saved_variables\\ElectroThermal\\vec_temperature.npy", vec_temperature)


    else:

        #Load previous calculated values
        max_payload = np.load("saved_variables\\ElectroThermal\\max_payload.npy")
        max_bat_size = np.load("saved_variables\\ElectroThermal\\max_bat_size.npy")
        res_check_gvw = np.load("saved_variables\\ElectroThermal\\res_check_gvw.npy")
        res_check_volume = np.load("saved_variables\\ElectroThermal\\res_check_volume.npy")
        res_check_travel_time = np.load("saved_variables\\ElectroThermal\\res_check_travel_time.npy")
        res_check_SOC = np.load("saved_variables\\ElectroThermal\\res_check_SOC.npy")
        res_check_home_SOC = np.load("saved_variables\\ElectroThermal\\res_min_p_home.npy")
        res_min_SOC = np.load("saved_variables\\ElectroThermal\\res_min_SOC.npy")
        res_k_factors_operation_classic = np.load("saved_variables\\ElectroThermal\\res_k_factors_operation_classic.npy")
        res_share_energy = np.load("saved_variables\\ElectroThermal\\res_share_energy.npy")

        # Load parameterization
        vec_payload = np.load("saved_variables\\ElectroThermal\\vec_payload.npy")
        vec_chemistry = np.load("saved_variables\\ElectroThermal\\vec_chemistry.npy")
        vec_bat_size = np.load("saved_variables\\ElectroThermal\\vec_bat_size.npy")
        vec_daily_range = np.load("saved_variables\\ElectroThermal\\vec_daily_range.npy")
        vec_p_charge_mcs = np.load("saved_variables\\ElectroThermal\\vec_p_charge_mcs.npy")
        vec_p_charge_home = np.load("saved_variables\\ElectroThermal\\vec_p_charge_home.npy")
        vec_temperature = np.load("saved_variables\\ElectroThermal\\vec_temperature.npy")


    # ----------------------------------------------------------------------------------------
    ### 4. TCO Calculation (incl. Aging Calculation // Sensitivity Analysis)
    # ----------------------------------------------------------------------------------------

    TCO_Recalculate = False
    # True = Recalculation of TCO Calculation
    # False = Reload precalculated results

    if TCO_Recalculate == True:

        # Sensitivity analyis parameter definition
        vec_annual_mileage = np.array([0.8*130000, 130000, 1.2*130000])  # Annual Mileage
        vec_cell_price = np.array([[0.8*140, 140, 1.2*140], [0.8 * 100, 100, 1.2 * 100]])# Cell Price
        vec_ene_sc = np.array([0.8*0.2515, 0.2515, 1.2*0.2515]) #Energy Cost @ Home
        vec_ene_fc = np.array([0.8*(0.663/1.19), (0.663/1.19), 1.2*(0.663/1.19)])


        # Result array initialization
        res_ageing = np.zeros(
            (len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
             len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature),len(vec_annual_mileage), 2))

        res_tco = np.zeros(
            (len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
             len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), len(vec_annual_mileage), np.size(vec_cell_price, axis=1), len(vec_ene_sc), len(vec_ene_fc), 10))

        res_m_delta = np.zeros(
            (len(vec_payload), len(vec_chemistry), len(vec_bat_size), len(vec_daily_range),
             len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), len(vec_annual_mileage), 1))


        # PREPROCESSING (Load Cell + Calculation Variation Array)
        cell_vec, array_parameters = calc_variations_preprocessing(vec_payload, vec_chemistry, vec_bat_size,
                                                                   vec_daily_range, vec_p_charge_mcs,
                                                                   vec_p_charge_home, vec_temperature)


        # Mass-Delta for Oportunity-Cost determination
        m_delta_lfp, m_delta_nmc = calc_delta_weight(vec_payload, vec_bat_size, vec_daily_range, vec_p_charge_mcs,
                                                     vec_p_charge_home, vec_temperature, res_check_volume, res_check_SOC)


        fullfactorial = True
        # True = Calculation of fullfactorial sensitivity analysis
        # False = Calculation of single value sensitivity analysis

        # Create variation array for sensitivity analysis
        array_sensitivity = calc_sensitivity_preprocessing(vec_annual_mileage, vec_cell_price, vec_ene_sc, vec_ene_fc,
                                                           fullfactorial)


        # Calculate TCO/Aging for every parameter and sensitivity analyis variation
        for variation in tqdm(array_parameters): # Loop over parameter variation
            for variation_sensi in array_sensitivity: #Loop over sensitivity-parameters

                # Lifetime Calculation -----------------------------------

                # Calulate Aging // NMC
                if variation[1] == 0:
                    # Calculate calendaric stress factor for rest days
                    k_factors_classic_rest = calc_ageing_schmalstieg(cell_vec[0], np.asarray([25]), 0.8, 0, calendaric=1)
                        # Assumption: 25°C // 80% SOC

                    # Load calculated stress factors
                    k_factors_operation_classic = res_k_factors_operation_classic[variation[0],variation[1], variation[2],
                    variation[3], variation[4], variation[5], variation[6]]


                    # Calculate battery lifetime
                    res_ageing_values = calc_ageing(par, variation[1],  vec_annual_mileage[variation_sensi[0]], vec_daily_range[variation[3]], k_factors_operation_classic,
                                k_factors_classic_rest)

                    # Load Mass delta NMC
                    m_delta = m_delta_nmc[variation[0], variation[2], variation[3], variation[4], variation[5], variation[6], 0]

                # Calulate Aging // LFP
                if variation[1] == 1:

                    # Calculate calendaric stress factor for rest days
                    result_classic_rest = calc_ageing_naumann(cell_vec[0], np.asarray([25]), 0.6, 0, calendaric=1)  # Assumption: BMS

                    # Load calculated stress factors
                    result_classic_operation = res_k_factors_operation_classic[variation[0], variation[1], variation[2],
                    variation[3], variation[4], variation[5], variation[6]]

                    # Calculate battery lifetime
                    res_ageing_values = calc_ageing(par, variation[1], vec_annual_mileage[variation_sensi[0]], vec_daily_range[variation[3]],
                                                    result_classic_operation,
                                                    result_classic_rest)

                    # Load Mass delta LFP
                    m_delta = m_delta_lfp[
                        variation[0], variation[2], variation[3], variation[4], variation[5], variation[6], 0]


                #Refactor to result arrays
                res_ageing[variation[0],variation[1], variation[2],
                variation[3], variation[4], variation[5], variation[6], variation_sensi[0]] = res_ageing_values[0:2]

                res_m_delta[variation[0], variation[1], variation[2],
                variation[3], variation[4], variation[5], variation[6], variation_sensi[0]] = m_delta


                # TCO Calculation -----------------------------------

                # Indexing
                payload_value = vec_payload[variation[0]]
                chemistry_value = vec_chemistry[variation[1]]
                bat_size_value = vec_bat_size[variation[2]]
                daily_range_value = vec_daily_range[variation[3]]
                charging_power_value = vec_p_charge_mcs[variation[4]]
                min_SOC_value = res_min_SOC[variation[0], variation[1], variation[2], variation[3], variation[4], variation[5], variation[6], 0]
                bat_life_years = res_ageing[variation[0], variation[1], variation[2], variation[3], variation[4], variation[5], variation[6], variation_sensi[0], 0]/365
                check_SOC = res_check_SOC[variation[0], variation[1], variation[2], variation[3], variation[4], variation[5], variation[6], 0]
                share_fc = res_share_energy[variation[0], variation[1], variation[2], variation[3], variation[4], variation[5], variation[6], 0]

                # Calulate TCO only for feasible parameter variations
                if check_SOC == 0:

                    # Calc GVW
                    gvw_value = ((bat_size_value / (par.gravimetric_energy_density[chemistry_value] / 1000)) /
                             par.c2p_grav[chemistry_value]) + par.m_wo_bat + payload_value

                    # Calc TCO
                    res_tco_values = calculate_tco(par, vec_annual_mileage[variation_sensi[0]], bat_size_value, vec_ene_sc[variation_sensi[2]], vec_ene_fc[variation_sensi[3]], energy_gvw_inter(gvw_value),
                                                                                bat_life_years, payload_value, chemistry_value, vec_cell_price[:,variation_sensi[1]], charging_power_value,share_fc )

                    # Refactor to result array
                    res_tco[variation[0], variation[1], variation[2],
                    variation[3], variation[4], variation[5], variation[6], variation_sensi[0], variation_sensi[1], variation_sensi[2], variation_sensi[3]] = res_tco_values



        np.save("saved_variables\\Cost\\res_ageing.npy", res_ageing)
        np.save("saved_variables\\Cost\\res_tco.npy", res_tco)
        np.save("saved_variables\\Cost\\vec_annual_mileage.npy", vec_annual_mileage)
        np.save("saved_variables\\Cost\\vec_cell_price.npy", vec_cell_price)
        np.save("saved_variables\\Cost\\vec_ene_sc.npy", vec_ene_sc)
        np.save("saved_variables\\Cost\\vec_ene_fc.npy", vec_ene_fc)
        np.save("saved_variables\\Cost\\res_m_delta.npy", res_m_delta)

    else:
        res_ageing = np.load("saved_variables\\Cost\\res_ageing.npy")
        res_tco = np.load("saved_variables\\Cost\\res_tco.npy")
        vec_annual_mileage = np.load("saved_variables\\Cost\\vec_annual_mileage.npy")
        vec_cell_price = np.load("saved_variables\\Cost\\vec_cell_price.npy")
        vec_ene_sc = np.load("saved_variables\\Cost\\vec_ene_sc.npy")
        vec_ene_fc = np.load("saved_variables\\Cost\\vec_ene_fc.npy")
        res_m_delta = np.load("saved_variables\\Cost\\res_m_delta.npy")

    # ----------------------------------------------------------------------------------------
    ### 5. Plots
    # ----------------------------------------------------------------------------------------

    # Figure 1 // Market Overview
    plot_market_overview()

    # Figure 3 // Power Profile example
    plot_power_profile()

    # Figure 5 // Feasibility
    plot_feasibility_stacked(res_check_SOC, res_min_SOC, max_payload, max_bat_size, vec_payload, vec_bat_size,
                        vec_chemistry, vec_daily_range, vec_p_charge_mcs, vec_temperature)

    # Figure 6 // Results
    plot_tco_km_paper(res_tco, max_payload, max_bat_size, vec_payload, vec_bat_size,
                      vec_chemistry, vec_daily_range, vec_p_charge_mcs, vec_temperature, vec_annual_mileage,
                      vec_cell_price)

    # Figure 6 // Sensitivity
    plot_tco_km_sensitivity(res_tco, res_check_SOC, res_check_gvw, max_bat_size, vec_payload, vec_bat_size,
                            vec_chemistry, vec_daily_range, vec_p_charge_mcs, vec_temperature, vec_annual_mileage,
                            vec_cell_price, vec_ene_sc, vec_ene_fc)

    # Figure 7 // Results
    plot_chem_advantage_paper(res_tco, res_min_SOC, max_payload, max_bat_size, vec_payload, vec_bat_size,
                              vec_daily_range, vec_p_charge_mcs, vec_temperature, vec_annual_mileage, res_m_delta, par)

    # Figure 7 // Sensitivity
    plot_chem_advantage_sensitivity(res_tco, res_min_SOC, max_payload, max_bat_size, vec_payload, vec_bat_size,
                                    vec_daily_range, vec_p_charge_mcs, vec_temperature, vec_annual_mileage, res_m_delta,
                                    par)

