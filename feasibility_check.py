"""
Created on Mar 3 10:46:50 2023

@author: schneiderFTM
"""

import numpy as np


def feasibility_lines(par, vec_bat_size, vec_chemistry):
    # Calculate maximal payload at each battery size and die maximal feasible battery size

    # Initialization
    maximal_payload = np.zeros((len(vec_chemistry), len(vec_bat_size)))
    maximal_bat_size = np.zeros((len(vec_chemistry)))

    #Loop over every chemistry
    for idx_chemistry in vec_chemistry:
        val_bat_weight = (vec_bat_size / (par.gravimetric_energy_density[idx_chemistry] / 1000)) / par.c2p_grav[idx_chemistry]
        maximal_payload[idx_chemistry] = par.max_gvw - (val_bat_weight + par.m_wo_bat)
        maximal_bat_size[idx_chemistry] = par.max_bat_volume * par.c2p_vol[idx_chemistry] *((par.volumetric_energy_density[idx_chemistry] / 1000))
    return maximal_payload, maximal_bat_size




def check_feasibility(par, payload, range, bat_size, chemistry, v_gvw_inter):

    # Calc battery volume and weight
    val_bat_weight = (bat_size / (par.gravimetric_energy_density[chemistry] / 1000)) / par.c2p_grav[chemistry]
    val_bat_volume = (bat_size / (par.volumetric_energy_density[chemistry] / 1000)) / par.c2p_vol[chemistry]

    # Calc GVW and Time for given Range based on average velocity
    val_gvw = val_bat_weight + par.m_wo_bat + payload
    time_for_range = range / v_gvw_inter(val_gvw)

    # Feasibility-Checks
    # 1. Check Exceeding GVW
    if val_gvw > par.max_gvw:
        check_gvw = 1
    else:
        check_gvw = 0

    # 2. Check Exceeding Volume
    if val_bat_volume > par.max_bat_volume:
        check_volume = 1
    else:
        check_volume = 0

    # 3. Check Exceeding Traveling-Time
    if time_for_range > par.max_travel_time:
        check_travel_time = 1
    else:
        check_travel_time = 0

    #4. Check Exceeding Volume AND GVW
    if val_gvw > par.max_gvw and val_bat_volume > par.max_bat_volume:
        check_gvw_volume = 1
    else:
        check_gvw_volume = 0

    return [check_gvw, check_volume, check_gvw_volume, check_travel_time], time_for_range, val_bat_weight




def calc_delta_weight(vec_payload, vec_bat_size, vec_daily_range,vec_p_charge_mcs,vec_p_charge_home, vec_temperature, res_check_volume, res_check_SOC ):
    # Caluclate mass delta between NMC and LFP Batteris

    # Use cases which are feasible for both NMC and LFP are compared directly
    # If only NMC is feasible, it is compared with the largest LFP which is feasible

    #Initialization
    m_delta_lfp = np.zeros((len(vec_payload), len(vec_bat_size), len(vec_daily_range),
                            len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), 1))
    m_delta_nmc = np.zeros((len(vec_payload), len(vec_bat_size), len(vec_daily_range),
                            len(vec_p_charge_mcs), len(vec_p_charge_home), len(vec_temperature), 1))

    # Loops over parameter variations
    for idx_temp in range(len(vec_temperature)):
        for idx_home in range(len(vec_p_charge_home)):
            for idx_mcs in range(len(vec_p_charge_mcs)):
                for idx_daily_range in range(len(vec_daily_range)):
                    flag_knick = 0
                    #Loop over every battery size
                    for idx_bat_size in range(len(vec_bat_size)):
                        flag_nmc = 0
                        flag_lfp = 0

                        # Volume Check
                        if res_check_volume[0, 1, idx_bat_size, 0] == 1 and flag_knick == 0:
                            flag_knick = 1
                            idx_knick_bat_size = idx_bat_size
                            if np.all(res_check_SOC[:, 1, idx_bat_size - 1, idx_daily_range, idx_mcs, 0, 0, 0] == 0):
                                idx_knick_payload = len(vec_payload)
                            else:
                                idx_knick_payload = min(np.where(
                                    res_check_SOC[:, 1, idx_bat_size - 1, idx_daily_range, idx_mcs, 0, 0, 0] == 1)[0])

                        for idx_payload in range(len(vec_payload)):

                            # SOC Check
                            check_nmc = res_check_SOC[
                                idx_payload, 0, idx_bat_size, idx_daily_range, idx_mcs, idx_home, idx_home, 0]

                            check_lfp = res_check_SOC[
                                idx_payload, 1, idx_bat_size, idx_daily_range, idx_mcs, idx_home, idx_home, 0]

                            if check_nmc == 1:
                                if flag_nmc == 0:
                                    flag_nmc = 1
                                    edge_nmc = idx_payload - 1

                            if check_lfp == 1:
                                if flag_lfp == 0:
                                    edge_lfp = idx_payload - 1  # Edge is last feasible Payload -> if check_SOC is 1 = first non feasible Payload
                                    flag_lfp = 1
                                if res_check_volume[idx_payload, 1, idx_bat_size, 0] == 1:
                                    if idx_payload - 1 > idx_knick_payload:
                                        edge_lfp = idx_knick_payload-1

                            if flag_nmc == 0 and (flag_lfp == 1 or flag_knick == 1):
                                if flag_knick == 1 and idx_payload > idx_knick_payload:
                                    m_delta_lfp[
                                        idx_payload, idx_bat_size, idx_daily_range, idx_mcs, idx_home, idx_home, 0] = abs(
                                        vec_payload[idx_payload] - vec_payload[edge_lfp])

                                if flag_lfp == 1 and flag_nmc == 0:
                                    m_delta_lfp[
                                        idx_payload, idx_bat_size, idx_daily_range, idx_mcs, idx_home, idx_home, 0] = abs(
                                        vec_payload[idx_payload] - vec_payload[edge_lfp])

                            elif flag_nmc == 1 and flag_lfp == 0:
                                m_delta_nmc[
                                    idx_payload, idx_bat_size, idx_daily_range, idx_mcs, idx_home, idx_home, 0] = abs(
                                    vec_payload[idx_payload] - vec_payload[edge_nmc])

    return m_delta_lfp, m_delta_nmc