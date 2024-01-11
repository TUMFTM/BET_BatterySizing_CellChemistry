"""
Created on Mar 11 11:42:56 2023

@author: schneiderFTM
"""

### Cost - Model which calculates the TCO of a diesel truck and a electric truck without the battery cost

import numpy as np
from scipy.interpolate import interp1d

# Cost parameters
r = 1.095  # interest rate [Assumption]



def residualValue(total_mileage):
    ### Return relative resale value
    ### Model adopted from Friedrich und Kleiner (2017)
    result = 0.951 * np.exp(-0.002 * total_mileage / 1000)  # Result in % of original NLP
    return result


def calculate_tco(par, annual_mileage, bat_size, c_ene_sc,c_ene_fc , con, bat_life_years, payload, chemistry,vec_cell_price, charging_power, share_FC):

    # -Calculate Energy Cost-

    # Cost Extrapolation based on existing charging costs
    c_elec_fc_power = [22, 400]
    c_elec_fc_cost = [par.c_sc_22, (c_ene_fc / 1.19)] # exkl. VAT
    fc_cost_inter = interp1d(c_elec_fc_power, c_elec_fc_cost, kind='linear', fill_value='extrapolate')
    c_elec = share_FC * fc_cost_inter(charging_power) + (1 - share_FC) * c_ene_sc
    delta_c_elec = fc_cost_inter(charging_power) -c_ene_sc

    # - Overall Cost Function -

    # Energy consumption costs (due weekly)
    c_ene = sum([c_elec * con * annual_mileage / 52 * r ** (-t / 52)
                 for t in range(0, 52 * par.servicelife)])

    # Determine required battery replacements
    n_replacements = int(par.servicelife/bat_life_years )  # number of replacements
    t_installation = [bat_life_years * n  for n in
                      range(0, n_replacements + 1)]  # time of replacments in years
    t_scrappage = t_installation[1:]  # time of scrappage in years

    # Battery investment costs
    c_bat_inv = [vec_cell_price[chemistry] * par.c2p_cost * bat_size * (r ** -t) for t in t_installation]

    # Scrappage value
    c_bat_scrappage = [par.bat_scrappage * vec_cell_price[chemistry] * par.c2p_cost * bat_size * r ** -t
                       for t in t_scrappage]  # scrappage values

    # Residual value
    bat_eol_soh = (
                1 - par.servicelife / bat_life_years % 1)  # remaining capacity at end of lifetime
    c_bat_residual = ((par.bat_scrappage + bat_eol_soh * (1 - par.bat_scrappage))
                      * vec_cell_price[chemistry] * par.c2p_cost * bat_size * r ** -par.servicelife)  # Add resale value of remaining capacity at EOL

    # Imputed interest
    t_operation = [bat_life_years] * n_replacements  # investment till scrappage
    t_operation += [par.servicelife - t_installation[-1]]  # investment till resale
    avg_bound_investment = [(c_purchase + c_scrap) / 2 for (c_purchase, c_scrap)
                            in zip(c_bat_inv[:-1], c_bat_scrappage)]  # investment till scrappage
    avg_bound_investment += [(c_bat_inv[-1] + c_bat_residual) / 2]  # investment till resale
    c_bat_imputed_interests = [invest * (r ** t - 1) for (invest, t)
                               in zip(avg_bound_investment, t_operation)]

    # Total battery costs
    c_bat = (sum(c_bat_inv)
             - sum(c_bat_scrappage)
             - c_bat_residual
             + sum(c_bat_imputed_interests))



    # Total costs
    c_tot = sum([c_ene, c_bat])

    if payload != 0:
        c_tot_per_km = c_tot / (par.servicelife * annual_mileage)
        c_tot_per_tkm = c_tot / ((par.servicelife * annual_mileage)* (payload/1000))
    else:
        c_tot_per_km = 0
        c_tot_per_tkm = 0

    return [c_tot, c_tot_per_km, c_tot_per_tkm,  c_ene, c_bat, sum(c_bat_inv), - sum(c_bat_scrappage), - c_bat_residual , sum(c_bat_imputed_interests), delta_c_elec]


def calc_sensitivity_preprocessing(vec_annual_mileage, vec_cell_price, vec_enesc, vec_ene_fc, fullfactorial):

    # Index Combinations
    vec_annual_mileage_idxcomb = np.arange(0, len(vec_annual_mileage), 1)
    vec_cellprice_idxcomb = np.arange(0, len(vec_cell_price[0,:]), 1)
    vec_ene_sc_idxcomb = np.arange(0, len(vec_enesc), 1)
    vec_ene_fc_idxcomb = np.arange(0, len(vec_ene_fc), 1)

    if fullfactorial == True:
        array_parameters = np.array(
        np.meshgrid(vec_annual_mileage_idxcomb, vec_cellprice_idxcomb, vec_ene_sc_idxcomb, vec_ene_fc_idxcomb)).T.reshape(-1, 4)


    if fullfactorial == False:
        array_parameters =np.zeros((3*len(vec_annual_mileage_idxcomb)+len(vec_ene_fc_idxcomb), 4))

        for idx_vec_annual in range(len(vec_annual_mileage_idxcomb)):
            array_parameters[idx_vec_annual] = [idx_vec_annual, 1,1,1]

        for idx_cell_price in range(len(vec_annual_mileage_idxcomb)):
            array_parameters[3+idx_cell_price] = [1, idx_cell_price,1,1]

        for idx_ene_sc in range(len(vec_annual_mileage_idxcomb)):
            array_parameters[idx_ene_sc + 6] = [1, 1, idx_ene_sc, 1]

        for idx_ene_fc_variation in range(len(vec_ene_fc_idxcomb)):
            array_parameters[idx_ene_fc_variation + 9] = [1, 1, 1, idx_ene_fc_variation]

        array_parameters = array_parameters.astype(int)

    return array_parameters


def calc_delta_tco(vec_bat_size, res_min_SOC, idx_daily_range, idx_mcs, vec_payload, res_tco, m_delta, par, idx_bat_start_vol, idx_annual, idx_cell_price_nmc, idx_cell_price_lfp, idx_ene_sc, idx_ene_fc, vec_annual_mileage, c_opp ):

    res_tco_delta = np.zeros((len(vec_payload), len(vec_bat_size)))
    # Initialize Feasibility Edges
    vec_feasible_edge = np.zeros((2, len(vec_bat_size)))
    vec_feasible_edge[:, :] = vec_payload[-1]
    flag_vol_constraint = 0
    flag_payload_at_constraint = 0
    idx_payload_at_constraint = len(vec_payload) - 1

    # ----------- Delta Calculation -------------
    # Iterate over every payload and battery size
    for idx_bat_size in range(len(vec_bat_size)):

        # Get Feasibility-Lines
        # NMC
        if np.any(res_min_SOC[:, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, 0] == 0):
            vec_feasible_edge[0, idx_bat_size] = vec_payload[
                min(np.where(res_min_SOC[:, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, 0] == 0)[0])]
        # LFP
        if np.any(res_min_SOC[:, 1, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, 0] == 0):
            vec_feasible_edge[1, idx_bat_size] = vec_payload[
                min(np.where(res_min_SOC[:, 1, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, 0] == 0)[0])]

        if idx_bat_size > idx_bat_start_vol:
            flag_vol_constraint = 1

        counter_lfp = 1
        counter_nmc = 1

        for idx_payload in range(len(vec_payload)):

            if flag_vol_constraint == 1:
                if flag_payload_at_constraint == 0:
                    if np.any(res_min_SOC[:, 1, idx_bat_size - 1, idx_daily_range, idx_mcs, 0, 0, 0] == 0):
                        idx_payload_at_constraint = min(np.where(
                            res_min_SOC[:, 1, idx_bat_size - 1, idx_daily_range, idx_mcs, 0, 0, 0] == 0)[0])
                    flag_payload_at_constraint = 1

            tco_nmc = res_tco[
                idx_payload, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price_nmc, idx_ene_sc, idx_ene_fc, 0]
            tco_lfp = res_tco[
                idx_payload, 1, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price_lfp, idx_ene_sc, idx_ene_fc, 0]
            m_nmc = m_delta[idx_payload, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, 0]
            m_lfp = m_delta[idx_payload, 1, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, 0]

            # Case: Idx Payload below both feasibility edges
            if (vec_payload[idx_payload] < vec_feasible_edge[0, idx_bat_size] and vec_payload[idx_payload] <
                vec_feasible_edge[1, idx_bat_size]) or (tco_nmc == 0 and tco_lfp == 0):
                tco_delta = tco_nmc - tco_lfp

            # Case: Idx Payload is over LFP and below NMC feasibility edges
            if vec_payload[idx_payload] < vec_feasible_edge[0, idx_bat_size] and vec_payload[idx_payload] >= \
                    vec_feasible_edge[1, idx_bat_size]:
                tco_lfp_delta = tco_nmc + sum(
                    [(c_opp* (m_lfp / 1000) * vec_annual_mileage[idx_annual]) / (par.r ** t) for t in
                     range(par.servicelife)])
                tco_delta = tco_nmc - tco_lfp_delta
                counter_lfp += 1

            # Case: Idx Payload is over NMC and below LFP feasibility edges
            if vec_payload[idx_payload] < vec_feasible_edge[1, idx_bat_size] and vec_payload[idx_payload] > \
                    vec_feasible_edge[0, idx_bat_size]:
                tco_nmc_delta = tco_lfp + sum(
                    [(c_opp * (m_nmc / 1000) * vec_annual_mileage[idx_annual]) / (par.r ** t) for
                     t in range(par.servicelife)])
                tco_delta = tco_nmc_delta - tco_lfp
                counter_nmc += 1

            # Case: battery size exceeds LFP volume constraint
            if flag_vol_constraint == 1 and tco_nmc != 0:
                # Case: Idx Payload is over last feasible Payload at Volume Constraint
                if vec_payload[idx_payload] >= vec_payload[idx_payload_at_constraint]:
                    tco_lfp_delta = res_tco[
                                        idx_payload_at_constraint - 1, 1, idx_bat_start_vol, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price_lfp, idx_ene_sc, idx_ene_fc, 0] + sum(
                        [(c_opp * (m_lfp / 1000) * vec_annual_mileage[idx_annual]) / (par.r ** t) for t
                         in range(par.servicelife)])
                    tco_delta = tco_nmc - tco_lfp_delta

                # Case: Idx Payload is not over last feasible Payload at Volume Constraint
                if vec_payload[idx_payload] < vec_payload[idx_payload_at_constraint]:
                    tco_delta = 1

            res_tco_delta[idx_payload, idx_bat_size] = tco_delta

    # Plot Delta for each Combination
    a = res_tco_delta[:, :]
    a[a == 0] = np.nan
    b = a.copy()
    b[b > 0] = 1
    b[b < 0] = -1

    # Extract Switch-Line:
    idx_switch = np.asarray([np.argmax(b[:, i] < 0) for i in range(len(b[0, :]))])
    vec_switch = vec_payload[idx_switch]
    vec_switch[vec_switch == 0] = np.nan
    vec_switch = vec_switch - (vec_payload[-1] - vec_payload[-2]) / 2

    return vec_switch