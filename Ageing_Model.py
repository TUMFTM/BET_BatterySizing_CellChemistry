"""
Created on Fri Feb 7 8:42:33 2023

@author: schneiderFTM
"""

# Calculation of the battery lifetime for a given cell and usecase

import numpy as np
import random


def calc_ageing(par, chemistry, annual_mileage, daily_range, result_classic_operation, result_classic_rest):
    # Calculation of time until Battery Replacement is necessary

    # Array, which determines the Appearance of Operation and Rest Days according to Annual and Daily Range
    distribution_array = calc_operation_sequence(daily_range, annual_mileage)

    # Initialization
    Q_loss_classic = np.zeros(len(distribution_array))
    Q_loss_classic_cyc = np.zeros(len(distribution_array))
    Q_loss_classic_cal = np.zeros(len(distribution_array))
    Q_tot_age = 0
    operation_time = 1  # in day
    operation_cyc_classic = 0

    # NMC
    if chemistry == 0 and result_classic_operation.all() != 0:

        # Loop over every day in the service life of a truck
        for idx_day, day in enumerate(distribution_array):

            # Calc new Ageing State
            if day == 1:  # Operation Day

                operation_cyc_classic += 1
                Q_tot_age = result_classic_operation[0]*2

                # Equivalent throuput in order to reach current aging state
                Q_tot_eq = (Q_loss_classic_cyc[idx_day-1]/par.k_VW/(result_classic_operation[3]+result_classic_operation[4]))**2
                # Cyclic capacity loss
                Qloss_new_cyc = par.k_VW * (result_classic_operation[3] + result_classic_operation[4]) * np.sqrt(Q_tot_eq+ Q_tot_age)  # Scaled according to Teichert!

                # Equivalent Time in order to reach current aging state
                t_eq = (Q_loss_classic_cal[idx_day-1]/par.k_VW/result_classic_operation[1] / result_classic_operation[2])**(4/3)
                # Calednaric capacity loss
                Qloss_new_cal = par.k_VW * result_classic_operation[1] * result_classic_operation[2] * ((t_eq+operation_time) ** 0.75)


            if day == 0: # Rest Day
                # Equivalent Time in order to reach current aging state
                t_eq = (Q_loss_classic_cal[idx_day - 1] / par.k_VW / result_classic_rest[1] / result_classic_rest[
                    2]) ** (4 / 3)
                # Calednaric capacity loss
                Qloss_new_cal = par.k_VW * result_classic_rest[1] * result_classic_rest[2] * (
                            (t_eq + operation_time) ** 0.75)

            # Check of EoL Criterion
            if (Qloss_new_cal+Qloss_new_cyc) >= 0.2:
                res_operation_time_classic = idx_day
                break

            Q_loss_classic_cyc[idx_day] = Qloss_new_cyc
            Q_loss_classic_cal[idx_day] = Qloss_new_cal
            Q_loss_classic[idx_day] = Qloss_new_cal + Qloss_new_cyc


    # LFP
    elif chemistry == 1 and result_classic_operation.all() != 0:

        for idx_day, day in enumerate(distribution_array):

            # Calc new Ageing State
            if day == 1:  # Operation Day
                operation_cyc_classic += 1

                # Equivalent Time in order to reach current aging state
                t_eq = (Q_loss_classic_cal[idx_day - 1] / result_classic_operation[1] / result_classic_operation[2]) ** 2
                # Calednaric capacity loss
                Qloss_new_cal = result_classic_operation[1] * result_classic_operation[2] * np.sqrt(t_eq + operation_time*60*60*24)

                # Equivalent FEC in order to reach current aging state
                FEC_eq = (100*Q_loss_classic_cyc[idx_day - 1] /result_classic_operation[3]/result_classic_operation[4]) ** 2
                # Cyclic capacity loss
                Qloss_new_cyc = 0.01*(result_classic_operation[3] * result_classic_operation[4]) * np.sqrt(FEC_eq+result_classic_operation[0])

            if day == 0:  # Rest Day
                # Equivalent Time in order to reach current aging state
                t_eq = (Q_loss_classic_cal[idx_day - 1] / result_classic_rest[1] / result_classic_rest[2]) ** 2
                # Calednaric capacity loss
                Qloss_new_cal = result_classic_rest[1] * result_classic_rest[2] * np.sqrt(t_eq + operation_time*60*60*24)

            # Check of EoL Criterion
            if (Qloss_new_cal+Qloss_new_cyc) >= 0.2:
                res_operation_time_classic = idx_day
                break

            Q_loss_classic_cyc[idx_day] = Qloss_new_cyc
            Q_loss_classic_cal[idx_day] = Qloss_new_cal
            Q_loss_classic[idx_day] = Qloss_new_cal + Qloss_new_cyc

    else:

        res_operation_time_classic = 0
        operation_cyc_classic = 0




    return [res_operation_time_classic, operation_cyc_classic, Q_loss_classic, Q_loss_classic_cyc, Q_loss_classic_cal]

def calc_operation_sequence(daily_range, annual_mileage):
    # Determination of when a driving or a rest day occurs in the life of a truck according to the annual mileage

    # Define Array with the sequence of operation and rest-days distributed equaly over the year
    # Operation-Day =1
    # Rest-Day = 0

    operation_days = annual_mileage/daily_range
    operation_days_per_week = 7 * (operation_days/365)
    integer_part = int(operation_days_per_week)
    decimal_part = operation_days_per_week - integer_part

    operation_sequence = []
    for week in range(52*50):

        for day in range(7):
            operation_week = []
            for operation_day in range(integer_part):
                operation_week.append(1)

            operation_week.append(1 if random.random() < decimal_part else 0)

            for rest_day in range(7-len(operation_week)):
                operation_week.append(0)

        for day_idx in range(len(operation_week)):
            operation_sequence.append(operation_week[day_idx])

    return operation_sequence


def calc_ageing_naumann(cell, T_Cell, SOC, Crate, calendaric):
    # Calculation of the aging stress factors based on the Aging Model by Naumann et al.

    T_Cell = T_Cell + 273.15  # Model uses Kelvin

    # Calendar Ageing
    k_temp_Q_cal = np.mean(1.2571e-05 * np.exp((-17126 / 8.3145) * (1 / T_Cell - 1 / 298.15)))
    k_soc_Q_cal = np.mean(2.85750 * ((SOC - 0.5) ** 3) + 0.60225)


    # Cycling Aging
    if calendaric < 1:  # Check if only based on the scenario only calendaric aging occurs
        DODs, SOCavgs = rfcounting(SOC)  # Rainflow Counting Algorithm
        fec = sum(DODs)
        DODs = np.asarray(DODs)
        k_DOD_Q_cyc = sum((4.0253 * ((DODs - 0.6) ** 3) + 1.09230)*DODs)/fec
        Caging = abs(Crate)
        k_C_Q_cyc = sum(0.0971 + 0.063 * Caging)/len(Caging)
    else:
        k_DOD_Q_cyc = 0
        k_C_Q_cyc = 0
        fec = 0

    return [fec,  k_soc_Q_cal, k_temp_Q_cal, k_DOD_Q_cyc, k_C_Q_cyc]



def calc_ageing_schmalstieg(cell, T_Cell, SOC, Crate, calendaric):
    # Calculation of the aging stress factors based on the Aging Model by Schmalstieg et al.

    T_Cell = T_Cell + 273.15   # Model uses Kelvin
    T_Cell[T_Cell < 298.15] = 298.15 # Modification: Calendar aging is constant below 25°C, where the aging model is not valid

    # Calendar Aging
    Uocvs = cell.ocv_func(SOC)
    k_temp_Q_cal = np.mean(1e6 * np.exp(-6976/ T_Cell))
    k_soc_Q_cal = np.mean(7.543*Uocvs-23.75)

    # Cycling Aging
    if calendaric < 1:  # Check if only based on the scenario only calendaric aging occurs
        DODs, SOCavgs = rfcounting(SOC)
        Uavgs = cell.ocv_func(SOCavgs)

        Qtot = sum(DODs)*cell.Qnom
        DODs = np.asarray(DODs)
        k_DOD_Q_cyc = sum((4.081e-3*DODs) * DODs*cell.Qnom)/Qtot
        k_Uavgs_Q_cyc = sum((7.348e-3*(Uavgs-3.667)**2+7.6e-4) * DODs * cell.Qnom) / Qtot
    else:
        k_DOD_Q_cyc = 0
        k_Uavgs_Q_cyc = 0
        Qtot = 0

    return [Qtot, k_soc_Q_cal, k_temp_Q_cal, k_DOD_Q_cyc, k_Uavgs_Q_cyc]


def rfcounting(SOC):
    # Rainflow Counting algorithm to calculate weighted DOD factors

    # Remove Over-Charged phases
    SOC[SOC > SOC[0]] = SOC[0]

    # Remove constant SOC phases
    SOC_nozero = SOC[np.concatenate(([True], np.diff(SOC)!=0))]

    # Slice to start and end with the maximum SOC
    I = np.argmax(SOC_nozero)
    SOC_sorted = np.concatenate((SOC_nozero[I:], SOC_nozero[:I]))

    # Find extremas
    slope = np.diff(SOC_sorted)
    is_extremum = np.concatenate(([True], (slope[1:] * slope[:-1]) < 0, [True]))
    SOCs = SOC_sorted[is_extremum]

    # Find DODs
    DODs = []
    SOCavgs = []
    index = 1
    while len(SOCs) > index + 1:
        prevDOD = abs(SOCs[index] - SOCs[index-1])
        nextDOD = abs(SOCs[index] - SOCs[index+1])
        if nextDOD < prevDOD:
            index += 1
        else:
            DODs.append(prevDOD)
            SOCavgs.append((SOCs[index] + SOCs[index-1]) / 2)
            SOCs = np.delete(SOCs, [index-1, index])
            index = 1
    return DODs, SOCavgs








