"""
Created on Mar 17 18:42:33 2023

@author: schneiderFTM
"""

# Calculate energy consumption based on vehicle weight for multiple weights


from pandas import read_excel, read_csv
from lds_simulation import Simulation
import numpy as np


# Import driving cycle
drivingcycle = read_csv("inputs/Mission Profiles/LongHaul.vdri")  # Load long haul driving cycle

def energy_consumption_driving(par, gvw):

    sim = Simulation(drivingcycle)  # LDS
    lds_result = [sim.run(par, gvw)]  # Simulation with given payload
    v_avg = lds_result[0][1]  # km/h
    con_per_km = lds_result[0][0] + (par.p_aux/1000)/v_avg # kW/v_avg  # kW/km LDS-Results for Vecto Cylce @given Payload
    p_dem = lds_result[0][5]
    t = lds_result[0][3]
    return con_per_km, v_avg, p_dem, t

def calc_energy_consumption_depending_gvw(par, gvw_vec):

    # Initialization
    energy_gvw=[]
    v_gvw = []
    p_dem_gvw = []
    t_gvw = []

    # Loop over every gross vehicle weight in vector
    for gvw in gvw_vec:
        con, v, p_dem, t = energy_consumption_driving(par, gvw)
        energy_gvw.append(con)
        v_gvw.append(v)
        p_dem_gvw.append(p_dem)
        t_gvw.append(t)

    return energy_gvw, v_gvw, p_dem_gvw, t_gvw
