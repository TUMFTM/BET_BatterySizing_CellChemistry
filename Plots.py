"""
Created on Mar 1 13:22:26 2023

@author: schneiderFTM
"""

# Import functions and libaries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors
import matplotlib.ticker as tick
import os
import numpy as np
import pandas as pd
from calc_tco import calc_delta_tco


## General Parameterization

def cm2inch(value):
    return value / 2.54

#Two-Collum Width
col_width_two_col_doc_in_cm = 8.89

# Font Sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# Plot Sizing
plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['hatch.linewidth'] = 0.5

# TUM CI
color_TUM = []
color_TUM.append('#003359')  # DarkBlue
color_TUM.append('#0065BD')  # Blue
color_TUM.append('#E37222')  # Orange
color_TUM.append('#A2AD00')  # Green
color_TUM.append('#98C6EA')  # Green


def plot_market_overview():
    # Figure 1 // Marketing Overview

    # Plot Setup
    fig1, axes = plt.subplots(nrows=1,
                              ncols=1,
                              figsize=(
                                  cm2inch(1 * col_width_two_col_doc_in_cm),
                                  1 * cm2inch(5.5)))  # Width, height in inches
    legend_elements_1 = [Line2D([], [], color="white", marker='o', markersize=8, markerfacecolor=color_TUM[0]),
                         Line2D([], [], color="white", marker='o', markersize=8, markerfacecolor=color_TUM[2])]



    # Data Input
    data = pd.read_excel("inputs/Market_Overview/Truck_Market_Overview.xlsx")
    df = pd.DataFrame(data, columns=['Model', 'OEM', 'GVW [kg]', 'Chemistry', 'Installed energy (kWh)'])
    colors = np.where(np.logical_or(df["Chemistry"] == ('NMC'), df["Chemistry"] == ('NCA')), color_TUM[0], color_TUM[2])
    possible_payload = (42000-7500-df['GVW [kg]'])/1000


    # Plot
    plt.scatter(df['Installed energy (kWh)'], possible_payload,s = 30, c=colors, zorder=5)

    # add labels to all points
    for (xi, yi, model, oem) in zip(df['Installed energy (kWh)'], possible_payload, df['Model'], df['OEM'] ):
        if oem == "DAF":
            plt.text(xi + 10, yi+0.3, oem +' '+ model, va='center', ha='left')
        if oem == "MAN":
            plt.text(xi + 10, yi + 0.28, oem + ' ' + model, va='center', ha='left', zorder=5)
        if oem == "Volvo":
            plt.text(xi + 10, yi + 0.17, oem + ' ' + model, va='center', ha='left')
        if oem == "Scania":
            plt.text(xi + 10, yi -0.1, oem + ' ' + model, va='center', ha='left')
        if oem == "Daimler":
            plt.text(xi + 10, yi - 0.31, oem + ' ' + model, va='center', ha='left')

        if oem == "Nikola":
            plt.text(xi + 10, yi , oem + ' ' + model, va='center', ha='left')



    axes.set(xlim=[500, 900], ylim= [21.5, 25.5], ylabel= "Possible Payload in t", xlabel="Installed battery capacity in kWh")

    # Plot average payload diesel
    plt.axhline(y=(40000-7900-7500)/1000, color="#999999", linestyle='--', zorder=4)
    plt.text(675, (40000-7900-7500+200)/1000, 'Avg. payload diesel truck')

    # Legend
    fig1.legend(legend_elements_1,
                ['NMC/NCA Chemistry', 'LFP Chemistry'],
                loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol= 2)

    # Save and Show
    mypath_tex = os.path.abspath("Figures\\General")
    fig1.savefig(mypath_tex + "\\" + f"Market_Overview" + ".pdf", bbox_inches='tight')
    plt.show()

def plot_power_profile():
    # Figure 3 // Power Profile Example


    # Load Power Values
    p_power = np.load("saved_variables\\Additional_PlotData\\p_profile_example.npy")

    # Value Conversion
    h_vec = np.linspace(0, 24, len(p_power))
    p_power = p_power / 1000

    # Plot Setup
    fig1, axes = plt.subplots(nrows=1,
                              ncols=1,
                              figsize=(
                                  cm2inch(1 * col_width_two_col_doc_in_cm),
                                  1 * cm2inch(4)))  # Width, height in inches

    # Plot
    axes.plot(h_vec, p_power, linewidth=1, color=color_TUM[0])

    # Plot Limits
    axes.set(xlim=[0, 24], ylim=[-400, 800], ylabel="Power-Profile in kW", xlabel="Time in h")

    # Save and Show
    mypath_tex = os.path.abspath("Figures\\General ")
    fig1.savefig(mypath_tex + "\\" + f"Power_Profile" + ".pdf", bbox_inches='tight')
    plt.show()

def plot_feasibility_stacked(res_check_SOC, res_min_SOC, max_payload, max_bat_size, vec_payload, vec_bat_size, vec_chemistry, vec_daily_range, vec_p_charge_mcs, vec_temperature):
    # Figure 5 // Feasibility for different Charging Powers and Daily Ranges


    # For X different Daily Ranges


    # Plot Limits
    vec_payload = vec_payload/1000
    plot_payload_max = 26

    #Figure Definition
    fig1, axes = plt.subplots(nrows=2,
                              ncols=len(vec_daily_range),
                              figsize=(
                                  cm2inch(3 * col_width_two_col_doc_in_cm), len(vec_chemistry)*cm2inch(5.5)))  # Width, height in inches

    # Loop over calculated daily ranges (X- Axis)
    for idx_range in range(len(vec_daily_range)):

        ## NMC

        # Extrakt Feasibiity Lines for each charging power
        vec_feasible_nmc = np.zeros((len(vec_p_charge_mcs), len(vec_bat_size)))
        vec_feasible_nmc[:, :] = vec_payload[-1]
        for idx_mcs in range(len(vec_p_charge_mcs)):
            for idx_bat_size in range(len(vec_bat_size)):
                if np.any(res_min_SOC[:, 0, idx_bat_size, idx_range, idx_mcs, 0, 0, 0] == 0):
                    vec_feasible_nmc[idx_mcs, idx_bat_size] = vec_payload[
                        min(np.where(res_min_SOC[:, 0, idx_bat_size, idx_range, idx_mcs, 0, 0, 0] == 0)[0])]

        # Plot Feasiblilty Lines and shaded Areas
        axes[0, idx_range].plot(vec_bat_size,vec_feasible_nmc[0, :], color=color_TUM[0], linestyle="solid", zorder=4)
        axes[0, idx_range].fill_between(vec_bat_size, vec_feasible_nmc[0, :], 0, color=color_TUM[0], alpha=0.3,
                                       zorder=4)
        axes[0, idx_range].plot(vec_bat_size,vec_feasible_nmc[1, :], color=color_TUM[1], linestyle="dotted", zorder=3)
        axes[0, idx_range].fill_between(vec_bat_size, vec_feasible_nmc[1, :], 0, color=color_TUM[1], alpha=0.3,
                                        zorder=3)
        axes[0, idx_range].plot(vec_bat_size,vec_feasible_nmc[2, :], color=color_TUM[2], linestyle="dashed", zorder=2)
        axes[0, idx_range].fill_between(vec_bat_size, vec_feasible_nmc[2, :], 0, color=color_TUM[2], alpha=0.3,
                                        zorder=2)
        axes[0, idx_range].plot(vec_bat_size,vec_feasible_nmc[3, :], color=color_TUM[3], linestyle='dashdot', zorder=1)
        axes[0, idx_range].fill_between(vec_bat_size, vec_feasible_nmc[3, :], 0, color=color_TUM[3], alpha=0.3,
                                        zorder=1)

        # Plot Volume-Shade
        axes[0, idx_range].fill_betweenx(vec_payload, max_bat_size[0], vec_bat_size[-1], color="white", hatch="///", edgecolor=color_TUM[1], linewidth=1, zorder=5)

        # Plot GVW-Shade
        axes[0, idx_range].fill_between(vec_bat_size, max_payload[0]/1000, plot_payload_max,
                             color="white", hatch="///", edgecolor="#A2AD00", linewidth=1.0, zorder=5)


        ## LFP

        # Extrakt Feasibiity Lines for each charging power
        vec_feasible_lfp = np.zeros((len(vec_p_charge_mcs), len(vec_bat_size)))
        vec_feasible_lfp[:, :] = vec_payload[-1]
        for idx_mcs in range(len(vec_p_charge_mcs)):
            for idx_bat_size in range(len(vec_bat_size)):
                if np.any(res_min_SOC[:, 1, idx_bat_size, idx_range, idx_mcs, 0, 0, 0] == 0):
                    vec_feasible_lfp[idx_mcs, idx_bat_size] = vec_payload[
                        min(np.where(res_min_SOC[:, 1, idx_bat_size, idx_range, idx_mcs, 0, 0, 0] == 0)[0])]

        # Plot Feasiblilty Lines and shaded Areas
        axes[1,idx_range].plot(vec_bat_size, vec_feasible_lfp[0, :], color=color_TUM[0], linestyle="solid", zorder=4)
        axes[1,idx_range].fill_between(vec_bat_size, vec_feasible_lfp[0, :], 0, color=color_TUM[0], alpha=0.3, zorder=4)
        axes[1,idx_range].plot(vec_bat_size,vec_feasible_lfp[1, :], color=color_TUM[1], linestyle="dotted", zorder=3)
        axes[1,idx_range].fill_between(vec_bat_size, vec_feasible_lfp[1, :], 0, color=color_TUM[1], alpha=0.3,
                                       zorder=3)
        axes[1,idx_range].plot(vec_bat_size,vec_feasible_lfp[2, :], color=color_TUM[2], linestyle="dashed", zorder=2)
        axes[1,idx_range].fill_between(vec_bat_size, vec_feasible_lfp[2, :], 0, color=color_TUM[2], alpha=0.3,
                                        zorder=2)
        axes[1,idx_range].plot(vec_bat_size,vec_feasible_lfp[3, :], color=color_TUM[3], linestyle='dashdot', zorder=1)
        axes[1,idx_range].fill_between(vec_bat_size, vec_feasible_lfp[3, :], 0, color=color_TUM[3], alpha=0.3,
                                        zorder=1)

        # Plot Volume-Shade
        axes[1, idx_range].fill_betweenx(vec_payload, max_bat_size[1], vec_bat_size[-1], color="white", hatch="///", edgecolor=color_TUM[1], linewidth=1, zorder=5)

        # Plot GVW-Shade
        axes[1, idx_range].fill_between(vec_bat_size, max_payload[1]/1000, plot_payload_max,
                                      color="white", hatch="///", edgecolor="#A2AD00", linewidth=1, zorder=5)

        # Set Figure Limits
        axes[0, idx_range].xaxis.set_ticklabels([])
        axes[0, idx_range].set_ylim([0, plot_payload_max])
        axes[0, idx_range].set_xlim([vec_bat_size[0], 1000])
        axes[1, idx_range].set_ylim([0, plot_payload_max])
        axes[1, idx_range].set_xlim([vec_bat_size[0], 1000])

        #Titels and Labels
        axes[0, idx_range].set_title(f" Range: {vec_daily_range[idx_range]}km")
        axes[1, idx_range].set_xlabel("Battery capacity in kWh")
        axes[1, 0].set_ylabel("Payload in t")
        axes[0, 0].set_ylabel("Payload in t")
        pad = 5
        if idx_range == 0:
            axes[0, idx_range].annotate(f'NMC', xy=(0, 0.5),
                                    xytext=(-axes[0, idx_range].yaxis.labelpad - pad, 0),
                                    xycoords=axes[0, idx_range].yaxis.label, textcoords='offset points',
                                     ha='right', va='center')
            axes[1, idx_range].annotate(f'LFP', xy=(0, 0.5),
                                        xytext=(-axes[1,idx_range].yaxis.labelpad - pad, 0),
                                        xycoords=axes[1,idx_range].yaxis.label, textcoords='offset points', ha='right', va='center')

        if idx_range > 0:
            axes[0, idx_range].yaxis.set_ticklabels([])
            axes[1, idx_range].yaxis.set_ticklabels([])


        # Add additional Text to Plots
        if idx_range == 2:
            axes[0, idx_range].text(930, 1.05, 'Exceeding Veh. Package', rotation=90,
                         rotation_mode='anchor',
                         transform_rotates_text=True, zorder=6, size='small')

            axes[0, idx_range].text(825, 24, 'Exceeding GVW', rotation=0,
                         rotation_mode='anchor',
                         transform_rotates_text=True, zorder=6, size='small')
            axes[1, idx_range].text(780, 1.05, 'Exceeding  Veh. Package', rotation=90,
                                    rotation_mode='anchor',
                                    transform_rotates_text=True, zorder=6, size='small')

            axes[1, idx_range].text(800, 23, 'Exceeding GVW', rotation=0,
                                    rotation_mode='anchor',
                                    transform_rotates_text=True, zorder=6, size='small')

    # Save and Show
    mypath = os.path.abspath("Figures\\Feasibility")
    fig1.savefig(mypath + "\\" + f"Feasibility" + ".pdf", bbox_inches='tight')
    fig1.savefig(mypath + "\\" + f"Feasibility" + ".svg", bbox_inches='tight')
    plt.show()

def plot_tco_km_paper(res_tco, max_payload, max_bat_size, vec_payload, vec_bat_size,
             vec_chemistry, vec_daily_range, vec_p_charge_mcs, vec_temperature, vec_annual_mileage, vec_cell_price):
    # Figure 6 // TCO Results

    # Value Conversion
    plot_payload_max = 26
    vec_payload = vec_payload / 1000

    # Parameterization
    idx_range = 1  # 700km
    idx_annual_mileage = 1  # 130000
    idx_cell_price = 1  # no sensitivity
    idx_sc_price = 1  # no sensitivity
    idx_fc_price = 1  # no sensitivity
    idx_mcs_plot = [0, 1]
    idx_mcs_data = [0, 2]

    # Plot Setup
    fig1, axes = plt.subplots(nrows=len(idx_mcs_plot),
                              ncols=2,
                              figsize=(
                                  cm2inch(4/3 * col_width_two_col_doc_in_cm),
                                  len(idx_mcs_plot) * cm2inch(5.5)))  # Width, height in inches

    # Loop over chosen charging powers
    for idx_col, idx_mcs in zip(idx_mcs_plot, idx_mcs_data):

        # Replace zeros with nan
        a = res_tco[:, 0, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
        b = res_tco[:, 1, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
        a[a == 0] = np.nan
        b[b == 0] = np.nan

        ## NMC
        # Plot Contour
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",color_TUM[0]])
        Plot1 =  axes[0, idx_col].contourf(vec_bat_size, vec_payload, a,
                                          cmap=cmap, levels=np.linspace(0, 0.8, 9))

        # Plot Volume-Shade
        axes[0, idx_col].fill_betweenx(vec_payload[1:], max_bat_size[0], vec_bat_size[-1],
                                       color="white", hatch="///", edgecolor=color_TUM[1], linewidth=1, zorder=5)

        # Plot GVW-Shade
        axes[0, idx_col].fill_between(vec_bat_size, max_payload[0]/1000, plot_payload_max,
                                      color="white", hatch="///", edgecolor="#A2AD00", linewidth=1, zorder=5)


        ## LFP
        # Plot Contour
        Plot2 = axes[1, idx_col].contourf(vec_bat_size, vec_payload, b, 5,
                                          cmap=cmap, levels=np.linspace(0, 0.8, 9))

        # Plot Volume-Shade
        axes[1, idx_col].fill_betweenx(vec_payload[1:],max_bat_size[1], vec_bat_size[-1], color="white", hatch="///", edgecolor=color_TUM[1], linewidth=1, zorder=5)

        # Plot GVW-Shade
        axes[1, idx_col].fill_between(vec_bat_size,
                                      max_payload[1]/1000, plot_payload_max,
                                      color="white", hatch="///", edgecolor="#A2AD00", linewidth=1, zorder=5)

        # Set Plot limits
        axes[0, idx_col].set_ylim([vec_payload[1], 26])
        axes[1, idx_col].set_ylim([vec_payload[1], 26])
        axes[0, idx_col].set_xlim([vec_bat_size[0], 1000])
        axes[1, idx_col].set_xlim([vec_bat_size[0], 1000])

        # Add Lables and additional text
        axes[0, idx_col].xaxis.set_ticklabels([])
        axes[1, idx_col].set_xlabel("Battery capacity in kWh")
        pad = 5
        # Check if last row // Ticks and Axes Labels
        if idx_col == 0:
            axes[0, idx_col].set_ylabel("Payload in t")
            axes[1, idx_col].set_ylabel("Payload in t")
            axes[0, idx_col].set_title(f"{vec_p_charge_mcs[idx_mcs]} kW")
            axes[0, idx_col].annotate(f'NMC', xy=(0, 0.5),
                                      xytext=(-axes[0, idx_col].yaxis.labelpad - pad, 0),
                                      xycoords=axes[0, idx_col].yaxis.label, textcoords='offset points', ha='right', va='center')
            axes[1, idx_col].annotate(f'LFP', xy=(0, 0.5),
                                      xytext=(-axes[1, idx_col].yaxis.labelpad - pad, 0),
                                      xycoords=axes[1, idx_col].yaxis.label, textcoords='offset points', ha='right', va='center')
        elif idx_col == 1:
            axes[0, idx_col].set_title(f"{vec_p_charge_mcs[idx_mcs]} kW")
            fig1.subplots_adjust(right=0.87)
            cbar_ax = fig1.add_axes([0.9, 0.11, 0.02, 0.77])
            cbar1 = fig1.colorbar(Plot1, format=tick.FormatStrFormatter('%.2f'), cax=cbar_ax)
            cbar1.ax.set_ylabel('Battery Related Cost in €/km')
            cbar2 = fig1.colorbar(Plot2, format=tick.FormatStrFormatter('%.2f'), cax=cbar_ax)
            cbar2.ax.set_ylabel('Battery Related Cost in €/km')
            axes[0, idx_col].yaxis.set_ticklabels([])
            axes[1, idx_col].yaxis.set_ticklabels([])
            axes[1, idx_col].text(860, 2.6, 'Exceeding Veh. Package', rotation=90,
                                    rotation_mode='anchor',
                                    transform_rotates_text=True, zorder=6, size='small')

            axes[1, idx_col].text(730, 24, 'Exceeding GVW', rotation=0,
                                    rotation_mode='anchor',
                                    transform_rotates_text=True, zorder=6, size='small')

    # Save and Show
    mypath = os.path.abspath("Figures\\TCO")
    fig1.savefig(mypath + "\\" + f"TCO_Plot_Paper" + ".svg", bbox_inches='tight')
    fig1.savefig(mypath + "\\" + f"TCO_Plot_Paper" + ".pdf", bbox_inches='tight')
    plt.show()

def plot_tco_km_sensitivity(res_tco,res_check_soc, res_check_gvw, max_bat_size, vec_payload, vec_bat_size,
             vec_chemistry, vec_daily_range, vec_p_charge_mcs, vec_temperature, vec_annual_mileage, vec_cell_price, vec_ene_sc, vec_ene_fc):
    # Figure 6 // Sensitivity Analysis

    # Plot Setup
    fig1, axes = plt.subplots(nrows=2,
                              ncols=1,
                              figsize=(
                                  cm2inch((2/3) * col_width_two_col_doc_in_cm), 2 * cm2inch(5.5)))  # Width, height in inches


    # Plot Parameterization
    idx_range = 1 # 700km
    idx_annual_mileage = 1 # 130000
    idx_cell_price = 1 # no sensitivity
    idx_sc_price = 1  # no sensitivity
    idx_fc_price = 1  # no sensitivity
    idx_mcs = 2
    idx_payload = 5

    # Style definition
    linestyle_low = 'dashed'
    linestyle_high = ':'

    # Sensitivity Annual Mileage
    for idx_annual_mileage in range(len(vec_annual_mileage)):
        if idx_annual_mileage != 1:
            # Indexing
            a = res_tco[idx_payload, 0, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
            b = res_tco[idx_payload, 1, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
            a[a == 0] = np.nan
            b[b == 0] = np.nan

            # Plot Lines
            if idx_annual_mileage == 0:
                axes[0].plot(vec_bat_size, a, label=f'AM{idx_annual_mileage}', color=color_TUM[0], linestyle = linestyle_low)
                axes[1].plot(vec_bat_size,b, label=f'AM{idx_annual_mileage}', color=color_TUM[0], linestyle = linestyle_low)
            if idx_annual_mileage == 2:
                axes[0].plot(vec_bat_size, a, label=f'AM{idx_annual_mileage}', color=color_TUM[0], linestyle = linestyle_high)
                axes[1].plot(vec_bat_size,b, label=f'AM{idx_annual_mileage}', color=color_TUM[0], linestyle = linestyle_high)
        else:
            a = res_tco[idx_payload, 0, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,
                idx_fc_price, 1]
            b = res_tco[idx_payload, 1, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,
                idx_fc_price, 1]

            # Baseline
            a[a == 0] = np.nan
            b[b == 0] = np.nan
            axes[0].plot(vec_bat_size, a, color="black", linestyle = 'solid')
            axes[1].plot(vec_bat_size, b, color="black", linestyle = 'solid')


    idx_annual_mileage = 1  # 130000
    # Cell Price
    for idx_cell_price in [0, 2]:
        # Plot Lines
        a = res_tco[idx_payload, 0, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
        b = res_tco[idx_payload, 1, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
        a[a == 0] = np.nan
        b[b == 0] = np.nan
        if idx_cell_price == 0:
            axes[0].plot(vec_bat_size, a, label=f'CellP{idx_cell_price}', color=color_TUM[1], linestyle=linestyle_low)
            axes[1].plot(vec_bat_size, b, label=f'CellP{idx_cell_price}', color=color_TUM[1], linestyle=linestyle_low)
        if idx_cell_price == 2:
            axes[0].plot(vec_bat_size, a, label=f'CellP{idx_cell_price}', color=color_TUM[1], linestyle=linestyle_high)
            axes[1].plot(vec_bat_size, b, label=f'CellP{idx_cell_price}', color=color_TUM[1], linestyle=linestyle_high)

    idx_cell_price = 1  # no sensitivity
    # Slow Charging Price
    for idx_sc_price in [0, 2]:
        # Plot Lines
        a = res_tco[idx_payload, 0, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
        b = res_tco[idx_payload, 1, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
        a[a == 0] = np.nan
        b[b == 0] = np.nan
        if idx_sc_price == 0:
            axes[0].plot(vec_bat_size,a, label=f'ChargeP{idx_sc_price}', color=color_TUM[2], linestyle=linestyle_low)
            axes[1].plot(vec_bat_size,b, label=f'ChargeP{idx_sc_price}', color=color_TUM[2], linestyle=linestyle_low)
        if idx_sc_price == 2:
            axes[0].plot(vec_bat_size,a, label=f'ChargeP{idx_sc_price}', color=color_TUM[2], linestyle=linestyle_high)
            axes[1].plot(vec_bat_size,b, label=f'ChargeP{idx_sc_price}', color=color_TUM[2], linestyle=linestyle_high)

    idx_sc_price = 1
    # Fast Charging Price
    for idx_fc_price in [0, 2]:
        # Plot Lines
        a = res_tco[idx_payload, 0, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
        b = res_tco[idx_payload, 1, :, idx_range, idx_mcs, 0, 0, idx_annual_mileage, idx_cell_price, idx_sc_price,idx_fc_price, 1]
        a[a == 0] = np.nan
        b[b == 0] = np.nan
        if idx_fc_price == 0:
            axes[0].plot(vec_bat_size, a, label=f'ChargeP{idx_fc_price}', color=color_TUM[3], linestyle=linestyle_low)
            axes[1].plot(vec_bat_size, b, label=f'ChargeP{idx_fc_price}', color=color_TUM[3], linestyle=linestyle_low)
        if idx_fc_price == 2:
            axes[0].plot(vec_bat_size, a, label=f'ChargeP{idx_fc_price}', color=color_TUM[3], linestyle=linestyle_high)
            axes[1].plot(vec_bat_size, b, label=f'ChargeP{idx_fc_price}', color=color_TUM[3], linestyle=linestyle_high)


    # Plot Volume-Shade
    axes[0].fill_betweenx([0.4, 0.8], max_bat_size[0], 1000, color="white",
                                     hatch="///", edgecolor=color_TUM[1], linewidth=1, zorder=5)
    axes[1].fill_betweenx([0.4, 0.8], max_bat_size[1], 1000, color="white",
                          hatch="///", edgecolor=color_TUM[1], linewidth=1, zorder=5)

    # Set Plot Limits and add Labels
    axes[0].xaxis.set_ticklabels([])
    axes[0].set_xlim([vec_bat_size[0], 1000])
    axes[1].set_xlim([vec_bat_size[0], 1000])
    axes[0].set_ylim([0.4, 0.8])
    axes[1].set_ylim([0.4, 0.8])
    axes[1].set_xlabel("Battery capacity in kWh")
    axes[0].set_ylabel("BRC per km in €/km")
    axes[1].set_ylabel("BRC per km in €/km")
    axes[0].set_title(f"{vec_p_charge_mcs[idx_mcs]} kW // {vec_payload[idx_payload]/1000} t")
    axes[1].text(950, 0.45, 'Exceeding Veh. Package', rotation=90,
                          rotation_mode='anchor',
                          transform_rotates_text=True, zorder=6, size='small')

    # Save and Show
    mypath = os.path.abspath("Figures\\TCO")
    fig1.savefig(
        mypath + "\\" + f"Sensitivity{vec_payload[idx_payload]}" + ".svg",
        bbox_inches='tight')
    fig1.savefig(
        mypath + "\\" + f"Sensitivity{vec_payload[idx_payload]}" + ".pdf",
        bbox_inches='tight')
    plt.show()





def plot_chem_advantage_paper(res_tco, res_min_SOC, max_payload, max_bat_size, vec_payload, vec_bat_size,
                              vec_daily_range, vec_p_charge_mcs, vec_temperature, vec_annual_mileage, m_delta, par):
    # Figure 7 // Chemistry Advantage

    # Initialization
    res_tco_delta = np.zeros(
            (len(vec_payload), len(vec_bat_size), len(vec_daily_range),
             len(vec_p_charge_mcs), 1, 1, len(vec_annual_mileage),3 ,3, 1))

    # Indexing
    vec_payload = vec_payload/1000
    plot_payload_max = 26
    idx_annual = 1
    idx_mcs_data = [0, 2]
    idx_mcs_plot = [0, 1]
    vec_range_data = [0, 2]
    vec_range_plot = [0, 1]
    idx_cell_price = 1
    idx_ene_sc = 1
    idx_ene_fc = 1

    # Plot Setup
    fig1, axes = plt.subplots(nrows=2,
                              ncols=2,
                              figsize=(
                                  cm2inch((4/3) * col_width_two_col_doc_in_cm),
                                  2 * cm2inch(5.5)))  # Width, height in inches

    # Loop over charging Power and Daily Range
    for idx_row, idx_mcs in zip(idx_mcs_plot, idx_mcs_data):
        for idx_col, idx_daily_range in zip(vec_range_plot, vec_range_data):

            # Plot Volume-Shade // NMC
            axes[idx_row, idx_col].fill_betweenx(vec_payload, max_bat_size[0], vec_bat_size[-1], color="white",
                                                 hatch="///", edgecolor=color_TUM[1], linewidth=1, zorder=5)

            # Plot GVW-Shade // NMC
            axes[idx_row, idx_col].fill_between(vec_bat_size,
                                            max_payload[0]/1000, plot_payload_max,
                                 color="white", hatch="///", edgecolor="#A2AD00", linewidth=1, zorder=5)

            # Plot Volume-Line // LFP
            axes[idx_row, idx_col].plot([max_bat_size[1], max_bat_size[1]],
                               [vec_payload[0], plot_payload_max], color=color_TUM[1], linestyle=":", zorder=1)

            # Plot GVW-Line // LFP
            axes[idx_row, idx_col].plot(vec_bat_size, max_payload[1]/1000,
                               color='#A2AD00', linestyle=':', zorder=5)

            # Initialize Feasibility Edges
            vec_feasible_edge = np.zeros((2, len(vec_bat_size)))
            vec_feasible_edge[:, :] = vec_payload[-1]
            flag_vol_constraint = 0
            flag_payload_at_constraint = 0
            idx_payload_at_constraint = len(vec_payload) - 1
            idx_bat_start_vol = min(np.where(vec_bat_size > max_bat_size[1])[0])

            #----------- Delta Calculation -------------
            #Iterate over every payload and battery size
            for idx_bat_size in range(len(vec_bat_size)):

                #Get Feasibility-Lines
                #NMC
                if np.any(res_min_SOC[:, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, 0] == 0):
                    vec_feasible_edge[0, idx_bat_size] = vec_payload[
                        min(np.where(res_min_SOC[:, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, 0] == 0)[0])]
                #LFP
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
                            if np.any(res_min_SOC[:, 1, idx_bat_size-1, idx_daily_range, idx_mcs, 0, 0, 0] == 0):
                                idx_payload_at_constraint = min(np.where(
                                        res_min_SOC[:, 1, idx_bat_size-1, idx_daily_range, idx_mcs, 0, 0, 0] == 0)[0])
                            flag_payload_at_constraint = 1

                    tco_nmc = res_tco[idx_payload, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, idx_ene_fc, 0]
                    tco_lfp = res_tco[idx_payload, 1, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, idx_ene_fc, 0]
                    m_nmc = m_delta[idx_payload, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, 0]
                    m_lfp = m_delta[idx_payload, 1, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, 0]

                    # Case: Idx Payload below both feasibility edges
                    if (vec_payload[idx_payload] < vec_feasible_edge[0, idx_bat_size] and vec_payload[idx_payload] < vec_feasible_edge[1, idx_bat_size]) or (tco_nmc == 0 and tco_lfp ==0):
                        tco_delta = tco_nmc-tco_lfp

                    # Case: Idx Payload is over LFP and below NMC feasibility edges
                    if vec_payload[idx_payload] < vec_feasible_edge[0, idx_bat_size] and vec_payload[idx_payload] >= vec_feasible_edge[1, idx_bat_size]:
                        tco_lfp_delta = tco_nmc + sum([(par.opp_cost*(m_lfp/1000)*vec_annual_mileage[idx_annual]) / (par.r ** t) for t in range(par.servicelife)])
                        tco_delta = tco_nmc - tco_lfp_delta
                        counter_lfp +=1

                    # Case: Idx Payload is over NMC and below LFP feasibility edges
                    if vec_payload[idx_payload] < vec_feasible_edge[1, idx_bat_size] and vec_payload[idx_payload] > vec_feasible_edge[0, idx_bat_size]:
                        tco_nmc_delta = tco_lfp + sum([(par.opp_cost * (m_nmc / 1000) * vec_annual_mileage[idx_annual]) / (par.r ** t) for
                             t in range(par.servicelife)])
                        tco_delta = tco_nmc_delta - tco_lfp
                        counter_nmc +=1

                    # Case: battery size exceeds LFP volume constraint
                    if flag_vol_constraint == 1 and tco_nmc != 0:
                        # Case: Idx Payload is over last feasible Payload at Volume Constraint
                        if vec_payload[idx_payload] >= vec_payload[idx_payload_at_constraint]:
                            tco_lfp_delta = res_tco[idx_payload_at_constraint-1, 1, idx_bat_start_vol, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, idx_ene_fc, 0] + sum(
                                [(par.opp_cost * (m_lfp / 1000) * vec_annual_mileage[idx_annual]) / (par.r ** t) for t in range(par.servicelife)])
                            tco_delta = tco_nmc - tco_lfp_delta

                        # Case: Idx Payload is not over last feasible Payload at Volume Constraint
                        if vec_payload[idx_payload] < vec_payload[idx_payload_at_constraint]:
                            tco_delta = 1

                    res_tco_delta[idx_payload, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, 0] = tco_delta


            # Plot Delta for each Combination
            a = res_tco_delta[:, :, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, 0]
            a[a == 0] = np.nan
            b=a.copy()
            # Simplify Result Array // + or -
            b[b>0] = 1
            b[b<0] = -1

            # Plot
            Plot1 = axes[idx_row, idx_col].contourf(vec_bat_size, vec_payload, b,
                                              cmap=matplotlib.colors.ListedColormap(["#0A2D57","#e37222"]),
                                              levels=[-1.1, 0, 1.1])

            # Check if first row // Column titles
            if idx_row == 0:
                axes[idx_row, idx_col].set_title(f" Daily Range {vec_daily_range[idx_daily_range]}km")
                axes[idx_row, idx_col].xaxis.set_ticklabels([])

            # Check if last row // Ticks and Axes Labels
            if idx_row == 1:
                axes[idx_row, idx_col].set_xlabel("Battery capacity in kWh")

            # Check if first column // Ticks and Axes Labels
            if idx_col == 0:
                axes[idx_row, idx_col].set_ylabel("Payload in t")
                pad = 5
                axes[idx_row, idx_col].annotate(f'{vec_p_charge_mcs[idx_mcs]}kW', xy=(0, 0.5), rotation=90,
                                          xytext=(-axes[idx_row, idx_col].yaxis.labelpad - pad, 0),
                                          xycoords=axes[idx_row, idx_col].yaxis.label, textcoords='offset points',
                                           ha='right', va='center')

            # Check if second column // Ticks and Axes Labels
            if idx_col == 1:
                axes[idx_row, idx_col].yaxis.set_ticklabels([])
                axes[1, idx_col].arrow(max_bat_size[1], 6, 70,0, width=0.2, facecolor = "#999999", edgecolor = "#475058", head_length=8)
                axes[1, idx_col].text(700, 8,"LFP \nnot feasible", fontsize= SMALL_SIZE, color = "#475058")

            # Set Plot Limits
            axes[idx_row, idx_col].set_ylim([0, 26])
            axes[idx_row, idx_col].set_xlim([vec_bat_size[0], 1000])


    # Save and Show
    mypath = os.path.abspath("Figures\\Chemistry_Advantage")
    fig1.savefig(
        mypath + "\\" + f"Chem_Advantage" + ".svg",
        bbox_inches='tight')
    fig1.savefig(
        mypath + "\\" + f"Chem_Advantage" + ".pdf",
        bbox_inches='tight')
    plt.show()

def plot_chem_advantage_sensitivity(res_tco, res_min_SOC,  max_payload, max_bat_size, vec_payload, vec_bat_size, vec_daily_range, vec_p_charge_mcs, vec_temperature, vec_annual_mileage, m_delta, par):
    # Figure 7 // Chemistry Advantage Sensitivity

    # Initialization
    res_tco_delta = np.zeros(
        (len(vec_payload), len(vec_bat_size), len(vec_daily_range),
         len(vec_p_charge_mcs), 1, 1, len(vec_annual_mileage), 3, 3, 1))

    # Indexing
    vec_payload = vec_payload / 1000
    plot_payload_max = 26
    idx_annual = 1
    idx_mcs_data = [0, 2]
    idx_mcs_plot = [0, 1]
    idx_daily_range = 2
    idx_cell_price = 1
    idx_ene_sc = 1
    idx_ene_fc = 1

    # Plot Setup
    fig1, axes = plt.subplots(nrows=2,
                              ncols=1,
                              figsize=(
                                  cm2inch((2/ 3) * col_width_two_col_doc_in_cm),
                                  2 * cm2inch(5.5)))  # Width, height in inches

    # Loop over rows
    for idx_row, idx_mcs in zip(idx_mcs_plot, idx_mcs_data):

        # Plot Volume-Shade // NMC
        axes[idx_row].fill_betweenx(vec_payload, max_bat_size[0], vec_bat_size[-1],
                                             color="white", hatch="///", edgecolor=color_TUM[1], linewidth=1,
                                             zorder=5)

        # Plot GVW-Shade // NMC
        axes[idx_row].fill_between(vec_bat_size,
                                            max_payload[0]/1000, plot_payload_max,
                                            color="white", hatch="///", edgecolor="#A2AD00", linewidth=1, zorder=5)

        # Plot Volume-Line // LFP
        axes[idx_row].plot([max_bat_size[1], max_bat_size[1]],
                                    [vec_payload[0], plot_payload_max], color=color_TUM[1], linestyle=":", zorder=1)

        # Plot GVW-Line // LFP
        axes[idx_row].plot(vec_bat_size, max_payload[1]/1000,
                                    color='#A2AD00', linestyle=':', zorder=5)

        # Initialize Feasibility Edges
        vec_feasible_edge = np.zeros((2, len(vec_bat_size)))
        vec_feasible_edge[:, :] = vec_payload[-1]
        flag_vol_constraint = 0
        flag_payload_at_constraint = 0
        idx_payload_at_constraint = len(vec_payload) - 1

        idx_bat_start_vol = min(np.where(vec_bat_size>max_bat_size[1])[0])

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
                    idx_payload, 0, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, idx_ene_fc, 0]
                tco_lfp = res_tco[
                    idx_payload, 1, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, idx_ene_fc, 0]
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
                        [(par.opp_cost * (m_lfp / 1000) * vec_annual_mileage[idx_annual]) / (par.r ** t) for t in
                         range(par.servicelife)])
                    tco_delta = tco_nmc - tco_lfp_delta
                    counter_lfp += 1

                # Case: Idx Payload is over NMC and below LFP feasibility edges
                if vec_payload[idx_payload] < vec_feasible_edge[1, idx_bat_size] and vec_payload[idx_payload] > \
                        vec_feasible_edge[0, idx_bat_size]:
                    tco_nmc_delta = tco_lfp + sum(
                        [(par.opp_cost * (m_nmc / 1000) * vec_annual_mileage[idx_annual]) / (par.r ** t) for
                         t in range(par.servicelife)])
                    tco_delta = tco_nmc_delta - tco_lfp
                    counter_nmc += 1

                # Case: battery size exceeds LFP volume constraint
                if flag_vol_constraint == 1 and tco_nmc != 0:
                    # Case: Idx Payload is over last feasible Payload at Volume Constraint
                    if vec_payload[idx_payload] >= vec_payload[idx_payload_at_constraint]:
                        tco_lfp_delta = res_tco[
                                            idx_payload_at_constraint - 1, 1, idx_bat_start_vol, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, idx_ene_fc, 0] + sum(
                            [(par.opp_cost * (m_lfp / 1000) * vec_annual_mileage[idx_annual]) / (par.r ** t) for t
                             in range(par.servicelife)])
                        tco_delta = tco_nmc - tco_lfp_delta

                    # Case: Idx Payload is not over last feasible Payload at Volume Constraint
                    if vec_payload[idx_payload] < vec_payload[idx_payload_at_constraint]:
                        tco_delta = 1

                res_tco_delta[
                    idx_payload, idx_bat_size, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, 0] = tco_delta

        # Plot Delta for each Combination
        a = res_tco_delta[:, :, idx_daily_range, idx_mcs, 0, 0, idx_annual, idx_cell_price, idx_ene_sc, 0]
        a[a == 0] = np.nan
        b = a.copy()
        b[b > 0] = 1
        b[b < 0] = -1

        # Plot
        Plot1 = axes[idx_row].contourf(vec_bat_size, vec_payload, b,
                                                cmap=matplotlib.colors.ListedColormap(
                                                    ["#0A2D57", "#e37222"]),
                                                levels=[-1.001,  0,  1.001], alpha = 0.5)

        # Extract Switch-Line:
        idx_switch = np.asarray([np.argmax(b[:, i] < 0) for i in range(len(b[0, :]))])
        vec_switch = vec_payload[idx_switch]
        vec_switch[vec_switch == 0] = np.nan
        vec_switch = vec_switch - (vec_payload[-1]-vec_payload[-2])/2
        axes[idx_row].plot(vec_bat_size, vec_switch, color = "#0A2D57", alpha = 0.5)

        # Plot Shade, which indicates the area which exceeds lfp volume constraint and
        # idx_bat_start_vol

        ### ------------------------- SENSITIVITY ------------------------------------

        # Sensitivity Parameterization
        vec_c_opp = [0.8*0.0358, 1.2*0.0358]
        vec_idx_cell_price_nmc = [0, 2]
        idx_cell_price_lfp = 1

        # Opportunity Cost
        for idx_opp, c_opp in enumerate(vec_c_opp):
            idx_cell_price_nmc = 1

            vec_switch = calc_delta_tco(vec_bat_size, res_min_SOC, idx_daily_range, idx_mcs, vec_payload, res_tco, m_delta, par,
                       idx_bat_start_vol, idx_annual, idx_cell_price_nmc, idx_cell_price_lfp, idx_ene_sc, idx_ene_fc,
                       vec_annual_mileage, c_opp)

            if idx_opp == 0:
                axes[idx_row].plot(vec_bat_size, vec_switch, linestyle = ":", color = "#0A2D57")
            if idx_opp == 1:
                axes[idx_row].plot(vec_bat_size, vec_switch, linestyle="--", color="#0A2D57")

        # NMC Cell Price
        for idx_sensi, idx_cell_price_nmc in enumerate(vec_idx_cell_price_nmc):
            c_opp = 0.0358

            vec_switch = calc_delta_tco(vec_bat_size, res_min_SOC, idx_daily_range, idx_mcs, vec_payload,
                                        res_tco, m_delta, par,
                                        idx_bat_start_vol, idx_annual, idx_cell_price_nmc, idx_cell_price_lfp,
                                        idx_ene_sc, idx_ene_fc,
                                        vec_annual_mileage, c_opp)

            if idx_sensi == 0:
                if idx_row == 1:
                    vec_switch[4] = np.nan
                axes[idx_row].plot(vec_bat_size, vec_switch, linestyle=":", color=color_TUM[2])
            if idx_sensi == 1:
                if idx_row == 0:
                    vec_switch[5] = 7
                axes[idx_row].plot(vec_bat_size, vec_switch, linestyle="--", color=color_TUM[2])

        # Title and labels
        # Check if first row // Collumn titles
        if idx_row == 0:
            axes[idx_row].xaxis.set_ticklabels([])

        # Check if last row // Ticks and Axes Labels
        if idx_row == 1:
            axes[idx_row].set_xlabel("Battery capacity in kWh")
        axes[idx_row].set_ylabel("Payload in t")
        axes[0].set_title("700km")
        axes[1].arrow(max_bat_size[1], 6, 70, 0, width=0.2, facecolor="#999999", edgecolor="#475058",
                                   head_length=8)
        axes[1].text(700, 8, "LFP \nnot feasible", fontsize=SMALL_SIZE, color="#475058")

        # Set Plot limit
        axes[idx_row].set_ylim([0, 26])
        axes[idx_row].set_xlim([vec_bat_size[0], 1000])

    mypath = os.path.abspath("Figures\\Chemistry_Advantage")
    fig1.savefig(
        mypath + "\\" + f"Chem_Advantage_Sensitivity" + ".svg",
        bbox_inches='tight')
    fig1.savefig(
        mypath + "\\" + f"Chem_Advantage_Sensitivity" + ".pdf",
        bbox_inches='tight')

    plt.show()