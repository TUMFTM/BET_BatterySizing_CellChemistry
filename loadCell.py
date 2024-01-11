"""
Created on Mar 11 11:12:52 2023

@author: schneiderFTM
"""
# Loading cell data according to selected Cell

# Import Function
from battery_data import bat_nau_setting
from battery_data import bat_ss_setting

#Class Initialization
class Cell:
    pass

# Load Cell data according to chosen chemistry
def load_cell_data(Chemistry):
    if Chemistry == 0:
        # Naumann et al model
        cell = Cell()
        cell = bat_ss_setting(cell)
    elif Chemistry == 1:
        # Schmalstieg et al model
        cell = Cell()
        cell = bat_nau_setting(cell)
    return cell
