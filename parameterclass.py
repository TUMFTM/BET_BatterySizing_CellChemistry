
class ParameterClass():

    def __init__(self):

        #General Parameters
        self.connection_time = 5 # in min // Connection/Payment Time which is lost in each charging event
        self.m_wo_bat = 13765 # in kg //  Mass of BET without Battery
        self.max_gvw = 42000 # in kg // Maximal allowed Gross Vehicle Weight of a BET
        self.max_travel_time = 9 # in h // Maximal allowed driving time in this analysis
        self.max_bat_volume = 3250  # in l // Maximal space for the battery in a Tractor-truck

        #Truck Parameters
        self.motor_power = 352.3675316051114e3  # Max power in W [EEA average]
        self.fr = 0.005  # Rolling friction coefficient [EEA average]
        self.cd_a = 5.2
        self.p_aux = 2.3e3
        self.eta = 0.85  # overall powertrain efficiency [Earl 2018]

        #Cell Parameters
        #Indexes: 0 = NMC // 1 = LFP
        self.gravimetric_energy_density = [273, 176] #in Wh/kg # Wassiliadis et al
        self.volumetric_energy_density = [685, 376]  # in Wh/l # Wassiliadis et al
        self.c2p_grav = [0.59, 0.71] # Wassiliadis et al
        self.c2p_vol = [0.39, 0.55] # Wassiliadis et al

        self.percantage_eol = 0.8 # EoL-Criterion  // End of Life: SOH = 80%
        self.percantage_safety = 0.07 # Assumption // SOC-Puffer based on Range Anxiety

        #Thermal Model
        self.Pcooler = 10e3  # Cooling power in W [Schimpe et al.]
        self.Pheater = 11.2e3  # Heating power in W [Schimpe et al.]
        self.COPcool = -3  # Coefficient of performance of cooling system in pu [Schimpe et al.]
        self.COPheat = 4  # Coefficient of performance of heating system in pu [Schimpe et al., Danish Energy Agency 2012]
        self.Ebat_Neubauer = 22.1e3  # Battery size used by Neubauer et al. in Wh [Neubauer et al.]
        self.Rth_Neubauer = 4.343  # Thermal resistance between housing and ambient used by Neubauer et al. in W/K [Neubauer et al.]
        self.Cth_Neubauer = 182e3  # Thermal mass of battery in J/K [Neubauer]
        self.T_Cool_on = 33  # Activation Cooling Threshold [Neubauer et al.]
        self.T_Cool_off = 31.5  # Deactivation Cooling Threshold [ID3 Paper]
        self.T_Heat = 10  # Heating Threshold [Neubauer et al.]
        self.cp_cell = 1045 # J/(kgK) Teichert Dissertation
        self.k_bh = 0.899  # K/W # Dissertation Teichert
        self.k_out = 10.9  # W/K # Dissertation Teichert
        self.c_aluminium = 896  # J/(kgK) Dissertation Teichert

        # Cost Model
        self.opp_cost = 0.0358 # Opportunity-Cost // in €/km/t per kg Hunter et al. 2021
        self.servicelife = 8 # Servicelife of the truck //Noll et al.
        self.c2p_cost = 2.07 # Cost Cell2Pack Scaling Factor //  Köng et. al
        self.c_fc_400 = 0.29 # Fast Charging cost // Ionity price
        self.c_sc_22 = 0.2515 # AC- Charging Price // Teichert et al.
        self.r = 1.095  # Interest rate // Teichert et al.
        self.bat_scrappage = 0.15 # residual value of battery at end of life in % [Burke and Fulton 2019]

        # Aging Model
        self.k_VW = 0.43 # NMC scaling factor according to Teichert