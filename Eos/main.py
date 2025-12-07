# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 19:20:42 2025

@author: finnb
"""

"Parameters"
# Environment
air_temp_celsius = 20 # deg C, surrounding air temp

# Tanks
tank_ox_diam_out = 0.11 # m, oxidizer tank outer diameter
tank_ox_thick = 0.003 # m, oxidizer tank wall thickness
tank_ox_len = 1 # m, oxidizer tank length, total inner length
tank_fuel_diam_out = 0.05 # m, fuel tank outer diameter
tank_fuel_thick = 0.002 # m, fuel tank wall thickness
tank_fuel_len = 1 # m, fuel tank length, inner length below piston

# Valves
valve_ox_cv = 1.0 # flow coefficient of oxidizer run valve
valve_fuel_cv = 1.0 # flow coefficient of fuel run valve 

# Injectors
inj_ox_number = 12 # number of individual oxidizer orifices
inj_ox_diam_mm = 2.0 # mm, diameter of one ox orifice
inj_ox_cd = 0.7 # discharge coefficient oxidizer
inj_fuel_number = 12 # number of individual fuel orifices
inj_fuel_diam_mm = 1.3 # mm, diameter of one fuel orifice
inj_fuel_cd = 0.7 # discharge coefficient fuel

# Chamber
chamber_diam = 0.05 # m, chamber inner diameter
chamber_length = 0.1 # m, chamber inner diameter
chamber_throat = 0.003 # m, nozzle throat diameter
chamber_exit = 0.006 # m, nozzle exit diameter
chamber_Cstar_overwrite = 0.8 # combustion efficiency overwrite

# Rocket
rocket_mass_dry = 15 # kg, expected dry mass


"Script"
# imports
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from CoolProp.CoolProp import PropsSI as propsi
import rocketcea
from rocketcea.cea_obj import CEA_Obj 

# geometry
tank_ox_diam = tank_ox_diam_out - 2*tank_ox_thick # m, oxidizer tank inner diameter
tank_fuel_diam = tank_fuel_diam_out - 2*tank_fuel_thick # m, fuel tank inner diameter
tank_fuel_vol = (tank_fuel_diam/2)**2*np.pi * tank_fuel_len # m^3, fuel tank volume
tank_ox_vol = (tank_ox_diam/2)**2*np.pi * tank_ox_len - tank_fuel_vol# m^3, oxidizer tank volume

# initial conditions
gravity = 9.81 # m/s^2
air_R = 287.07 # surrounding air gas constant 
air_pressure_zero = 101325.0  # Pa, standard pressure at MSL in Pascal
air_temp = 273.15 + air_temp_celsius # K


"Functions"
def injector_flow_ox(T1, P1, P2)
    mdot = 0 # kg/s, mass flow rate
    return  mdot 

def injector_flow_fuel(T1, P1, P2)
    mdot = 0 # kg/s, mass flow rate
    return  mdot 


"Testing ground"

Q_init = propsi("Z", "T", 290, "Q", 1, "NitrousOxide")  

print(
    f"NO2 Tank\n"
    f"Q_init:       {Q_init:.2f} \n"
    )