#!/usr/bin/env python3
"""Compute pressure in a rigid N2O tank using CoolProp.

Given: mass (kg), internal volume (L), temperature (K).
Outputs pressure in Pa and bar.
"""
from CoolProp.CoolProp import PropsSI

def pressure_for_mass_volume_temperature(mass_kg: float, volume_l: float, temperature_k: float) -> float:
    """Return pressure in Pa for given mass (kg), volume (L), temperature (K) of N2O in a rigid tank."""
    volume_m3 = volume_l / 1000.0
    density = mass_kg / volume_m3  # kg/m3
    # Use density (Dmass) and temperature to get pressure
    p_pa = PropsSI('P', 'T', temperature_k, 'Dmass', density, 'N2O')
    return p_pa

def liquid_fill_level(mass_kg: float, volume_l: float, temperature_k: float) -> float:
    """Return liquid fill level (volume fraction) for N2O in a rigid tank.
    
    Returns fraction between 0 and 1, where 1 = tank completely filled with liquid.
    """
    volume_m3 = volume_l / 1000.0
    p_pa = pressure_for_mass_volume_temperature(mass_kg, volume_l, temperature_k)
    
    # Get saturated liquid and vapor densities at this temperature
    rho_liquid = PropsSI('Dmass', 'T', temperature_k, 'Q', 0, 'N2O')  # Q=0 for liquid
    rho_vapor = PropsSI('Dmass', 'T', temperature_k, 'Q', 1, 'N2O')   # Q=1 for vapor
    
    # Using quality (dryness fraction) to find liquid volume fraction
    # mass = mass_liquid + mass_vapor
    # density_avg = mass / volume
    # quality Q = mass_vapor / mass_total
    overall_density = mass_kg / volume_m3
    
    # From two-phase relation: rho_avg = rho_liquid * (1-Q) + rho_vapor * Q
    # Solve for Q: Q = (rho_liquid - rho_avg) / (rho_liquid - rho_vapor)
    if rho_liquid != rho_vapor:
        quality = (rho_liquid - overall_density) / (rho_liquid - rho_vapor)
        liquid_fraction = 1.0 - quality
        liquid_fraction = max(0.0, min(1.0, liquid_fraction))  # Clamp to [0, 1]
    else:
        liquid_fraction = 0.0
    
    return liquid_fraction

def main():
    mass = 8.0       # kg
    volume_l = 10.0  # liters
    T = 293.0        # K
    try:
        p = pressure_for_mass_volume_temperature(mass, volume_l, T)
    except Exception as e:
        print(f"Error computing pressure with CoolProp: {e}")
        return
    print(f"Mass: {mass} kg, Volume: {volume_l} L, T: {T} K")
    print(f"Density = {mass/(volume_l/1000.0):.3f} kg/m^3")
    print(f"Pressure = {p:.3f} Pa = {p/1e5:.6f} bar")
    print(f"Liquid fill level = {liquid_fill_level(mass, volume_l, T)*100:.2f}%")

if __name__ == '__main__':
    main()
