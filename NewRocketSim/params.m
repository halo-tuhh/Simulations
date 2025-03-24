%% Parameters
%%% Rocket
rocket_mass = 6;
tank_volume = 20 * 1e-3;
tank_radius = 6e-2;
nozzle_radius = 4e-3;
K_loss = 0.0;

%%% Propellant
prop_mass = 7.5;
prop_density = 998;

%%% Pressurant
pres_pressure = 60 * 1e5;
pres_temp = 293;
pres_gamma = 1.4; % air 
% pres_gamma = 1.29; % co2
pres_R = 287; % air
% pres_R = 189; % co2


%%% constants
g = 9.81;
atmos_pressure = 1e5;
atmos_density = 1.2;

%% calcs
nozzle_area = pi*nozzle_radius^2;
tank_area = pi*tank_radius^2;
area_ratio = nozzle_area / tank_area;
pres_volume = tank_volume - prop_mass/prop_density;
pres_mass = pres_pressure * pres_volume / (pres_R * pres_temp);
pres_density = pres_mass / pres_volume;

pres_init_p = pres_pressure * ( pres_volume / tank_volume)^pres_gamma;
pres_init_rho = pres_mass / tank_volume;
pres_init_t = pres_init_p * tank_volume / ( pres_R * pres_mass );
pres_init_c = sqrt(pres_gamma*pres_R*pres_init_t);