%% Parameters
params

%% ODE solving
x_init = [0; 0; prop_mass+pres_mass];

tspan = 0:0.01:30; 
opts = odeset('Events', @stopping_vel); 
[sol_t, sol_x, te, ~] = ode45(@(t_,x_)eom(t_,x_), tspan, x_init, opts);

%% other values

sol_pressure = 0;
sol_thrust = 0;
sol_accel = 0;
for i = 1:length(sol_x)
    sol_pressure(i) = pressure(sol_x(i,3));
    [~, thrust] = flow(sol_x(i,3));
    sol_thrust(i) = thrust;
    sol_accel(i) = thrust / (sol_x(i,3)+rocket_mass) / g - 1;
end

apogee = max(sol_x(:,1));
max_vel = max(sol_x(:,2));
max_accel = max(sol_accel);
impulse = trapz(sol_t, sol_thrust);
isp = impulse / (prop_mass + pres_mass) / g;
max_thrust = max(sol_thrust);

%% display and plot

disp(['  Apogee     ', 'v_max     ', 'g_max'])
disp([apogee, max_vel, max_accel])
disp(['  Impulse    ', 'ISP       ', 'T_max'])
disp([impulse, isp, max_thrust])


figure(1)
subplot(2,3,1)
plot(sol_t, sol_x(:,1));
xlabel('Time, [s]')
ylabel('Altitude, [m]')
subplot(2,3,2)
plot(sol_t, sol_x(:,2));
xlabel('Time, [s]')
ylabel('Velocity, [m/s]')
subplot(2,3,3)
plot(sol_t, sol_accel);
xlabel('Time, [s]')
ylabel('Acceleration, [g]')
subplot(2,3,4)
plot(sol_t, rocket_mass+sol_x(:,3));
xlabel('Time, [s]')
ylabel('Mass, [kg]')
subplot(2,3,5)
plot(sol_t, sol_pressure);
xlabel('Time, [s]')
ylabel('Pressure, [Pa]')
subplot(2,3,6)
plot(sol_t, sol_thrust);
xlabel('Time, [s]')
ylabel('Thrust, [N]')

%% calcs
function [x_dot] = eom(t, x)
    alt = x(1); v = x(2); m = x(3);
    params
    
    [m_dot, thrust] = flow(m);

    v_dot = thrust / (rocket_mass + m) - g;
    alt_dot = x(2);

    x_dot = [alt_dot; v_dot; m_dot];
end

function [m_dot, thrust] = flow(m)
    params
    p = pressure(m);
    if m > pres_mass
        u = sqrt( 2*p / prop_density / (1 + K_loss - area_ratio^2) );
        m_dot = - prop_density * nozzle_area * u;
        thrust = - m_dot * u;
    elseif m > 0
        rho = m / tank_volume;
        T = p/rho / pres_R;
        u = sqrt(2*pres_gamma/(pres_gamma+1) * pres_R * T); 
        % rho_throat = rho * (1 + (pres_gamma-1)/2 )^(1/(1-pres_gamma));
        m_dot = - rho * nozzle_area * u;
        thrust = - u * m_dot;
    else
        m_dot = 0;
        thrust = 0;
    end
    
end


function [p] = pressure(m)
    params 
    if m > pres_mass
        p = pres_pressure * ( pres_volume / (tank_volume - (m-pres_mass)/prop_density) )^pres_gamma;
    elseif m > 0
        p = pres_init_p * ( m/pres_mass )^pres_gamma;
    else
        p = 0;
    end
end


function [zero_alt, isterm, direct] = stopping_alt(~,x)
    params
    zero_alt = x(1);
    isterm = 1;
    direct = 0;
end

function [zero_vel, isterm, direct] = stopping_vel(~,x)
    params
    zero_vel = x(2)+0.001;
    isterm = 1;
    direct = 0;
end