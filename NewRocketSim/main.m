%% Parameters
params

%% ODE solving
x_init = [0; 0; prop_mass+pres_mass];

tspan = 0:0.01:30; 
opts = odeset('Events', @stopping); 
[sol_t, sol_x, te, ~] = ode45(@(t_,x_)eom(t_,x_), tspan, x_init, opts);

sol_pressure = 0;
for i = 1:length(sol_x)
    sol_pressure(i) = pressure(sol_x(i,3));
end

apogee = max(sol_x(:,1))
max_vel = max(sol_x(:,2))

figure(1)
subplot(2,2,1)
plot(sol_t, sol_x(:,1));
xlabel('Time, [s]')
ylabel('Altitude, [m]')
subplot(2,2,2)
plot(sol_t, sol_x(:,2));
xlabel('Time, [s]')
ylabel('Velocity, [m/s]')
subplot(2,2,3)
plot(sol_t, rocket_mass+sol_x(:,3));
xlabel('Time, [s]')
ylabel('Mass, [kg]')
subplot(2,2,4)
plot(sol_t, sol_pressure);
xlabel('Time, [s]')
ylabel('Pressure, [Pa]')

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


function [zero_alt, isterm, direct] = stopping(~,x)
    params
    zero_alt = x(1);
    isterm = 1;
    direct = 0;
end