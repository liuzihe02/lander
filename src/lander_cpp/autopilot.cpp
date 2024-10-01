// Abstracting out autopilot code

#include "lander.h"

// Constants
double K_h = 2e-2;
double K_p = 2;
double delta = 0.5;

// Calculate error
double calculate_error()
{
    double altitude = position.abs() - MARS_RADIUS;
    vector3d pos_norm = position.norm();                  // Position unit vector
    return -(0.5 + K_h * altitude + velocity * pos_norm); // Scalar product of velocity and position unit vector, custom implementation
}

void autopilot(void)
// Autopilot to adjust the engine throttle, parachute and attitude control
{
    if (!agent_flag)
    {
        autopilot_control();
    }
    else
    {
        autopilot_agent();
    }
}

void autopilot_control(void)
{
    // Calculate (pure) controller output
    double error, P_out;
    error = calculate_error();
    P_out = K_p * error;

    // Calculate if it is safe to deploy the parachute

    // First calculate if the lander is decelerating
    vector3d acceleration = get_acceleration();
    bool decelerating = (acceleration * velocity < 0);

    // Then check if it is safe in general and decelerating
    if (safe_to_deploy_parachute() && decelerating)
    {
        parachute_status = DEPLOYED;
    }

    // Calculate throttle
    if (P_out <= -delta)
    {
        throttle = 0;
    }
    else if (P_out > -delta && P_out < 1 - delta)
    {
        throttle = P_out + delta;
    }
    else
    {
        throttle = 1;
    }
}

void autopilot_agent(void)
{
    throttle = 1.0;
    // acess global variable
    vector<double> states = agent.getState();
    // at this point, we have position and velocity
    // have just gotten an action
    // Optional: Print current state or other relevant information
    std::cout << " Time: " << states[0]
              // state
              << " Rx: " << states[1]
              << " Ry: " << states[2]
              << " Rz: " << states[3]
              << " Vx: " << states[4]
              << " Vy: " << states[5]
              << " Vz: " << states[6]

              << " Fuel Used: " << (1.0 - states[10]) * FUEL_CAPACITY
              << " Altitude: " << states[11]
              // action made
              << " Throttle Action: " << throttle
              << endl;
}