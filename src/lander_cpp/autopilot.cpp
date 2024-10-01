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

    // at this point, we have position and velocity
    // have just gotten an action
    // Optional: Print current state or other relevant information
    std::cout << " Time: " << simulation_time
              // state
              << " Vx: " << velocity.x
              << " Vy: " << velocity.y
              << " Vz: " << velocity.z
              << " Rx: " << position.x
              << " Ry: " << position.y
              << " Rz: " << position.z
              << " Altitude: " << altitude
              << " Fuel Used: " << (1.0 - fuel) * FUEL_CAPACITY
              // action made
              << " Throttle Action: " << throttle
              << endl;
}