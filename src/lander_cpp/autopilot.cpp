// Abstracting out autopilot code

#include "lander.h"

// Constants
double K_h = 2e-2;
double K_p = 2;
double delta = 0.5;

// actually declare the agent here. this should be globally accessible!
Agent agent;

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

    cout << "time" << simulation_time << "throttle" << throttle << endl;
}

void autopilot_agent(void)
{
    // PARACHUTE ALGORITHIM GENERIC
    //  First calculate if the lander is decelerating
    vector3d acceleration = get_acceleration();
    bool decelerating = (acceleration * velocity < 0);

    // Then check if it is safe in general and decelerating
    if (safe_to_deploy_parachute() && decelerating)
    {
        parachute_status = DEPLOYED;
    }

    // I CANT SEEM TO ACCESS GLOBAL AGENT ACTIONS, SO IVE SET THROTTLE IN UPDATE

    // // get the first element of a tuple
    // double throttle_action = std::get<0>(agent.actions);
    // cout << "in autopilot, agent throttle is: " << get<0>(agent.actions) << " global throttle is: " << ::throttle << endl;
    // // update GLOBAL throttle
    // ::throttle = throttle_action;

    // vector<double> states = agent.getState();
    // std::cout << " RIGHT AFTER THROTTLE SET - Time: " << states[0]
    //           // state
    //           << " Rx: " << states[1] << " Global Pos X: " << position.x
    //           << " Ry: " << states[2] << " Global Pos Y: " << position.y
    //           << " Vx: " << states[4] << " Global V X: " << velocity.x
    //           << " Vy: " << states[5] << " Global V Y: " << position.x

    //           << " Fuel Left: " << (states[10]) * FUEL_CAPACITY
    //           << " Altitude: " << states[11] << " Global Altitude: " << altitude
    //           // action made
    //           << " Throttle Action: " << throttle
    //           << endl;
}