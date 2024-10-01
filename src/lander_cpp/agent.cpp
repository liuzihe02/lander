
#include <vector>
// Implementation (Agent.cpp)
#include "lander.h"

Agent::Agent()
{
    // empty constructor
    // make sure you dont call reset here else you run into problems
}

void Agent::reset()
{
    reset_simulation();
    // Set initial conditions
    // BE CAREFUL OF THIS GLOBAL VARIABLE POSITION
    // we change global varibles from within; no ideal but no choice
    // we need to do this for update_lander_state to work
    ::position = vector3d(0.0, -(MARS_RADIUS + 10000.0), 0.0);
    ::velocity = vector3d(0.0, 0.0, 0.0);
    ::orientation = vector3d(0.0, 0.0, 910.0);
    ::delta_t = 3;
    ::parachute_status = NOT_DEPLOYED;
    ::stabilized_attitude = true;
    ::autopilot_enabled = true;
}

std::vector<double> Agent::getState()
{
    // Create a NEW vector and populate it with copies of the global variables
    std::vector<double> state = {
        ::simulation_time,
        ::position.x, ::position.y, ::position.z,
        ::velocity.x, ::velocity.y, ::velocity.z,
        ::orientation.x, ::orientation.y, ::orientation.z,
        ::fuel,
        ::altitude};
    return state; // This returns a COPY that can be modified without affecting the originals
}

void Agent::step()
// given the environment, take an action
{
    // I think no syncing  needed here
    // syncFromGlobals();

    // global env has changed
    // this will call autopilot with the agent
    update_lander_state();
}

bool Agent::isDone() const
{
    return landed || crashed;
}

double Agent::getReward() const
{
    if (landed && !crashed)
        return 100.0;
    if (crashed)
        return -100.0;
    return -1.0; // Small negative reward for each step
}

// void Agent::syncToGlobals()
// {
//     // Copy our private state to the global variables
//     ::simulation_time = this->simulation_time;
//     ::position = this->position;
//     ::velocity = this->velocity;
//     ::orientation = this->orientation;
//     ::fuel = this->fuel;
//     ::landed = this->landed;
//     ::crashed = this->crashed;
//     ::altitude = this->altitude;
// }

// void Agent::syncFromGlobals()
// {
//     // Copy global variables to our private state
//     this->simulation_time = ::simulation_time;
//     this->position = ::position;
//     this->velocity = ::velocity;
//     this->orientation = ::orientation;
//     this->fuel = ::fuel;
//     this->landed = ::landed;
//     this->crashed = ::crashed;
//     this->altitude = ::altitude;
// }