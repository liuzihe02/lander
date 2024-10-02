
#include <vector>
#include <tuple>
// Implementation (Agent.cpp)
#include "lander.h"

Agent::Agent()
{
    // empty constructor
    // make sure you dont call reset here else you run into problems
}

vector<double> Agent::reset(vector<double> init_conditions)
// returns the observations after resetting
// currently init_conditions is a length-9 vector containing position, velocity and orientation!
{

    reset_simulation();
    // Set initial conditions
    // BE CAREFUL OF THIS GLOBAL VARIABLE POSITION
    // we change global varibles from within; no ideal but no choice
    // we need to do this for update_lander_state to work

    // Extract initial conditions from init_conditions and set globally
    ::position = vector3d(init_conditions[0], init_conditions[1], init_conditions[2]);
    ::velocity = vector3d(init_conditions[3], init_conditions[4], init_conditions[5]);
    ::orientation = vector3d(init_conditions[6], init_conditions[7], init_conditions[8]);

    // these are always fixed!
    ::delta_t = 0.1;
    ::parachute_status = NOT_DEPLOYED;
    ::stabilized_attitude = true;
    ::autopilot_enabled = true;

    // Convert the initial state to double and return it
    vector<double> init_state = this->getState();

    return init_state;
}

// given the action, move the environment. does not return state for us!
void Agent::update(tuple<double> new_actions)
// given the environment, take an action
// very similar to Python Gym Env step
{

    // this actions will be available globally! is an instance variable, that is public
    this->actions = new_actions;

    ::throttle = std::get<0>(new_actions);

    // cout << "in update, agent throttle is: " << get<0>(this->actions) << " global throttle is: " << ::throttle << endl;

    // global env has changed
    // this will call autopilot with the agent
    update_lander_state();

    // vector<double> states = agent.getState();
    // std::cout << " Right After Update - Time: " << states[0]
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

vector<double> Agent::getState()
{
    // Create a NEW vector and populate it with copies of the global variables
    std::vector<double> state = {
        ::simulation_time,
        ::position.x, ::position.y, ::position.z,
        ::velocity.x, ::velocity.y, ::velocity.z,
        ::orientation.x, ::orientation.y, ::orientation.z,
        ::fuel,
        ::altitude,
        ::climb_speed,
        ::ground_speed};
    return state; // This returns a COPY that can be modified without affecting the originals
}

// tuple<double> Agent::getActions()
// {
//     cout << "in get, agent throttle is: " << get<0>(this->actions) << " global throttle is: " << ::throttle << endl;
//     // copy will automatically be made
//     return this->actions;
// }

bool Agent::isDone() const
{
    // return the global ones, instead of the local variables
    return ::landed || ::crashed;
}

double Agent::getReward(vector<double> rew_list) const

/**
 *note that delta_t is in terms of 0.1 seconds the simulation ends after 100-1000 seconds so average number of steps is 1000
 * when the end rewards are too large, incentive to explore more
 * I can afford to explore alot and get the big payout in the end
 */
{
    if (::landed && !::crashed)
        return rew_list[0];
    if (::crashed)
        return rew_list[1];
    // this controls the descent time
    return rew_list[2]; // Small negative reward for each step
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