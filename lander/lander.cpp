// Mars lander simulator
// Version 1.11
// Mechanical simulation functions
// Gabor Csanyi and Andrew Gee, August 2019

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation, to make use of it
// for non-commercial purposes, provided that (a) its original authorship
// is acknowledged and (b) no modified versions of the source code are
// published. Restriction (b) is designed to protect the integrity of the
// exercise for future generations of students. The authors would be happy
// to receive any suggested modifications by private correspondence to
// ahg@eng.cam.ac.uk and gc121@eng.cam.ac.uk.

#include "lander.h"
// this is for using the value of pi as constant M_PI
#include <cmath>

vector3d get_acceleration(void)
{
  // declare the types of all our variables used
  vector3d a_total, f_gravity, f_thrust, lander_drag, chute_drag;
  double mass;

  // get current mass
  mass = UNLOADED_LANDER_MASS + FUEL_DENSITY * FUEL_CAPACITY * fuel;

  // first get the acceleration due only to gravity, get the unit vector of position, then divide by the norm squared
  f_gravity = -(GRAVITY * MARS_MASS * mass) * position.norm() / position.abs2();

  f_thrust = thrust_wrt_world();

  // multiply by the relevant constants to the velocity unit vector, lander area has a circular base
  lander_drag = -0.5 * atmospheric_density(position) * DRAG_COEF_LANDER * (M_PI * pow(LANDER_SIZE, 2)) * velocity.abs2() * velocity.norm();

  // if parachute deployed, get that drag too
  if (parachute_status == DEPLOYED)
  {
    // the parachute area trumps the lander area, 5 sqaures each of length 2* lander size
    chute_drag = -0.5 * atmospheric_density(position) * DRAG_COEF_LANDER * (5.0 * 2.0 * LANDER_SIZE * 2.0 * LANDER_SIZE) * velocity.abs2() * velocity.norm();
  }
  else
  {
    chute_drag = vector3d(0, 0, 0);
  }

  // get current total acceleration
  a_total = (f_gravity + f_thrust + lander_drag + chute_drag) / mass;
  return a_total;
}

void euler_method()
// run simulation using euler method
{
  // note that position and velocity are global variables!
  vector3d acceleration;

  // // first value of position must use euler as 2 values of position needed for verlet
  if (simulation_time == 0)
  {
    // update acceleration
    acceleration = get_acceleration();
    // use a variable to store previous position
    position = position + delta_t * velocity;
    velocity = velocity + delta_t * acceleration;
  }
}

void verlet_method()
// run the simulation using verlet method
{
  // note that position and velocity are global variables!
  vector3d acceleration, position_next;

  // we use a static variable for previous position so when calling the function again, we can simply get the previous value
  static vector3d position_prev;

  // // first value of position must use euler as 2 values of position needed for verlet
  if (simulation_time == 0)
  {
    // first step,we step forward once, initialize the position and position_prev, using euler method
    acceleration = get_acceleration();
    position_prev = position;
    position = position + velocity * delta_t;
  }
  else
  {
    // update acceleration
    acceleration = get_acceleration();
    // use a variable to store previous position
    position_next = position * 2 - position_prev + acceleration * delta_t * delta_t;
    velocity = (position_next - position_prev) * 0.5 / delta_t;

    // x_prev <- x, move one step forward
    position_prev = position;
    // x <- x_next, move one step forward
    position = position_next;
  }
}

void numerical_dynamics(void)
// This is the function that performs the numerical integration to update the
// lander's pose. The time step is delta_t (global variable).
{
  // change this for the mode
  verlet_method();

  // Here we can apply an autopilot to adjust the thrust, parachute and attitude
  if (autopilot_enabled)
    autopilot();

  // Here we can apply 3-axis stabilization to ensure the base is always pointing downwards
  if (stabilized_attitude)
    attitude_stabilization();
}

void initialize_simulation(void)
// Lander pose initialization - selects one of 10 possible scenarios
{
  // The parameters to set are:
  // position - in Cartesian planetary coordinate system (m)
  // velocity - in Cartesian planetary coordinate system (m/s)
  // orientation - in lander coordinate system (xyz Euler angles, degrees)
  // delta_t - the simulation time step
  // boolean state variables - parachute_status, stabilized_attitude, autopilot_enabled
  // scenario_description - a descriptive string for the help screen

  scenario_description[0] = "circular orbit";
  scenario_description[1] = "descent from 10km";
  scenario_description[2] = "elliptical orbit, thrust changes orbital plane";
  scenario_description[3] = "polar launch at escape velocity (but drag prevents escape)";
  scenario_description[4] = "elliptical orbit that clips the atmosphere and decays";
  scenario_description[5] = "descent from 200km";
  scenario_description[6] = "";
  scenario_description[7] = "";
  scenario_description[8] = "";
  scenario_description[9] = "";

  switch (scenario)
  {

  case 0:
    // a circular equatorial orbit
    position = vector3d(1.2 * MARS_RADIUS, 0.0, 0.0);
    velocity = vector3d(0.0, -3247.087385863725, 0.0);
    orientation = vector3d(0.0, 90.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = true;
    break;

  case 1:
    // a descent from rest at 10km altitude
    position = vector3d(0.0, -(MARS_RADIUS + 10000.0), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 910.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = true;
    autopilot_enabled = true;
    break;

  case 2:
    // an elliptical polar orbit
    position = vector3d(0.0, 0.0, 1.2 * MARS_RADIUS);
    velocity = vector3d(3500.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 90.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = true;
    break;

  case 3:
    // polar surface launch at escape velocity (but drag prevents escape)
    position = vector3d(0.0, 0.0, MARS_RADIUS + LANDER_SIZE / 2.0);
    velocity = vector3d(0.0, 0.0, 5027.0);
    orientation = vector3d(0.0, 0.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = true;
    autopilot_enabled = true;
    break;

  case 4:
    // an elliptical orbit that clips the atmosphere each time round, losing energy
    position = vector3d(0.0, 0.0, MARS_RADIUS + 100000.0);
    velocity = vector3d(4000.0, 0.0, 0.0);
    orientation = vector3d(0.0, 90.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = true;
    break;

  case 5:
    // a descent from rest at the edge of the exosphere
    position = vector3d(0.0, -(MARS_RADIUS + EXOSPHERE), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 90.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = true;
    break;

  case 6:
    break;

  case 7:
    break;

  case 8:
    break;

  case 9:
    break;
  }
}
