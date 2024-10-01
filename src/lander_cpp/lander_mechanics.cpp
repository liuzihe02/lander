// include this in the lander.h
#include "lander.h"

/**
 *
 * CORE FUNCTIONALITY. For the stuff involving OpenGL, I've added a flag to use GLUT
 *
 *
 */

bool safe_to_deploy_parachute(void)
// Checks whether the parachute is safe to deploy at the current position and velocity
{
    double drag;

    // Assume high Reynolds number, quadratic drag = -0.5 * rho * v^2 * A * C_d
    drag = 0.5 * DRAG_COEF_CHUTE * atmospheric_density(position) * 5.0 * 2.0 * LANDER_SIZE * 2.0 * LANDER_SIZE * velocity_from_positions.abs2();
    // Do not use the global variable "altitude" here, in case this function is called from within the
    // numerical_dynamics function, before altitude is updated in the update_visualization function
    if ((drag > MAX_PARACHUTE_DRAG) || ((velocity_from_positions.abs() > MAX_PARACHUTE_SPEED) && ((position.abs() - MARS_RADIUS) < EXOSPHERE)))
        return false;
    else
        return true;
}

void update_visualization(void)
// The visualization part of the idle function. Re-estimates altitude, velocity, climb speed and ground
// speed from current and previous positions. Updates throttle and fuel levels, then redraws all subwindows.
{
    static vector3d last_track_position;
    vector3d av_p, d;
    double a, b, c, mu;

    simulation_time += delta_t;
    altitude = position.abs() - MARS_RADIUS;

    // Use average of current and previous positions when calculating climb and ground speeds
    av_p = (position + last_position).norm();
    if (delta_t != 0.0)
        velocity_from_positions = (position - last_position) / delta_t;
    else
        velocity_from_positions = vector3d(0.0, 0.0, 0.0);
    climb_speed = velocity_from_positions * av_p;
    ground_speed = (velocity_from_positions - climb_speed * av_p).abs();

    // Check to see whether the lander has landed
    if (altitude < LANDER_SIZE / 2.0)
    {
        // RENDERING HERE
        if (render)
        {
            glutIdleFunc(NULL);
        }

        // Estimate position and time of impact
        d = position - last_position;
        a = d.abs2();
        b = 2.0 * last_position * d;
        c = last_position.abs2() - (MARS_RADIUS + LANDER_SIZE / 2.0) * (MARS_RADIUS + LANDER_SIZE / 2.0);
        mu = (-b - sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
        position = last_position + mu * d;
        simulation_time -= (1.0 - mu) * delta_t;
        altitude = LANDER_SIZE / 2.0;
        landed = true;
        if ((fabs(climb_speed) > MAX_IMPACT_DESCENT_RATE) || (fabs(ground_speed) > MAX_IMPACT_GROUND_SPEED))
            crashed = true;
        velocity_from_positions = vector3d(0.0, 0.0, 0.0);
    }

    // Update throttle and fuel (throttle might have been adjusted by the autopilot)
    if (throttle < 0.0)
        throttle = 0.0;
    if (throttle > 1.0)
        throttle = 1.0;
    fuel -= delta_t * (FUEL_RATE_AT_MAX_THRUST * throttle) / FUEL_CAPACITY;
    if (fuel <= 0.0)
        fuel = 0.0;
    if (landed || (fuel == 0.0))
        throttle = 0.0;
    throttle_control = (short)(throttle * THROTTLE_GRANULARITY + 0.5);

    // Check to see whether the parachute has vaporized or the tethers have snapped
    if (parachute_status == DEPLOYED)
    {
        if (!safe_to_deploy_parachute() || parachute_lost)
        {
            parachute_lost = true; // to guard against the autopilot reinstating the parachute!
            parachute_status = LOST;
        }
    }

    // Update record of lander's previous positions, but only if the position or the velocity has
    // changed significantly since the last update
    if (!track.n || (position - last_track_position).norm() * velocity_from_positions.norm() < TRACK_ANGLE_DELTA || (position - last_track_position).abs() > TRACK_DISTANCE_DELTA)
    {
        track.pos[track.p] = position;
        track.n++;
        if (track.n > N_TRACK)
            track.n = N_TRACK;
        track.p++;
        if (track.p == N_TRACK)
            track.p = 0;
        last_track_position = position;
    }

    if (render)
    {
        // Redraw everything
        refresh_all_subwindows();
    }
}

void attitude_stabilization(void)
// Three-axis stabilization to ensure the lander's base is always pointing downwards
// calculate desired orientation
{
    vector3d up, left, out;
    double m[16];

    up = position.norm(); // this is the direction we want the lander's nose to point in

    // !!!!!!!!!!!!! HINT TO STUDENTS ATTEMPTING THE EXTENSION EXERCISES !!!!!!!!!!!!!!
    // For any-angle attitude control, we just need to set "up" to something different,
    // and leave the remainder of this function unchanged. For example, suppose we want
    // the attitude to be stabilized at stabilized_attitude_angle to the vertical in the
    // close-up view. So we need to rotate "up" by stabilized_attitude_angle degrees around
    // an axis perpendicular to the plane of the close-up view. This axis is given by the
    // vector product of "up"and "closeup_coords.right". To calculate the result of the
    // rotation, search the internet for information on the axis-angle rotation formula.

    // Set left to something perpendicular to up
    left.x = -up.y;
    left.y = up.x;
    left.z = 0.0;
    if (left.abs() < SMALL_NUM)
    {
        left.x = -up.z;
        left.y = 0.0;
        left.z = up.x;
    }
    left = left.norm();
    out = left ^ up;
    // Construct modelling matrix (rotation only) from these three vectors
    m[0] = out.x;
    m[1] = out.y;
    m[2] = out.z;
    m[3] = 0.0;
    m[4] = left.x;
    m[5] = left.y;
    m[6] = left.z;
    m[7] = 0.0;
    m[8] = up.x;
    m[9] = up.y;
    m[10] = up.z;
    m[11] = 0.0;
    m[12] = 0.0;
    m[13] = 0.0;
    m[14] = 0.0;
    m[15] = 1.0;
    // Decomponse into xyz Euler angles
    orientation = matrix_to_xyz_euler(m);
}

vector3d thrust_wrt_world(void)
// Works out thrust vector in the world reference frame, given the lander's orientation
{
    double m[16], k, delayed_throttle, lag = ENGINE_LAG;
    vector3d a, b;

    // these static variables maintain the state
    static double lagged_throttle = 0.0;
    static double last_time_lag_updated = -1.0;

    if (simulation_time < last_time_lag_updated)
        lagged_throttle = 0.0; // simulation restarted

    // clamps throttle and also sets it to be a sensible value
    if (throttle < 0.0)
        throttle = 0.0;
    if (throttle > 1.0)
        throttle = 1.0;
    if (landed || (fuel == 0.0))
        throttle = 0.0;

    if (simulation_time != last_time_lag_updated)
    {

        // Delayed throttle value from the throttle history buffer
        if (throttle_buffer_length > 0)
        {
            delayed_throttle = throttle_buffer[throttle_buffer_pointer];
            throttle_buffer[throttle_buffer_pointer] = throttle;
            throttle_buffer_pointer = (throttle_buffer_pointer + 1) % throttle_buffer_length;
        }
        else
            delayed_throttle = throttle;

        // Lag, with time constant ENGINE_LAG
        if (lag <= 0.0)
            k = 0.0;
        else
            k = pow(exp(-1.0), delta_t / lag);
        lagged_throttle = k * lagged_throttle + (1.0 - k) * delayed_throttle;

        // last_time_lag is simulation time lagged by one step
        last_time_lag_updated = simulation_time;
    }

    if (stabilized_attitude && (stabilized_attitude_angle == 0))
    { // specific solution, avoids rounding errors in the more general calculation below
        b = lagged_throttle * MAX_THRUST * position.norm();
    }
    else
    {
        a.x = 0.0;
        a.y = 0.0;
        a.z = lagged_throttle * MAX_THRUST;
        // this updates m
        xyz_euler_to_matrix(orientation, m);
        b.x = m[0] * a.x + m[4] * a.y + m[8] * a.z;
        b.y = m[1] * a.x + m[5] * a.y + m[9] * a.z;
        b.z = m[2] * a.x + m[6] * a.y + m[10] * a.z;
    }
    // this is the thrust force vector
    return b;
}

void update_lander_state(void)
// The GLUT idle function, called every time round the event loop
{
    if (render)
    {
        unsigned long delay;

        // User-controlled delay
        if ((simulation_speed > 0) && (simulation_speed < 5))
        {
            delay = (5 - simulation_speed) * MAX_DELAY / 4;
#ifdef _WIN32
            Sleep(delay / 1000); // milliseconds
#else
            usleep((useconds_t)delay); // microseconds
#endif
        }
    }

    // This needs to be called every time step, even if the close-up view is not being rendered,
    // since any-angle attitude stabilizers reference closeup_coords.right
    update_closeup_coords();

    // Update historical record
    last_position = position;

    // Mechanical dynamics, update the position and velocity
    numerical_dynamics();

    // Refresh the visualization
    update_visualization();
}

void reset_simulation(void)
// Resets the simulation to the initial state
{
    vector3d p, tv;
    unsigned long i;

    // Reset these three lander parameters here, so they can be overwritten in initialize_simulation() if so desired
    stabilized_attitude_angle = 0;
    throttle = 0.0;
    fuel = 1.0;

    // Restore initial lander state
    initialize_simulation();

    // Check whether the lander is underground - if so, make sure it doesn't move anywhere
    landed = false;
    crashed = false;
    altitude = position.abs() - MARS_RADIUS;
    if (altitude < LANDER_SIZE / 2.0)
    {
        if (render)
        {
            glutIdleFunc(NULL);
        }
        landed = true;
        velocity = vector3d(0.0, 0.0, 0.0);
    }

    // Visualisation routine's record of various speeds and velocities
    velocity_from_positions = velocity;
    last_position = position - delta_t * velocity_from_positions;
    p = position.norm();
    climb_speed = velocity_from_positions * p;
    tv = velocity_from_positions - climb_speed * p;
    ground_speed = tv.abs();

    // Miscellaneous state variables
    throttle_control = (short)(throttle * THROTTLE_GRANULARITY + 0.5);
    simulation_time = 0.0;
    track.n = 0;
    track.p = 0;
    parachute_lost = false;
    closeup_coords.initialized = false;
    closeup_coords.backwards = false;
    closeup_coords.right = vector3d(1.0, 0.0, 0.0);
    update_closeup_coords();

    // Initialize the throttle history buffer
    if (delta_t > 0.0)
        throttle_buffer_length = (unsigned long)(ENGINE_DELAY / delta_t + 0.5);
    else
        throttle_buffer_length = 0;
    if (throttle_buffer_length > 0)
    {
        if (throttle_buffer != NULL)
            delete[] throttle_buffer;
        throttle_buffer = new double[throttle_buffer_length];
        for (i = 0; i < throttle_buffer_length; i++)
            throttle_buffer[i] = throttle;
        throttle_buffer_pointer = 0;
    }

    if (render)
    {
        // Reset GLUT state
        if (paused || landed)
            refresh_all_subwindows();
        else
        {
            glutIdleFunc(update_lander_state);
        }
    }
}