#include "lander.h"

/**
 * MAIN FUNCTION
 *
 *
 */

// declare core variables, that I change regularly
// IVE ADDED THIS: whether or not to use GLUT to simulate or no picture
bool render = false;
bool agent_flag = true;

int main(int argc, char *argv[])
// Initializes GLUT windows and lander state, then enters GLUT main loop
{
    if (render)
    {
        if (agent_flag)
        {
            cout << "issues with global variables. Not implemented yet!" << endl;
            return 1;
        }
        // TODO: the referencing & here MAY CAUSE ISSUES
        run_graphics(argc, argv);
        return 0;
    }
    // DONT USE GLUT RENDERING
    else
    {

        run_one_episode();
        return 0;
    }
}

void run_graphics(int argc, char *argv[])
{
    {
        int i;

        // Main GLUT window
        std::cout << "init here" << endl;
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(PREFERRED_WIDTH, PREFERRED_HEIGHT);
        view_width = (PREFERRED_WIDTH - 4 * GAP) / 2;
        view_height = (PREFERRED_HEIGHT - INSTRUMENT_HEIGHT - 4 * GAP);
        main_window = glutCreateWindow("Mars Lander (Gabor Csanyi and Andrew Gee, August 2019)");
        glDrawBuffer(GL_BACK);
        glLineWidth(2.0);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glutDisplayFunc(draw_main_window);
        glutReshapeFunc(reshape_main_window);
        glutIdleFunc(update_lander_state);
        glutKeyboardFunc(glut_key);
        glutSpecialFunc(glut_special);

        // The close-up view subwindow
        closeup_window = glutCreateSubWindow(main_window, GAP, GAP, view_width, view_height);
        glDrawBuffer(GL_BACK);
        setup_lights();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
        glEnable(GL_CULL_FACE); // we only need back faces for the parachute
        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_NORMALIZE);
        glDepthFunc(GL_LEQUAL);
        glShadeModel(GL_SMOOTH);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE); // we need two-sided lighting for the parachute
        glEnable(GL_COLOR_MATERIAL);
        glFogi(GL_FOG_MODE, GL_EXP);
        glutDisplayFunc(draw_closeup_window);
        glutMouseFunc(closeup_mouse_button);
        glutMotionFunc(closeup_mouse_motion);
        glutKeyboardFunc(glut_key);
        glutSpecialFunc(glut_special);
        texture_available = generate_terrain_texture();
        if (!texture_available)
            do_texture = false;
        closeup_offset = 50.0;
        closeup_xr = 10.0;
        closeup_yr = 0.0;
        terrain_angle = 0.0;

        // The orbital view subwindow
        orbital_window = glutCreateSubWindow(main_window, view_width + 3 * GAP, GAP, view_width, view_height);
        glDrawBuffer(GL_BACK);
        setup_lights();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
        glEnable(GL_CULL_FACE); // since the only polygons in this view define a solid sphere
        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_NORMALIZE);
        glDepthFunc(GL_LEQUAL);
        glShadeModel(GL_SMOOTH);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        glEnable(GL_COLOR_MATERIAL);
        glutDisplayFunc(draw_orbital_window);
        glutMouseFunc(orbital_mouse_button);
        glutMotionFunc(orbital_mouse_motion);
        glutKeyboardFunc(glut_key);
        glutSpecialFunc(glut_special);
        quadObj = gluNewQuadric();
        orbital_quat.v.x = 0.53;
        orbital_quat.v.y = -0.21;
        orbital_quat.v.z = 0.047;
        orbital_quat.s = 0.82;
        normalize_quat(orbital_quat);
        save_orbital_zoom = 1.0;
        orbital_zoom = 1.0;

        // The instrument subwindow
        instrument_window = glutCreateSubWindow(main_window, GAP, view_height + 3 * GAP, 2 * (view_width + GAP), INSTRUMENT_HEIGHT);
        glutDisplayFunc(draw_instrument_window);
        glutKeyboardFunc(glut_key);
        glutSpecialFunc(glut_special);

        // Generate the random number table
        srand(0);
        for (i = 0; i < N_RAND; i++)
            randtab[i] = (float)rand() / RAND_MAX;

        // Initialize the simulation state
        reset_simulation();
        microsecond_time(time_program_started);

        glutMainLoop();
    }
}

void run_one_episode()
{
    if (agent_flag)
    {

#include <vector>

        std::vector<double> init_conditions = {
            0.0,                    // x position
            -(MARS_RADIUS + 10000), // y position
            0.0,                    // z position
            50.0,                   // x velocity
            0.0,                    // y velocity
            0.0,                    // z velocity
            0.0,                    // roll
            0.0,                    // pitch
            910.0                   // yaw
        };
        agent.reset(init_conditions);

        float test_throttle = 0.0;

        // Main simulation loop
        while (!landed && !crashed)
        // for (int i = 0; i < 10; i++)
        {
            // acess global variable
            vector<double> states = agent.getState();
            // at this point, we have position and velocity
            // have just gotten an action
            // Optional: Print current state or other relevant information
            std::cout << " PRESTEP - Time: " << states[0]
                      // state
                      << " Rx: " << states[1] << " Global Pos X: " << position.x
                      << " Ry: " << states[2] << " Global Pos Y: " << position.y
                      << " Vx: " << states[4] << " Global V X: " << velocity.x
                      << " Vy: " << states[5] << " Global V Y: " << position.x

                      << " Fuel Left (Proportion): " << states[10]
                      << " Altitude: " << states[11] << " Global Altitude: " << altitude
                      // action made
                      << " Throttle Action: " << throttle
                      << endl;

            // step with increasing throttle to test
            if (test_throttle < 1.0)
            {
                test_throttle += 0.00012;
            }

            agent.update(std::make_tuple(test_throttle));

            states = agent.getState();
            std::cout << " POSTSTEP - Time: " << states[0]
                      // state
                      << " Rx: " << states[1] << " Global Pos X: " << position.x
                      << " Ry: " << states[2] << " Global Pos Y: " << position.y
                      << " Vx: " << states[4] << " Global V X: " << velocity.x
                      << " Vy: " << states[5] << " Global V Y: " << position.x

                      << " Fuel Left (Proportion): " << states[10]
                      << " Altitude: " << states[11] << " Global Altitude: " << altitude
                      // action made
                      << " Throttle Action: " << throttle
                      << endl;
        }

        // Simulation ended
        if (crashed)
        {
            std::cout << "Lander crashed!" << std::endl;
        }
        else
        {
            std::cout << "Lander landed safely!" << std::endl;
        }

        // Print final stats
        std::cout << "Crashed status " << crashed << std::endl;
        std::cout << "Final altitude: " << altitude << std::endl;
        std::cout << "Ground speed at landing: " << ground_speed << std::endl;
        std::cout << "Descent rate at landing: " << -climb_speed << std::endl;
        std::cout << "Remaining fuel " << fuel * FUEL_CAPACITY << " litres" << std::endl;
    }

    // choose not to render, but no agent
    else
    {
        std::cout << "youve chosen not to render, without an agent. Not implemented yet!" << std::endl;
    }
}
