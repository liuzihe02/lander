# %% ###############
# ASSIGNMENT 2 #####
####################
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_t

# %%
"""
VECTORIZE CODE
Note that we have renamed position and velocity to r and v respectively for convenience.
"""

# Initialization
# Constants
G = 6.67430e-11  # Gravitational constant
M = 6.42e23  # Mass of Mars in kg
m = 1000  # Mass of the moving body in kg (arbitrary choice)

t_max = 100
dt = 2

t_array, n = get_t(t_max, dt)


# Gravitational force function
# G,M,m are global variables that we access here!
def gravitational_force(r):
    """get the gravitational force

    Args:
        r (np array): position 3D np array

    Returns:
        np array: gravity force as another 3D np array
    """
    r_mag = np.linalg.norm(r)
    if r_mag == 0:
        return np.zeros_like(r)
    # power 3 here to normalize r!
    return (-G * M * m / (r_mag**3)) * r


# Euler method for gravity
def euler_method_gravity(r0, v0, t_max, dt):
    t_array, n = get_t(t_max, dt)
    # first dim is the time dimension, second dim is the position dimension
    r = np.zeros((n, 3))
    v = np.zeros((n, 3))
    a = np.zeros((n, 3))

    r[0] = r0
    v[0] = v0
    a[0] = gravitational_force(r[0]) / m

    for i in tqdm(range(0, n - 1)):
        r[i + 1] = r[i] + dt * v[i]
        v[i + 1] = v[i] + a[i] * dt
        a[i + 1] = gravitational_force(r[i + 1]) / m

    return r, v, a


# Verlet method
def verlet_method_gravity(r0, v0, t_max, dt):
    t_array, n = get_t(t_max, dt)
    r = np.zeros((n, 3))
    v = np.zeros((n, 3))
    a = np.zeros((n, 3))

    r[0] = r0
    v[0] = v0
    a[0] = gravitational_force(r[0]) / m

    # First step using Euler method
    r[1] = r[0] + v[0] * dt

    for i in tqdm(range(1, n - 1)):
        a[i] = gravitational_force(r[i]) / m
        r[i + 1] = 2 * r[i] - r[i - 1] + a[i] * dt**2
        v[i] = (r[i + 1] - r[i - 1]) / (2 * dt)

    # Calculate final velocity and acceleration
    v[-1] = (r[-1] - r[-2]) / dt
    a[-1] = gravitational_force(r[-1]) / m

    return r, v, a


# unlike previously, this has no general analytical solution!

# %% Problem 1
# Scenario 1: Straight down descent
r0 = np.array([0, -1e7, 0])  # 10,000 km above surface
v0 = np.array([0, 0, 0])  # Zero initial velocity
t_max = 6000  # Adjust as needed
dt = 0.01
t_1, n = get_t(t_max, dt)

# r_euler, v_euler, a_euler = euler_method_gravity(r0, v0, t_max, dt)
r_verlet, v_verlet, a_verlet = verlet_method_gravity(r0, v0, t_max, dt)

altitude = r_verlet[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(t_1, r_verlet[:, 1])
plt.title("Scenario 1: Straight Down Descent")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid(True)
plt.show()

# %%
## Scenarios 2-4 require separate plotting functions


def calculate_circular_orbit_velocity(r):
    r_mag = np.linalg.norm(r)
    v_mag = np.sqrt(G * M / r_mag)
    return v_mag


def calculate_escape_velocity(r):
    r_mag = np.linalg.norm(r)
    v_escape = np.sqrt(2 * G * M / r_mag)
    return v_escape


# only plot the first 2 axis, we assume the third axis has no height
def plot_orbit(r, title):
    plt.figure(figsize=(10, 10))
    plt.plot(r[:, 0], r[:, 1])
    plt.title(title)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


# %% Scenario 2: Circular orbit
r0_circular = np.array([0, 4e6, 0])  # 4000 km from center, in the y-axis
v_circular = calculate_circular_orbit_velocity(r0_circular)
# v_circular in the x axis
v0_circular = np.array([v_circular, 0, 0])
t_max_circular = 20000
dt_circular = 1

r_circular, v_circular, a_circular = verlet_method_gravity(
    r0_circular, v0_circular, t_max_circular, dt_circular
)
plot_orbit(r_circular, "Scenario 2: Circular Orbit")

# %%Scenario 3: Elliptical orbit
r0_elliptical = np.array([0, 4e6, 0])
v_elliptical = (
    calculate_circular_orbit_velocity(r0_elliptical) * 0.4
)  # 80% of circular orbit velocity
v0_elliptical = np.array([v_elliptical, 0, 0])
t_max_elliptical = 40000
dt_elliptical = 1

r_elliptical, v_elliptical, a_elliptical = verlet_method_gravity(
    r0_elliptical, v0_elliptical, t_max_elliptical, dt_elliptical
)
plot_orbit(r_elliptical, "Scenario 3: Elliptical Orbit")

# %% Scenario 4: hyperbolic escape
r0_hyperbolic = np.array([0, 4e6, 0])
v_escape = calculate_escape_velocity(r0_hyperbolic)
v0_hyperbolic = np.array([v_escape * 1.2, 0, 0])  # 120% of escape velocity
t_max_hyperbolic = 40000
dt_hyperbolic = 1

r_hyperbolic, v_hyperbolic, a_hyperbolic = verlet_method_gravity(
    r0_hyperbolic, v0_hyperbolic, t_max_hyperbolic, dt_hyperbolic
)
plot_orbit(r_hyperbolic, "Scenario 4: Hyperbolic Escape")
