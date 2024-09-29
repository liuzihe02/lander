# %% ###############
# ASSIGNMENT 2 #####
####################
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
""" VECTORIZE CODE"""

# Initialization
m = 1  # mass
k = 1  # spring constant
x0 = 0  # initial position
v0 = 1  # initial velocity

t_max = 100
dt = 2


def get_t(t_max, dt):
    t_array = np.arange(0, t_max, dt)
    n = len(t_array)
    return t_array, n


t_array, n = get_t(t_max, dt)


# Euler method (vectorized)
def euler_method(x0, v0, m, k, t_max, dt):
    t_array, n = get_t(t_max, dt)

    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    e = np.zeros(n)
    # start off initialization
    x[0] = x0
    v[0] = v0
    a[0] = -k * x[0] / m
    e[0] = 0.5 * k * x[0] ** 2 + 0.5 * m * v[0] ** 2

    for i in range(1, n):
        # a uses the PREVIOUS value of x
        a[i] = -k * x[i - 1] / m
        # update x with previous x and previous v
        x[i] = x[i - 1] + dt * v[i - 1]
        # update v with previous v and update a
        v[i] = v[i - 1] + dt * a[i]
        # update e with all the current values
        e[i] = 0.5 * k * x[i] ** 2 + 0.5 * m * v[i] ** 2

    return x, v, a, e


# Verlet method (vectorized)
def verlet_method(x0, v0, m, k, t_max, dt):
    t_array, n = get_t(t_max, dt)

    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    e = np.zeros(n)

    x[0] = x0
    v[0] = v0
    a[0] = -k * x[0] / m
    e[0] = 0.5 * k * x[0] ** 2 + 0.5 * m * v[0] ** 2

    # Calculate x[1] using Euler method for the first step
    x[1] = x[0] + v[0] * dt

    # from second elem up till the second last elem. Note that at position i, we calculate x[i+1] and [i] for everything else!
    # so we start with the second position, where we calculate x[3]
    for i in range(1, n - 1):
        # use the current value of x[i] which is populated one time step earlier!
        a[i] = -k * x[i] / m
        x[i + 1] = 2 * x[i] - x[i - 1] + a[i] * dt**2
        v[i] = (x[i + 1] - x[i - 1]) / (2 * dt)
        e[i] = 0.5 * k * x[i] ** 2 + 0.5 * m * v[i] ** 2

    # Calculate final velocity and acceleration
    a[-1] = -k * x[-1] / m
    # just use the last and current x for v
    v[-1] = (x[-1] - x[-2]) / dt
    e[-1] = 0.5 * k * x[-1] ** 2 + 0.5 * m * v[-1] ** 2

    return x, v, a, e


# unlike previously, the analytical solution needs more parameters here than just n
def analytical_solution(x0, v0, m, k, t_max, dt):
    t_array, n = get_t(t_max, dt)

    omega = np.sqrt(k / m)
    A = np.sqrt(x0**2 + (v0 / omega) ** 2)
    phi = np.arctan2(-v0 / (omega * A), x0 / A)

    x = A * np.cos(omega * t_array + phi)
    v = -A * omega * np.sin(omega * t_array + phi)
    a = -A * omega**2 * np.cos(omega * t_array + phi)
    e = np.full(n, 0.5 * k * A**2)  # Energy is constant

    return x, v, a, e


# Run simulations
x_euler, v_euler, a_euler, e_euler = euler_method(x0, v0, m, k, t_max, dt)
x_verlet, v_verlet, a_verlet, e_verlet = verlet_method(x0, v0, m, k, t_max, dt)
x_analytic, v_analytic, a_analytic, e_analytic = analytical_solution(
    x0, v0, m, k, t_max, dt
)

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.title("Euler Method")
plt.plot(t_array, x_euler, label="x (m)")
plt.plot(t_array, v_euler, label="v (m/s)")
plt.plot(t_array, a_euler, label="a (m/s²)")
plt.plot(t_array, e_euler, label="energy")
plt.xlabel("time (s)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Verlet Method")
plt.plot(t_array, x_verlet, label="x (m)")
plt.plot(t_array, v_verlet, label="v (m/s)")
plt.plot(t_array, a_verlet, label="a (m/s²)")
plt.plot(t_array, e_verlet, label="energy")
plt.xlabel("time (s)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Analytical Solution")
plt.plot(t_array, x_analytic, label="x (m)")
plt.plot(t_array, v_analytic, label="v (m/s)")
plt.plot(t_array, a_analytic, label="a (m/s²)")
plt.plot(t_array, e_analytic, label="energy")
plt.xlabel("time (s)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
