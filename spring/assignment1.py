# %% ###############
# ASSIGNMENT 1 #####
####################
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_t
import time

# %%
""" VECTORIZE CODE"""

# Initialization
m = 1  # mass
k = 1  # spring constant
x0 = 0  # initial position
v0 = 1  # initial velocity

t_max = 100
dt = 0.0001


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

    for i in range(0, n - 1):
        # update x with previous x and previous v
        x[i + 1] = x[i] + dt * v[i]
        # update v with previous v and update a
        v[i + 1] = v[i] + dt * a[i]
        # a uses the current value of x
        a[i + 1] = -k * x[i + 1] / m

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

    # start from first elem
    # we always calculate the next elem stuff
    # so we go up till the second time step
    for i in range(1, n - 1):
        # use the CURRENT value of x[i] which is populated one time step earlier!
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
# Measure execution time
start_time = time.time()
x_euler, v_euler, a_euler, e_euler = euler_method(x0, v0, m, k, t_max, dt)
euler_time = time.time() - start_time

start_time = time.time()
x_verlet, v_verlet, a_verlet, e_verlet = verlet_method(x0, v0, m, k, t_max, dt)
verlet_time = time.time() - start_time

start_time = time.time()
x_analytic, v_analytic, a_analytic, e_analytic = analytical_solution(
    x0, v0, m, k, t_max, dt
)
analytical_time = time.time() - start_time

print(f"Euler method execution time: {euler_time:.4f} seconds")
print(f"Verlet method execution time: {verlet_time:.4f} seconds")
print(f"Analytical method execution time: {analytical_time:.4f} seconds")

t_array, n = get_t(t_max, dt)

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

## Question 2 is playing around with the code above, for different values of dt and t_max
# %% Question 3: now do this more rigourously
# we measure the Mean-Squared Error between the analytical solution and the verlet method, for each value of dt


def analyze_verlet_stability(t_max, dt_values):
    # Constants
    m = 1  # mass
    k = 1  # spring constant
    x0 = 0  # initial position
    v0 = 1  # initial velocity

    mse_values = []

    for dt in tqdm(dt_values):
        t_array = np.arange(0, t_max, dt)
        n = len(t_array)

        # Verlet method
        x_verlet, _, _, _ = verlet_method(x0, v0, m, k, t_max, dt)

        # Analytical solution
        x_analytic, _, _, _ = analytical_solution(x0, v0, m, k, t_max, dt)

        # Calculate MSE
        mse = np.mean((x_verlet - x_analytic) ** 2)

        mse_values.append(mse)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(dt_values, mse_values, "b-")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dt")
    plt.ylabel("Mean Squared Error")
    plt.title("Verlet Method Stability Analysis")
    plt.grid(True)

    plt.show()

    return dt_values, mse_values


t_max = 1000
dt_values = list(np.logspace(-4, 2, 50))
dt_values, mse_values = analyze_verlet_stability(t_max, dt_values)

print(f"Minimum dt tested: {min(dt_values):.6f}")
print(f"Maximum dt tested: {max(dt_values):.6f}")

"""we can see the MSE explode when dt reaches 1. beyond that, no values of MSE make sense
this is the critical value"""

# %%
