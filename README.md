# A PPO Method in Mars Lander
This project uses [Stables Baselines 3](https://stable-baselines3.readthedocs.io/en/master/#) to implement [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) to simulate spacecraft landing. Our graphics and simulation engine are implemented on `C++`, while model training and inference are in `Python`. 

## Quickstart

Install the relevant files in `requirements.txt`. To quickly view a comparison of PPO against proportional control, navigate to `src/lander_py/train.py` and run this file to train the PPO model. Then run `benchmark_agents` to plot the performance of these 2 methods.

I've abstracted away most of the C++ codebase using `Pybind11` into a module `lander_agent_cpp.so` available in the `build/` folder. This provides the `lander_agent_cpp.so` class, which is the interface where we interact with the `C++` environment.

> Unfortunately, I've not integrated the graphics engine with RL yet. This is because the `C++` codebase uses almost pure global variables and global functions, which makes encapsulation and abstraction incredibly difficult!

If you'd also like to run the graphics engine, set `render=true` and `agent_flag=false` in `main.cpp` to run the interactive graphics engine. Then build the project using the `CMake` file using the instructions below.

## Results

The proportional control algorithim did much better than PPO, unfortunately due to the problem of sparse rewards the agent was unable to successfully get positive signal and learn how to land. While during training steps were taken to encourage exploration (like increasing the entropy coefficient, see Equation 9 of the original PPO paper, and decreasing the penalty on each time step), the model was still unable to learn the landing algorithim.

Our observation (state) space consists of the velocity vector $V$, the position vector $R$, fuel left, altitude from Mars' surface $H$, and the climb speed $V*e_r$. (Note that the climb speed is a signed scalar, and the descent rate is the negative of the climb speed). Our action space consists of the throttle value  $\in [0,1]$. 

Most of the time, PPO learnt a constant value somewhere in the range of $0.3-0.8$ and stuck to it throughout.

<image src=base.png width=800>

This is despite during training, verifying that PPO is sampling from the *entire action space*.

### Reasoning

I suspect the biggest 2 problems leading to this is the problem of sparse rewards and compute. This is the reward function:

```
if landeded safely:
    return 100.0
else if crashed:
    return -100.0
else:
    return -1.0
```

As you can see, we only ever obtain rewards at the end of the episode, which makes getting signal to update the weights correctly extremely difficult. Using a simulation timestep $ \delta t$ of $0.1$, one episode takes on average $1000-4000$ timesteps hence $100-400$ seconds to complete. This means the agent has experienced on average thousands of interactions before it ever receives a signal. Hence, we need to do training for much longer in order to learn.

> We introduce a slight negative reward on every step to prevent the agent from being "lazy" and artifically prolonging the simulation

To address this issue, I tried modifying the reward function to introduce varying rewards throughout the episode:

```
if landeded safely:
    return 100.0
else if crashed:
    return -100.0
else:
    if altitude<1000:
        return -climb_speed**2
    return scale_factor * abs(climb_speed)
```

We can see regularly, the agent is penalised slightly (primarily to address the lazy issue) but also penalizing having too large of speeds.

> We use absolute of the climb speed here to prevent the agent from cheating by going in the opposite direction (upwards) and obtaining positive reward this way

Further work is required for this environment, via better reward shaping or simply training for longer.


## Repository structure

```
.
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── src
│   ├── assignment2.py
│   ├── lander_cpp
│   │   ├── agent.cpp
│   │   ├── agent_wrapper.cpp
│   │   ├── autopilot.cpp
│   │   ├── lander.cpp
│   │   ├── lander.h
│   │   ├── lander_graphics.cpp
│   │   ├── lander_mechanics.cpp
│   │   └── main.cpp
│   ├── lander_py
│   │   ├── benchmark_agents.py
│   │   ├── lander_env.py
│   │   ├── test_lander_agent_cpp.py
│   │   ├── test_lander_env.py
│   │   └── train.py
│   └── spring
│       ├── assignment1.py
│       ├── assignment3.cpp
│       ├── spring.cpp
│       ├── spring.py
│       └── visualize_cpp.py
└── utils.py
```

`src/lander_cpp/` contains the `OpenGL` graphics method in `lander_graphics` and core numerical simulation methods in `lander_mechanics`. `agent.cpp` contains our `Agent` interface with Python, while `agent_wrapper` uses `Pybind` to wrap around our `Agent` class so that we can call them in Python. `autopilot` contains the C++ proportional controller implementation. `main.cpp` contains the main function to display the graphics engine and run an agent without using the engine.

> As basically all functions and variables are global, it's very difficult to containerize methods and integrate our `Agent` class seamlessly into the graphics engine. Major refactoring is needed to containerize `update_lander_state()`, `autopilot()`, `numerical_dynamics()`, `update_visualization()` by taking in our `Agent` class as a parameter, or accessing variables locally

In `src/lander_py/`, this contains code to train and do evaluate our agents. `test_lander_agent_cpp` tests whether `pybind11` has successfully translated all our methods, while `test_lander_env` tests whether our `gymnasium` environment is working as intended.

`spring/` contains some of the assignment code for simulating simple harmonic motion.

## Compiling the project

We use `Cmake` to compile our lander project (but not for spring). Make sure you have `Cmake`, `Pybind11`,`OpenGL` and `GLUT` installed for the `C++` projects.

1. Ensure `CMakeLists.txt` is in the root directory
2. Create a build directory in the root: `mkdir build && cd build`
3. Run CMake: `cmake ..`
4. Build the project: `make`

> After compiling, you can do `import build.lander_agent_cpp as lander_agent_cpp` to use modules from our C++ `Agent` class, and run C++ modules directly from Python

### Running Files

- After building, we can run `./lander` from `build` folder, to run `main.cpp`
- To use the modules in `lander_agent_cpp.so`, to import from this file like a regular Python files containing classes

### OpenGL on WSL

To set up OpenGL to display correctly on WSL, follow these [instructions](https://gist.github.com/Mluckydwyer/8df7782b1a6a040e5d01305222149f3c) to setup VcXsrv server.
- use VcXsrv to open a new server
- set `export DISPLAY=[Your IP Address]:0` where `0` means your screen number. You should've set this to zero previously.
- change `LIBGL_ALWAYS_INDIRECT=1`


## Refactoring of Global Variables

Previously, the repo used a flag to selectively declare variables. All variables are now declared as `extern` in `lander.h`, and fully declared at the start of the file where they are first used.
- Ideally we would use singleton classes here or encapsulated classes, but there's simply too many variables interacting with each other

- added a `render` variable that allows me to run simulations without `GLUT`
    - currently, only `render=true, agent_flag=false` and `render=false,agent_flag=true` are supported

## Tips and Tricks

- Making model (actor, critic smaller)
- Changing reward function to be like energy
    - maybe try inverse of radius, and exponential of energy next?
- PPO Params
    - Increasing entropy
    - Batch size
    - num epochs
    - Learning rate
- Normalizing action space
- Normalizing rewards to be positive
    - Agent can't cheat by trying to die as fast as possible

### Normalization

- Really crucial to normalize observations space, especially when observations have a large range
    - the `gym` package's `NormalizeObservations` wrapper environment is key for this!
    - keeps a running mean of the observation, but note that the way it does this is by wrapping around the `step` function
    - this is why for plotting, you only plot stuff defined in the `step` function for info
        - important things like the un-normalized observations
        - the stuff coming out of `step` should only be what the model sees

- You can also normalize rewards using the `NormalizeReward` wrapper environment which keep the exponential moving average having a fixed variance
    - this is extremely helpful as I don't need to manually set constants to change my reward

- I also "normalized" the actions space by doing a linear transformation from the original action space of $throttle \in [0,1]$ to $[-1,1]$ so that the model can learn better

- So currently, our base `LanderEnv` accepts transformed actions and outputs untransformed observations



