# lander
This project uses [Stables Baselines 3](https://stable-baselines3.readthedocs.io/en/master/#) to implement [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) to simulate spacecraft landing. Our graphics and simulation engine are implemented on `C++` while training and inference are in `Python`. 

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

## Reposity structure

```


## Notes

### OpenGL on WSL

To set up OpenGL to display correctly on WSL, follow these [instructions](https://gist.github.com/Mluckydwyer/8df7782b1a6a040e5d01305222149f3c) to setup VcXsrv server.
- use VcXsrv to open a new server
- set `export DISPLAY=[Your IP Address]:0` where `0` means your screen number. You should've set this to zero previously.
- change `LIBGL_ALWAYS_INDIRECT=1`

### Makefile

Note that currently, the make files only compiles lander projects.

## Main Changes

Refactoring of global variables. All variables are declared as `extern` in lander.h, and declared at the start of the file when they are first used.
- I know this is bad practice, but I have no choice as I need to modularise the code
- Ideally we would use singleton classes here or OOP to encapsulate classes
    - I don't think its a good idea to have so many global variables, but theres nothing I can do about it.
    - This is causing me alot of headaches.


- added a `render` variable that allows me to run simulations without GLUT
    - The functions starting with `mech` are rewritten by me to allow to run without GLUT

- `lander.cpp`
    - changes like in assignment
- `autopilot.cpp`
- `main.cpp`
    - 

## Extra Requirements

### `cmake`

1. Place `CMakeLists.txt` in the root directory of your project
2. Create a build directory in the root: `mkdir build && cd build`
3. Run CMake: `cmake ..`
4. Build the project: `make`

### `pybind`
Note that you need to install `pybind` in linux to wrap your C++ classes around Python wrappers. Then, you can call C++ classes from Python as if you were running them in C++. This is needed as the codebase is in C++, but we want to do training in Python.
- Due to our functions in C++ being all global currently, this complicates things

### `libtorch`

When compiling and linking with LibTorch, you need to add the neccessary headers

### Running Files

- After building, we can run `./lander` from `build` folder, to run `main.cpp`
- To use the modules in `lander_agent_cpp.so`, to import from this file like a regular Python files containing classes