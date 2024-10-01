# lander
A deep RL method for spacecraft landing

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
