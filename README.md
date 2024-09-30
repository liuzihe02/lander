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

- lander.cpp
- autopilot.cpp
- added a render variable that allows me to run simulations without GLUT
