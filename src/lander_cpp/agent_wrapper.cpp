// lander_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lander.h"

namespace py = pybind11;

PYBIND11_MODULE(lander_cpp, m)
{
    py::class_<Lander>(m, "Lander")
        .def(py::init<>())
        .def("reset", &Lander::reset)
        .def("get_state", &Lander::getState)
        .def("step", &Lander::step)
        .def("is_done", &Lander::isDone)
        .def("get_reward", &Lander::getReward);
}