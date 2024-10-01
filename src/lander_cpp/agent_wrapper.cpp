#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lander.h"
#include <tuple>
#include <vector>

namespace py = pybind11;

class PyAgent : public Agent
{
public:
    using Agent::Agent; // Inherit the constructor

    // Override virtual functions
    vector<double> reset() override
    {
        PYBIND11_OVERRIDE(vector<double>, Agent, reset);
    }

    std::tuple<vector<double>, double, bool> step(std::tuple<double> actions) override
    {
        PYBIND11_OVERRIDE(std::tuple<vector<double>, double, bool>, Agent, step, actions);
    }

    std::tuple<double> getActions() override
    {
        PYBIND11_OVERRIDE(std::tuple<double>, Agent, getActions);
    }

    std::vector<double> getState() override
    {
        PYBIND11_OVERRIDE(std::vector<double>, Agent, getState);
    }
};

PYBIND11_MODULE(lander_agent_cpp, m)
{
    // this stuff in strings is what we'll actually call our functions in python
    // the top string is the name of the class
    py::class_<Agent, PyAgent>(m, "Agent")
        .def(py::init<>())
        .def("reset", &Agent::reset)
        .def("get_state", &Agent::getState)
        .def("step", &Agent::step);
}