#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lander.h"

namespace py = pybind11;

class PyAgent : public Agent
{
public:
    using Agent::Agent; // Inherit the constructor

    // Override virtual functions
    void reset() override
    {
        PYBIND11_OVERRIDE(void, Agent, reset);
    }

    std::vector<double> getState() override
    {
        PYBIND11_OVERRIDE(std::vector<double>, Agent, getState);
    }

    void step() override
    {
        PYBIND11_OVERRIDE(void, Agent, step);
    }

    bool isDone() const override
    {
        PYBIND11_OVERRIDE(bool, Agent, isDone);
    }

    double getReward() const override
    {
        PYBIND11_OVERRIDE(double, Agent, getReward);
    }
};

PYBIND11_MODULE(lander_agent_cpp, m)
{
    // this stuff in strings is what well actually call our functions in python
    // the top string is the name of the class
    py::class_<Agent, PyAgent>(m, "Agent")
        .def(py::init<>())
        .def("reset", &Agent::reset)
        .def("get_state", &Agent::getState)
        .def("step", &Agent::step)
        .def("is_done", &Agent::isDone)
        .def("get_reward", &Agent::getReward);
}