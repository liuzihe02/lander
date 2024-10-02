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
    std::vector<double> reset(std::vector<double> init_conditions) override
    {
        PYBIND11_OVERRIDE(std::vector<double>, Agent, reset, init_conditions);
    }

    void update(std::tuple<double> actions) override
    {
        PYBIND11_OVERRIDE(void, Agent, update, actions);
    }

    // std::tuple<double> getActions() override
    // {
    //     PYBIND11_OVERRIDE(std::tuple<double>, Agent, getActions);
    // }

    std::vector<double> getState() override
    {
        PYBIND11_OVERRIDE(std::vector<double>, Agent, getState);
    }

    bool isLanded() const override
    {
        PYBIND11_OVERRIDE(bool, Agent, isLanded);
    }

    bool isCrashed() const override
    {
        PYBIND11_OVERRIDE(bool, Agent, isCrashed);
    }
};

PYBIND11_MODULE(lander_agent_cpp, m)
{
    py::class_<Agent, PyAgent>(m, "PyAgent")
        .def(py::init<>())
        .def("reset", &Agent::reset)
        .def("update", &Agent::update)
        //.def("get_actions", &Agent::getActions)
        .def("get_state", &Agent::getState)
        .def("is_landed", &Agent::isLanded)
        .def("is_crashed", &Agent::isCrashed);
}