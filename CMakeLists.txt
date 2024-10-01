cmake_minimum_required(VERSION 3.10)

project(MarsLander)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add compiler flags
add_compile_options(-O3 -Wno-deprecated-declarations)

# Set the source directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lander_cpp)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Add pybind11
find_package(pybind11 REQUIRED)

# Add original lander executable
add_executable(lander
    ${SRC_DIR}/lander.cpp
    ${SRC_DIR}/lander_graphics.cpp
    ${SRC_DIR}/lander_mechanics.cpp
    ${SRC_DIR}/autopilot.cpp
    ${SRC_DIR}/agent.cpp
    ${SRC_DIR}/main.cpp
)

# Include the source directory
target_include_directories(lander PRIVATE ${SRC_DIR})

# Link libraries
target_link_libraries(lander
    OpenGL::GL
    OpenGL::GLU
    GLUT::GLUT
)

#  BELOW CODE TO GENERATE THE PYTHON ENVIRONMENT MODULE, from our Agent class

# Add the Python module
pybind11_add_module(lander_agent_cpp
${SRC_DIR}/lander.cpp
${SRC_DIR}/lander_graphics.cpp
${SRC_DIR}/lander_mechanics.cpp
${SRC_DIR}/autopilot.cpp
${SRC_DIR}/agent.cpp
${SRC_DIR}/agent_wrapper.cpp #now include the wrapper too
${SRC_DIR}/main.cpp # we do need main.cpp as some global variables are declared here
)

# Include the source directory for the Python module
target_include_directories(lander_agent_cpp PRIVATE ${SRC_DIR})

# If your lander code uses OpenGL/GLUT, you might need to link these for the Python module as well
target_link_libraries(lander_agent_cpp PRIVATE
    OpenGL::GL
    OpenGL::GLU
    GLUT::GLUT
)

# Optionally, you can set properties for the Python module
set_target_properties(lander_agent_cpp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # Set the output name of the module
    # OUTPUT_NAME "lander_agent_cpp"
    PREFIX ""
    SUFFIX ".so"
)