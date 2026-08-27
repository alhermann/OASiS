cmake_minimum_required(VERSION 3.10)
project(elasticity_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force deal.II path - must be set before find_package
set(DEAL_II_DIR /home/alexander/dealii/build CACHE PATH "deal.II directory" FORCE)

find_package(deal.II REQUIRED HINTS ${DEAL_II_DIR})

add_executable(elasticity_solver elasticity_solver.cpp)

target_link_libraries(elasticity_solver ${DEAL_II_TARGET})
