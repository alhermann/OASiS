cmake_minimum_required(VERSION 3.16)
project(fem_solver)

find_package(deal.II REQUIRED HINTS /home/alexander/dealii/build)

add_executable(fem_solver fem_solver.cpp)

target_link_libraries(fem_solver dealii::dealii)
