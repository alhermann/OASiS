cmake_minimum_required(VERSION 3.16)
project(coupled_heat_transfer)

find_package(deal.II REQUIRED HINTS ${DEAL_II_DIR})

add_executable(dealii_solver dealii_solver.cpp)
target_link_libraries(dealii_solver dealii::dealii)
