cmake_minimum_required(VERSION 3.14)
project(subdomain_B_solver)

find_package(deal.II REQUIRED HINTS ${DEAL_II_DIR})

add_executable(solver main.cpp)
target_link_libraries(solver deal_II)
