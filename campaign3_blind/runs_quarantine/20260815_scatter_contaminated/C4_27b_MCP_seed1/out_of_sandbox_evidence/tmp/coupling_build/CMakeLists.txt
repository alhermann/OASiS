cmake_minimum_required(VERSION 3.13)

find_package(deal.II 9.0 REQUIRED HINTS /home/alexander/dealii/build $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(heat_iface_dealii_transient CXX)

add_executable(heat_iface_dealii_transient heat_iface_dealii_transient.cc)
deal_ii_setup_target(heat_iface_dealii_transient)
