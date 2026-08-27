cmake_minimum_required(VERSION 3.13.4)
project(elast_iface_dealii)

find_package(deal.II 9.2.0 REQUIRED HINTS $ENV{DEAL_II_DIR} /home/alexander/dealii/build)
deal_ii_initialize_cached_variables()

add_executable(elast_iface_dealii elast_iface_dealii.cc)
deal_ii_setup_target(elast_iface_dealii)
