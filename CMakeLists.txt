project(macrosim)

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_SHARED_LIBS "Build shared library." ON)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
SET(ITOM_DIR "" CACHE PATH "base path to itom") 
set(PTX_DIR "${CMAKE_BINARY_DIR}/ptx" CACHE PATH "Path to where the samples look for the PTX code.")

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/macrosim_tracer/CMake"
  ${CMAKE_MODULE_PATH}
  )

ADD_SUBDIRECTORY(macrosim_tracer)
ADD_SUBDIRECTORY(macrosim_gui)