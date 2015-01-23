
SET (target_name PropagationMath)

find_package(CUDA 4)

if (BUILD_TARGET64)
   SET(bit_dest "")
   set( CUDA_64_BIT_DEVICE_CODE  ON)
else (BUILD_TARGET64)
   SET(bit_dest "32")
endif (BUILD_TARGET64)

include_directories(
					${CUDA_INCLUDE_DIRS}
					"${CMAKE_CURRENT_SOURCE_DIR}/include")

# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

# set compiler flags
SET(CUDA_NVCC_FLAGS "-arch;sm_20" "--maxrregcount=32" "--ptxas-options=-v")
#SET(CUDA_NVCC_FLAGS "-arch;sm_20")
					
cuda_add_library(${target_name} STATIC
  src/kernel.cu
  include/kernel.h
  src/PropagationMath.cpp
  include/PropagationMath.h
  )
  
set_property(TARGET ${target_name} PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

#try to create all pathes
#first try to create the directory
ADD_CUSTOM_COMMAND(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${MacroSim_INSTALL_DIR}lib${bit_dest}"
)

ADD_CUSTOM_COMMAND(TARGET ${target_name} POST_BUILD                 # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  			   # which executes "cmake - E copy_if_different..."
        "$<TARGET_FILE:${target_name}>"      										   # <--this is in-file
        "${MacroSim_INSTALL_DIR}lib${bit_dest}"                 				               # <--this is out-file path
)