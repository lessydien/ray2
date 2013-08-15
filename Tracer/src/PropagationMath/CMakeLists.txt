
SET (target_name PropagationMath)

find_package(FFTW 3)
find_package(CUDA 4)

if (BUILD_TARGET64)
   SET(bit_dest "")
   set( CUDA_64_BIT_DEVICE_CODE  ON)
else (BUILD_TARGET64)
   SET(bit_dest "32")
endif (BUILD_TARGET64)

include_directories(${FFTW_INCLUDE_DIR}
					${CUDA_INCLUDE_DIRS}
					"${CMAKE_CURRENT_SOURCE_DIR}/include")

# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

# set compiler flags
SET(CUDA_NVCC_FLAGS "-arch;sm_20" "--maxrregcount=32")
					
cuda_add_library(${target_name} STATIC
  src/kernel.cu
  include/kernel.h
  src/PropagationMath.cpp
  include/PropagationMath.h
  )
  
set_property(TARGET ${target_name} PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

SET(DebugLib "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug/PropagationMathd.lib")
SET(ReleaseLib "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/PropagationMath.lib")
ADD_CUSTOM_COMMAND(TARGET ${target_name} POST_BUILD            # Adds a post-build event 
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  			   # execute "cmake - E copy_if_different..."
	"${DebugLib}"	      										   # <--this is in-file
	"${MacroSim_INSTALL_DIR}lib${bit_dest}"                   # <--this is out-file path
)
ADD_CUSTOM_COMMAND(TARGET ${target_name} POST_BUILD            # Adds a post-build event 
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  			   # execute "cmake - E copy_if_different..."
	"${ReleaseLib}"	      										   # <--this is in-file
	"${MacroSim_INSTALL_DIR}lib${bit_dest}"                   # <--this is out-file path
)