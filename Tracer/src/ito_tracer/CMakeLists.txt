SET (target_name ITO-MacroSim)

#if(GLUT_FOUND AND OPENGL_FOUND)
#  include_directories(${GLUT_INCLUDE_DIR})
#  add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

if (BUILD_TARGET64)
   SET(bit_dest "")
else (BUILD_TARGET64)
   SET(bit_dest "32")
endif (BUILD_TARGET64)

# Add the path to matlab headers headers and Lex Headers to our include paths.
include_directories(
  ${MacroSim_INCLUDE}
  ${Flex_INCLUDE}
  )

# add matlab libraries
#LINK_LIBRARIES( ${matlab_eng_LIBRARY} ${matlab_mx_LIBRARY} ${matlab_mat_LIBRARY} )

# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
OPTIX_add_sample_executable( ITO-MacroSim
  #sample2.cpp
  
	pugiconfig.hpp
	pugixml.cpp
	pugixml.hpp

  myUtil.h
  myUtil.cpp

  MacroSimLib.cpp
  MacroSimLib.h
  CompoundGeometry.h
  CompoundGeometry.cpp
  SphericalLense.h
  SphericalLense.cpp
  
  Interpolator.cpp
  Interpolator.h
  Interpolator_host_device.h

  FlexZemax.cpp
  FlexZemax.h
  RayField.cpp
  RayField.h
  GeometricRayField.cpp
  GeometricRayField.h
  PathTracingRayField.cpp
  PathTracingRayField.h
  differentialRayTracing/DiffRayfield_Freeform.cpp
  differentialRayTracing/DiffRayfield_Freeform.h
  GaussBeamRayField.h
  GaussBeamRayField.cpp
  randomGenerator.h
  randomGenerator.cpp
  DiffRayField.h
  DiffRayField.cpp
  DiffRayField_RayAiming.h
  DiffRayField_RayAiming.cpp
  DiffRayField_RayAiming_Holo.h
  DiffRayField_RayAiming_Holo.cpp
  DiffRayField_RayAiming_Holo_test.cpp
  DiffRayField_RayAiming_Holo_test.h
  inputOutput.h
  inputOutput.cpp
  ScalarLightField.h
  ScalarLightField.cpp
  ScalarPlaneField.h
  ScalarPlaneField.cpp
  ScalarSphericalField.h
  ScalarSphericalField.cpp
  ScalarGaussianField.h
  ScalarGaussianField.cpp
  ScalarUserField.h
  ScalarUserField.cpp

  VectorLightField.h
  VectorLightField.cpp
  Field.h
  Field.cpp
  FieldLib.h
  FieldLib.cpp
  Field_Stack.h
  Field_Stack.cpp
  FieldParams.h
  IntensityField.h
  IntensityField.cpp
  PhaseSpaceField.h
  PhaseSpaceField.cpp
  IntensityField_Stack.h
  IntensityField_Stack.cpp
  Converter.cpp
  Converter.h
  converterMath.h
  ConverterMath.cpp
  Detector.cpp
  Detector.h
  Detector_Field.cpp
  Detector_Field.h
  Detector_PhaseSpace.cpp
  Detector_PhaseSpace.h
  Detector_Intensity.cpp
  Detector_Intensity.h
  Detector_Raydata.cpp
  Detector_Raydata.h
  DetectorParams.h
  DetectorLib.h
  DetectorLib.cpp
  PlaneSurface.cpp
  PlaneSurface.h
  PlaneSurface_intersect.h
  differentialRayTracing/PlaneSurface_DiffRays.cpp
  differentialRayTracing/PlaneSurface_DiffRays.h
  differentialRayTracing/PlaneSurface_DiffRays_intersect.h
  SinusNormalSurface.cpp
  SinusNormalSurface.h
  SinusNormalSurface_intersect.h
  parabolicSurface.cpp
  parabolicSurface.h
  parabolicSurface.cu
  parabolicSurface_intersect.h
  differentialRayTracing/SinusNormalSurface_DiffRays.cpp
  differentialRayTracing/SinusNormalSurface_DiffRays.h
  differentialRayTracing/SinusNormalSurface_DiffRays_intersect.h
  Parser.cpp
  Parser.h
  Parser_XML.cpp
  Parser_XML.h
  AsphericalSurface.cpp
  AsphericalSurface.h
  AsphericalSurface_intersect.h
  differentialRayTracing/AsphericalSurface_DiffRays.cpp
  differentialRayTracing/AsphericalSurface_DiffRays.h
  differentialRayTracing/AsphericalSurface_DiffRays_intersect.h
  CylPipe.cpp
  CylPipe.h
  CylPipe_intersect.h
  CylLenseSurface.cpp
  CylLenseSurface.h
  CylLenseSurface_intersect.h
  differentialRayTracing/CylPipe_DiffRays.cpp
  differentialRayTracing/CylPipe_DiffRays.h
  differentialRayTracing/CylPipe_DiffRays_intersect.h
  Coating.cpp
  Coating.h
  Coating_hit.h
  differentialRayTracing/Coating_DiffRays.cpp
  differentialRayTracing/Coating_DiffRays.h
  differentialRayTracing/Coating_DiffRays_hit.h
  Coating_NumCoeffs.cpp
  Coating_NumCoeffs.h
  Coating_NumCoeffs_hit.h
  Coating_DispersiveNumCoeffs.cpp
  Coating_DispersiveNumCoeffs.h
  Coating_DispersiveNumCoeffs_hit.h
  Coating_FresnelCoeffs.cpp
  Coating_FresnelCoeffs.h
  Coating_FresnelCoeffs_hit.h
  differentialRayTracing/Coating_NumCoeffs_DiffRays.cpp
  differentialRayTracing/Coating_NumCoeffs_DiffRays.h
  differentialRayTracing/Coating_NumCoeffs_DiffRays_hit.h
  ApertureStop.cpp
  ApertureStop.h
  ApertureStop_intersect.h
  differentialRayTracing/ApertureStop_DiffRays.cpp
  differentialRayTracing/ApertureStop_DiffRays.h
  differentialRayTracing/ApertureStop_DiffRays_intersect.h
  ConePipe.cpp
  ConePipe.h
  ConePipe_intersect.h
  differentialRayTracing/ConePipe_DiffRays.cpp
  differentialRayTracing/ConePipe_DiffRays.h
  differentialRayTracing/ConePipe_DiffRays_intersect.h
  complex.cpp
  complex.h
  IdealLense.cpp
  IdealLense.h
  IdealLense_Intersect.h
  differentialRayTracing/IdealLense_DiffRays.cpp
  differentialRayTracing/IdealLense_DiffRays.h
  differentialRayTracing/IdealLense_DiffRays_Intersect.h
  Geometry.cpp
  Geometry.h
  Geometry_Intersect.h
  GeometryGroup.cpp
  GeometryGroup.h
  GeometryLib.cpp
  GeometryLib.h
  differentialRayTracing/GeometryLib_DiffRays.h
  Group.cpp
  Group.h
  Material.cpp
  Material.h
  Material_hit.h
  MaterialPathTraceSource.cpp
  MaterialPathTraceSource.h
  MaterialPathTraceSource_hit.h
  differentialRayTracing/Material_DiffRays_hit.h
  differentialRayTracing/Material_DiffRays.cpp
  differentialRayTracing/Material_DiffRays.h
  MaterialLib.cpp
  MaterialLib.h
  differentialRayTracing/MaterialLib_DiffRays.h
  differentialRayTracing/Material_DiffRays_Lib.h
  CoatingLib.cpp
  CoatingLib.h
  differentialRayTracing/CoatingLib_DiffRays.h
  ScatterLib.cpp
  ScatterLib.h
  differentialRayTracing/ScatterLib_DiffRays.h
  MaterialFilter.cpp
  MaterialFilter.h
  MaterialFilter_hit.h
  
  MaterialIdealLense.cpp
  MaterialIdealLense.h
  MaterialIdealLense_hit.h
  differentialRayTracing/MaterialIdealLense_DiffRays.cpp
  differentialRayTracing/MaterialIdealLense_DiffRays.h
  differentialRayTracing/MaterialIdealLense_DiffRays_hit.h
  MaterialReflecting.cpp
  MaterialReflecting.h
  MaterialReflecting_hit.h
  differentialRayTracing/MaterialReflecting_DiffRays.cpp
  differentialRayTracing/MaterialReflecting_DiffRays.h
  differentialRayTracing/MaterialReflecting_DiffRays_hit.h
  MaterialReflecting_CovGlass.cpp
  MaterialReflecting_CovGlass.h
  MaterialReflecting_CovGlass_hit.h
  MaterialRefracting.cpp
  MaterialRefracting.h
  MaterialRefracting_hit.h
  differentialRayTracing/MaterialRefracting_DiffRays.cpp
  differentialRayTracing/MaterialRefracting_DiffRays.h
  differentialRayTracing/MaterialRefracting_DiffRays_hit.h
  MaterialDiffracting.cpp
  MaterialDiffracting.h
  MaterialDiffracting_hit.h
  differentialRayTracing/MaterialDiffracting_DiffRays.cpp
  differentialRayTracing/MaterialDiffracting_DiffRays.h
  differentialRayTracing/MaterialDiffracting_DiffRays_hit.h
  MaterialAbsorbing.cpp
  MaterialAbsorbing.h
  MaterialAbsorbing_hit.h
  differentialRayTracing/MaterialAbsorbing_DiffRays.cpp
  differentialRayTracing/MaterialAbsorbing_DiffRays.h
  differentialRayTracing/MaterialAbsorbing_DiffRays_hit.h
  MaterialDOE.cpp
  MaterialDOE.h
  MaterialDOE_hit.h
  MaterialLinearGrating1D.cpp
  MaterialLinearGrating1D.h
  MaterialLinearGrating1D_hit.h
  differentialRayTracing/MaterialLinearGrating1D_DiffRays.cpp
  differentialRayTracing/MaterialLinearGrating1D_DiffRays.h
  differentialRayTracing/MaterialLinearGrating1D_DiffRays_hit.h
  Scatter.cpp
  Scatter.h
  Scatter_hit.h
  differentialRayTracing/Scatter_DiffRays.cpp
  differentialRayTracing/Scatter_DiffRays.h
  differentialRayTracing/Scatter_DiffRays_hit.h
  Scatter_TorranceSparrow1D.cpp
  Scatter_TorranceSparrow1D.h
  Scatter_TorranceSparrow1D_hit.h
  Scatter_TorranceSparrow2D.cpp
  Scatter_TorranceSparrow2D.h
  Scatter_TorranceSparrow2D_hit.h
  Scatter_TorranceSparrow2D_PathTrace.cpp
  Scatter_TorranceSparrow2D_PathTrace.h
  Scatter_TorranceSparrow2D_PathTrace_hit.h
  
  differentialRayTracing/Scatter_TorranceSparrow1D_DiffRays.cpp
  differentialRayTracing/Scatter_TorranceSparrow1D_DiffRays.h
  differentialRayTracing/Scatter_TorranceSparrow1D_DiffRays_hit.h
  Scatter_DoubleCauchy1D.cpp
  Scatter_DoubleCauchy1D.h
  Scatter_DoubleCauchy1D_hit.h
  Scatter_DispersiveDoubleCauchy1D.cpp
  Scatter_DispersiveDoubleCauchy1D.h
  Scatter_DispersiveDoubleCauchy1D_hit.h
  differentialRayTracing/Scatter_DoubleCauchy1D_DiffRays.cpp
  differentialRayTracing/Scatter_DoubleCauchy1D_DiffRays.h
  differentialRayTracing/Scatter_DoubleCauchy1D_DiffRays_hit.h
  Scatter_Lambert2D.cpp
  Scatter_Lambert2D.h
  Scatter_Lambert2D_hit.h
  differentialRayTracing/Scatter_Lambert2D_DiffRays.cpp
  differentialRayTracing/Scatter_Lambert2D_DiffRays.h
  differentialRayTracing/Scatter_Lambert2D_DiffRays_hit.h
  microLensArray.cpp
  microLensArray.h
  microLensArraySurface.cpp
  microLensArraySurface.h
  microLensArraySurface_intersect.h
  microLensArraySurface.cu
  SphericalSurface.cpp
  SphericalSurface.h
  SphericalSurface_Intersect.h
  differentialRayTracing/SphericalSurface_DiffRays.cpp
  differentialRayTracing/SphericalSurface_DiffRays.h
  differentialRayTracing/SphericalSurface_DiffRays_Intersect.h
  SimAssistant.cpp
  SimAssistant.h
  SimAssistant_SingleSim.cpp
  SimAssistant_SingleSim.h
  SimAssistant_ParamSweep.cpp
  SimAssistant_ParamSweep.h
  SimAssistant_Layout.cpp
  SimAssistant_Layout.h
  SimAssistantLib.h
  rayTracingMath.h
  wavefrontIn.cpp
  wavefrontIn.h
  rayData.h
  GlobalConstants.h
  TopObject.h
  IlluminationPupilParams.h
  PupilParams.h
  PupilLib.h
  Pupil_RayAim_FarField.cpp
  Pupil_RayAim_FarField.h
  Pupil_RayAim_FarField_aim.h
  Pupil.cpp
  Pupil.h
  Pupil_aim.h
  
  
  missFunction.cu
  hitFunctionAbsorbing.cu
  differentialRayTracing/hitFunctionAbsorbing_DiffRays.cu
  hitFunctionAbsorbing_Scatter_TorranceSparrow1D.cu
  differentialRayTracing/hitFunctionAbsorbing_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  hitFunctionAbsorbing_CoatNumCoeffs.cu
  hitFunctionAbsorbing_CoatFresnelCoeffs.cu
  differentialRayTracing/hitFunctionAbsorbing_DiffRays_CoatNumCoeffs_DiffRays.cu
  hitFunctionAbsorbing_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  differentialRayTracing/hitFunctionAbsorbing_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu
  hitFunctionAbsorbing_Scatter_Lambert2D.cu
  differentialRayTracing/hitFunctionAbsorbing_DiffRays_Scatter_Lambert2D_DiffRays.cu
  hitFunctionAbsorbing_Scatter_DoubleCauchy1D_CoatNumCoeffs.cu
  
  hitFunctionRefracting.cu
  differentialRayTracing/hitFunctionRefracting_DiffRays.cu
  hitFunctionRefracting_Scatter_TorranceSparrow1D.cu
  differentialRayTracing/hitFunctionRefracting_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  hitFunctionRefracting_CoatNumCoeffs.cu
  differentialRayTracing/hitFunctionRefracting_DiffRays_CoatNumCoeffs_DiffRays.cu
  hitFunctionRefracting_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  differentialRayTracing/hitFunctionRefracting_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu
  
  hitFunctionFilter.cu
  hitFunctionFilter_CoatNumCoeffs.cu
  
  hitFunctionDOE.cu
  
  hitFunctionDiffracting.cu
  differentialRayTracing/hitFunctionDiffracting_DiffRays.cu
  
  hitFunctionReflecting.cu
  differentialRayTracing/hitFunctionReflecting_DiffRays.cu
  hitFunctionReflecting_Scatter_TorranceSparrow1D.cu
  differentialRayTracing/hitFunctionReflecting_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  hitFunctionReflecting_CoatNumCoeffs.cu
  differentialRayTracing/hitFunctionReflecting_DiffRays_CoatNumCoeffs_DiffRays.cu
  hitFunctionReflecting_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  differentialRayTracing/hitFunctionReflecting_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu
  hitFunctionReflecting_Scatter_Lambert2D.cu
  
  hitFunctionReflecting_Scatter_TorranceSparrow2D.cu
  hitFunctionReflecting_Scatter_TorranceSparrow2D_PathTrace.cu
  
  hitFunctionLinGrat1D.cu
  differentialRayTracing/hitFunctionLinGrat1D_DiffRays.cu
  hitFunctionLinGrat1D_Scatter_TorranceSparrow1D.cu
  differentialRayTracing/hitFunctionLinGrat1D_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  hitFunctionLinGrat1D_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  differentialRayTracing/hitFunctionLinGrat1D_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu
  hitFunctionLinGrat1D_Scatter_DoubleCauchy1D.cu
  differentialRayTracing/hitFunctionLinGrat1D_DiffRays_Scatter_DoubleCauchy1D_DiffRays.cu  
  
  hitFunctionIdealLense.cu
  differentialRayTracing/hitFunctionIdealLense_DiffRays.cu
  hitFunctionIdealLense_Scatter_TorranceSparrow1D.cu
  differentialRayTracing/hitFunctionIdealLense_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  hitFunctionIdealLense_CoatNumCoeffs.cu
  differentialRayTracing/hitFunctionIdealLense_DiffRays_CoatNumCoeffs_DiffRays.cu
  hitFunctionIdealLense_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  differentialRayTracing/hitFunctionIdealLense_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu

  hitFunctionReflecting_CovGlass.cu  
  
  hitFunctionPathTraceSource.cu
      
  rayGeneration.cu
  rayGeneration_PathTracing.cu
  differentialRayTracing/rayGenerationDiffRayField.cu
  differentialRayTracing/rayGenerationDiffRayField_Freeform.cu
  rayGenerationDiffRayField_RayAiming_Holo.cu
  rayGenerationDiffRayField_RayAiming_Holo_test.cu
  rayGenerationDiffRayField_RayAiming.cu
  
  planeSurface.cu
  differentialRayTracing/planeSurface_DiffRays.cu
  planeSurface_GaussBeam.cu
  sinusNormalSurface.cu
  differentialRayTracing/sinusNormalSurface_DiffRays.cu
  IdealLense.cu
  differentialRayTracing/IdealLense_DiffRays.cu
  IdealLense_GaussBeam.cu
  Selector_Visit_Program.cu
  sphericalSurface.cu
  differentialRayTracing/sphericalSurface_DiffRays.cu
  sphericalSurface_GaussBeam.cu
  apertureStop.cu
  differentialRayTracing/apertureStop_DiffRays.cu
  asphericalSurface.cu
  differentialRayTracing/asphericalSurface_DiffRays.cu
  asphericalSurface_GaussBeam.cu
  cylPipe.cu
  differentialRayTracing/cylPipe_DiffRays.cu
  cylLenseSurface.cu
  cylPipe_GaussBeam.cu
  conePipe.cu
  differentialRayTracing/conePipe_DiffRays.cu
  conePipe_GaussBeam.cu
  boundingBox.cu
  exception.cu
  
  # This is an extra option to compile to sm_13 (use sm_20 for Tesla?).  It is parsed out in the
  # OPTIX_add_sample_executable macro.
  OPTIONS -arch sm_20  
  
  )

  
SET(DebugLib "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug/ITO-MacroSimd.lib")
SET(ReleaseLib "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/ITO-MacroSim.lib")
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

#file(GLOB libsd "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Debug/*.lib")
#file(GLOB libs "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Release/*.lib")
#message(STATUS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
#message(STATUS "${MacroSim_INSTALL_DIR}lib${bit_dest}")

#foreach(file ${libsd})
#	message(STATUS "lib file name to be copied: ${file}")
#	ADD_CUSTOM_COMMAND(TARGET ${target_name} POST_BUILD            # Adds a post-build event 
#		COMMAND ${CMAKE_COMMAND} -E copy_if_different  			   # execute "cmake - E copy_if_different..."
#		"${file}"	      										   # <--this is in-file
#		"${MacroSim_INSTALL_DIR}lib${bit_dest}"                   # <--this is out-file path
#	)
#endforeach()

#foreach(file ${libs})
#	message(STATUS "lib file name to be copied: ${file}")
#	ADD_CUSTOM_COMMAND(TARGET ${target_name} POST_BUILD            # Adds a post-build event 
#		COMMAND ${CMAKE_COMMAND} -E copy_if_different  			   # execute "cmake - E copy_if_different..."
#		"${file}"	      										   # <--this is in-file
#		"${MacroSim_INSTALL_DIR}lib${bit_dest}"                   # <--this is out-file path
#	)
#endforeach()   
  
#  else()
#	# GLUT or OpenGL not found
#	message("Disabling materials, which require glut and opengl")
#  endif()

