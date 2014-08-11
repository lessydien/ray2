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
  "${CMAKE_CURRENT_SOURCE_DIR}/../nvcommon/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../nvcommon/GLEW/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../PropagationMath/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )

# add matlab libraries
#LINK_LIBRARIES( ${matlab_eng_LIBRARY} ${matlab_mx_LIBRARY} ${matlab_mat_LIBRARY} )

# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
OPTIX_add_sample_executable( ITO-MacroSim

  src/geometricRender/GeometryLib_GeomRender.cpp
  src/geometricRender/GeometryLib_GeomRender.h
  src/geometricRender/MaterialLib_GeomRender.cpp
  src/geometricRender/MaterialLib_GeomRender.h
  src/geometricRender/CoatingLib_GeomRender.cpp
  src/geometricRender/CoatingLib_GeomRender.h
  src/geometricRender/ScatterLib_GeomRender.cpp
  src/geometricRender/MaterialLight_GeomRender.cpp
  src/geometricRender/MaterialLight_GeomRender.h
  src/geometricRender/MaterialLight_GeomRender_hit.h
  src/geometricRender/ScatterLib_GeomRender.h
  src/geometricRender/GeometricRenderField.cpp
  src/geometricRender/GeometricRenderField.h
  src/geometricRender/GeometricRenderField_hostDevice.h
  src/geometricRender/MaterialFringeProj_GeomRender.cpp
  src/geometricRender/MaterialFringeProj_GeomRender.h
  src/geometricRender/MaterialFringeProj_GeomRender_hit.h
  src/geometricRender/PlaneSurface_GeomRender.cpp
  src/geometricRender/PlaneSurface_GeomRender.h
  src/geometricRender/planeSurface_GeomRender.cu
  src/geometricRender/PlaneSurface_GeomRender_intersect.h
  src/geometricRender/ApertureStop_GeomRender.cpp
  src/geometricRender/ApertureStop_GeomRender.h
  src/geometricRender/ApertureStop_GeomRender.cu
  src/geometricRender/ApertureStop_GeomRender_intersect.h
  src/geometricRender/AsphericalSurface_GeomRender.cpp
  src/geometricRender/AsphericalSurface_GeomRender.h
  src/geometricRender/AsphericalSurface_GeomRender.cu
  src/geometricRender/AsphericalSurface_GeomRender_intersect.h
  src/geometricRender/ConePipe_GeomRender.cpp
  src/geometricRender/ConePipe_GeomRender.h
  src/geometricRender/ConePipe_GeomRender.cu
  src/geometricRender/ConePipe_GeomRender_intersect.h
  src/geometricRender/CylPipe_GeomRender.cpp
  src/geometricRender/CylPipe_GeomRender.h
  src/geometricRender/CylPipe_GeomRender.cu
  src/geometricRender/CylPipe_GeomRender_intersect.h
  src/geometricRender/IdealLense_GeomRender.cpp
  src/geometricRender/IdealLense_GeomRender.h
  src/geometricRender/IdealLense_GeomRender.cu
  src/geometricRender/IdealLense_GeomRender_intersect.h
  src/geometricRender/SphericalSurface_GeomRender.cpp
  src/geometricRender/SphericalSurface_GeomRender.h
  src/geometricRender/SphericalSurface_GeomRender.cu
  src/geometricRender/SphericalSurface_GeomRender_intersect.h
  src/geometricRender/cadObject_GeomRender.cpp
  src/geometricRender/cadObject_GeomRender.h
  src/geometricRender/cadObject_GeomRender.cu
  src/geometricRender/cadObject_GeomRender_intersect.h
  src/geometricRender/MaterialAbsorbing_GeomRender.cpp
  src/geometricRender/MaterialAbsorbing_GeomRender.h
  src/geometricRender/MaterialAbsorbing_GeomRender_hit.h
  src/geometricRender/MaterialReflecting_GeomRender.cpp
  src/geometricRender/MaterialReflecting_GeomRender.h
  src/geometricRender/MaterialReflecting_GeomRender_hit.h
  src/geometricRender/MaterialRefracting_GeomRender.cpp
  src/geometricRender/MaterialRefracting_GeomRender.h
  src/geometricRender/MaterialRefracting_GeomRender_hit.h
  src/geometricRender/MaterialIdealLense_GeomRender.cpp
  src/geometricRender/MaterialIdealLense_GeomRender.h
  src/geometricRender/MaterialIdealLense_GeomRender_hit.h

  src/geometricRender/Coating_NumCoeffs_GeomRender.cpp
  src/geometricRender/Coating_NumCoeffs_GeomRender.h
  src/geometricRender/Coating_NumCoeffs_GeomRender_hit.h

  src/geometricRender/Scatter_Lambert2D_GeomRender.cpp
  src/geometricRender/Scatter_Lambert2D_GeomRender.h
  src/geometricRender/Scatter_Lambert2D_GeomRender_hit.h
  src/geometricRender/Scatter_TorranceSparrow2D_GeomRender.cpp
  src/geometricRender/Scatter_TorranceSparrow2D_GeomRender.h
  src/geometricRender/Scatter_TorranceSparrow2D_GeomRender_hit.h

  src/geometricRender/hitFunctionAbsorbing_GeomRender.cu
  src/geometricRender/hitFunctionAbsorbing_GeomRender_CoatNumCoeffs_GeomRender.cu
  src/geometricRender/hitFunctionAbsorbing_GeomRender_Scatter_Lambert2D_GeomRender.cu
  src/geometricRender/hitFunctionIdealLense_GeomRender.cu
  src/geometricRender/hitFunctionIdealLense_GeomRender_CoatNumCoeffs_GeomRender.cu
  src/geometricRender/hitFunctionReflecting_GeomRender.cu
  src/geometricRender/hitFunctionReflecting_GeomRender_CoatNumCoeffs_GeomRender.cu
  src/geometricRender/hitFunctionRefracting_GeomRender.cu
  src/geometricRender/hitFunctionRefracting_GeomRender_CoatNumCoeffs_GeomRender.cu
  src/geometricRender/hitFunctionReflecting_GeomRender_Scatter_TorranceSparrow2D_GeomRender.cu
  src/geometricRender/hitFunctionReflecting_GeomRender_Scatter_Lambert2D_GeomRender.cu
  

  src/hitFunctionRenderFringeProj.cu

  
	src/pugiconfig.hpp
	src/pugixml.cpp
	src/pugixml.hpp

  src/myUtil.h
  src/myUtil.cpp

  src/MacroSimLib.cpp
  include/MacroSimLib.h
  src/CompoundGeometry.h
  src/CompoundGeometry.cpp
  src/SphericalLense.h
  src/SphericalLense.cpp

  src/substrate.h
  src/substrate.cpp

  src/apertureArraySurface.cpp
  src/apertureArraySurface.h
  src/apertureArraySurface_intersect.h
  src/apertureArraySurface.cu
  
  src/stopArraySurface.cpp
  src/stopArraySurface.h
  src/stopArraySurface_intersect.h
  src/stopArraySurface.cu

  src/Interpolator.cpp
  src/Interpolator.h
  src/Interpolator_host_device.h

  src/FlexZemax.cpp
  src/FlexZemax.h
  src/RayField.cpp
  src/RayField.h
  src/GeometricRayField.cpp
  src/GeometricRayField.h
  src/GeometricRayField_PseudoBandwidth.cpp
  src/GeometricRayField_PseudoBandwidth.h
  src/PathTracingRayField.cpp
  src/PathTracingRayField.h
  src/PathIntTissueRayField.cpp
  src/PathIntTissueRayField.h
  src/differentialRayTracing/DiffRayfield_Freeform.cpp
  src/differentialRayTracing/DiffRayfield_Freeform.h
  src/GaussBeamRayField.h
  src/GaussBeamRayField.cpp
  src/randomGenerator.h
  src/randomGenerator.cpp
  src/DiffRayField.h
  src/DiffRayField.cpp
  src/DiffRayField_RayAiming.h
  src/DiffRayField_RayAiming.cpp
  src/DiffRayField_RayAiming_Holo.h
  src/DiffRayField_RayAiming_Holo.cpp
  src/DiffRayField_RayAiming_Holo_test.cpp
  src/DiffRayField_RayAiming_Holo_test.h
  src/inputOutput.h
  src/inputOutput.cpp
  src/ScalarLightField.h
  src/ScalarLightField.cpp
  src/ScalarPlaneField.h
  src/ScalarPlaneField.cpp
  src/ScalarSphericalField.h
  src/ScalarSphericalField.cpp
  src/ScalarGaussianField.h
  src/ScalarGaussianField.cpp
  src/ScalarUserField.h
  src/ScalarUserField.cpp

  src/VectorLightField.h
  src/VectorLightField.cpp
  src/Field.h
  src/Field.cpp
  src/FieldLib.h
  src/FieldLib.cpp
  src/Field_Stack.h
  src/Field_Stack.cpp
  src/FieldParams.h
  src/IntensityField.h
  src/IntensityField.cpp
  src/PhaseSpaceField.h
  src/PhaseSpaceField.cpp
  src/IntensityField_Stack.h
  src/IntensityField_Stack.cpp
  src/Converter.cpp
  src/Converter.h
  src/converterMath.h
  src/ConverterMath.cpp
  src/Detector.cpp
  src/Detector.h
  src/Detector_Field.cpp
  src/Detector_Field.h
  src/Detector_PhaseSpace.cpp
  src/Detector_PhaseSpace.h
  src/Detector_Intensity.cpp
  src/Detector_Intensity.h
  src/Detector_VolumeIntensity.cpp
  src/Detector_VolumeIntensity.h
  src/Detector_Raydata.cpp
  src/Detector_Raydata.h
  src/DetectorParams.h
  src/DetectorLib.h
  src/DetectorLib.cpp
  src/PlaneSurface.cpp
  src/PlaneSurface.h
  src/PlaneSurface_intersect.h
  src/differentialRayTracing/PlaneSurface_DiffRays.cpp
  src/differentialRayTracing/PlaneSurface_DiffRays.h
  src/differentialRayTracing/PlaneSurface_DiffRays_intersect.h
  src/SinusNormalSurface.cpp
  src/SinusNormalSurface.h
  src/SinusNormalSurface_intersect.h
  src/parabolicSurface.cpp
  src/parabolicSurface.h
  src/parabolicSurface.cu
  src/parabolicSurface_intersect.h
  src/differentialRayTracing/SinusNormalSurface_DiffRays.cpp
  src/differentialRayTracing/SinusNormalSurface_DiffRays.h
  src/differentialRayTracing/SinusNormalSurface_DiffRays_intersect.h
  src/Parser.cpp
  src/Parser.h
  src/Parser_XML.cpp
  src/Parser_XML.h
  src/AsphericalSurface.cpp
  src/AsphericalSurface.h
  src/AsphericalSurface_intersect.h
  src/differentialRayTracing/AsphericalSurface_DiffRays.cpp
  src/differentialRayTracing/AsphericalSurface_DiffRays.h
  src/differentialRayTracing/AsphericalSurface_DiffRays_intersect.h
  src/cadObject.cpp
  src/cadObject.h
  src/cadObject_intersect.h
  src/CylPipe.cpp
  src/CylPipe.h
  src/CylPipe_intersect.h
  src/CylLenseSurface.cpp
  src/CylLenseSurface.h
  src/CylLenseSurface_intersect.h
  src/differentialRayTracing/CylPipe_DiffRays.cpp
  src/differentialRayTracing/CylPipe_DiffRays.h
  src/differentialRayTracing/CylPipe_DiffRays_intersect.h
  src/Coating.cpp
  src/Coating.h
  src/Coating_hit.h
  src/Coating_NoCoating.cpp
  src/Coating_NoCoating.h 
  src/differentialRayTracing/Coating_DiffRays.cpp
  src/differentialRayTracing/Coating_DiffRays.h
  src/differentialRayTracing/Coating_DiffRays_hit.h
  src/Coating_NumCoeffs.cpp
  src/Coating_NumCoeffs.h
  src/Coating_NumCoeffs_hit.h
  src/Coating_DispersiveNumCoeffs.cpp
  src/Coating_DispersiveNumCoeffs.h
  src/Coating_DispersiveNumCoeffs_hit.h
  src/Coating_FresnelCoeffs.cpp
  src/Coating_FresnelCoeffs.h
  src/Coating_FresnelCoeffs_hit.h
  src/differentialRayTracing/Coating_NumCoeffs_DiffRays.cpp
  src/differentialRayTracing/Coating_NumCoeffs_DiffRays.h
  src/differentialRayTracing/Coating_NumCoeffs_DiffRays_hit.h
  src/ApertureStop.cpp
  src/ApertureStop.h
  src/ApertureStop_intersect.h
  src/differentialRayTracing/ApertureStop_DiffRays.cpp
  src/differentialRayTracing/ApertureStop_DiffRays.h
  src/differentialRayTracing/ApertureStop_DiffRays_intersect.h
  src/ConePipe.cpp
  src/ConePipe.h
  src/ConePipe_intersect.h
  src/differentialRayTracing/ConePipe_DiffRays.cpp
  src/differentialRayTracing/ConePipe_DiffRays.h
  src/differentialRayTracing/ConePipe_DiffRays_intersect.h
  src/complex.cpp
  src/complex.h
  src/volumeScattererBox.cpp
  src/volumeScattererBox.h
  src/volumeScattererBox_intersect.h
  src/IdealLense.cpp
  src/IdealLense.h
  src/IdealLense_Intersect.h
  src/differentialRayTracing/IdealLense_DiffRays.cpp
  src/differentialRayTracing/IdealLense_DiffRays.h
  src/differentialRayTracing/IdealLense_DiffRays_Intersect.h
  src/Geometry.cpp
  src/Geometry.h
  src/Geometry_Intersect.h
  src/GeometryGroup.cpp
  src/GeometryGroup.h
  src/GeometryLib.cpp
  src/GeometryLib.h
  src/differentialRayTracing/GeometryLib_DiffRays.h
  src/differentialRayTracing/GeometryLib_DiffRays.cpp
  src/Group.cpp
  src/Group.h
  src/Material.cpp
  src/Material.h
  src/Material_hit.h
  src/MaterialPathTraceSource.cpp
  src/MaterialPathTraceSource.h
  src/MaterialPathTraceSource_hit.h
  src/differentialRayTracing/Material_DiffRays_hit.h
  src/differentialRayTracing/Material_DiffRays.cpp
  src/differentialRayTracing/Material_DiffRays.h
  src/MaterialLib.cpp
  src/MaterialLib.h
  src/differentialRayTracing/MaterialLib_DiffRays.h
  src/differentialRayTracing/MaterialLib_DiffRays.cpp
  src/CoatingLib.cpp
  src/CoatingLib.h
  src/differentialRayTracing/CoatingLib_DiffRays.h
  src/differentialRayTracing/CoatingLib_DiffRays.cpp
  src/ScatterLib.cpp
  src/ScatterLib.h
  src/differentialRayTracing/ScatterLib_DiffRays.h
  src/differentialRayTracing/ScatterLib_DiffRays.cpp
  src/MaterialFilter.cpp
  src/MaterialFilter.h
  src/MaterialFilter_hit.h
  src/MaterialVolumeScatter.cpp
  src/MaterialVolumeScatter.h
  src/MaterialVolumeScatter_hit.h
  src/MaterialVolumeScatterBox.cpp
  src/MaterialVolumeScatterBox.h
  src/MaterialVolumeScatterBox_hit.h
  src/MaterialIdealLense.cpp
  src/MaterialIdealLense.h
  src/MaterialIdealLense_hit.h
  src/differentialRayTracing/MaterialIdealLense_DiffRays.cpp
  src/differentialRayTracing/MaterialIdealLense_DiffRays.h
  src/differentialRayTracing/MaterialIdealLense_DiffRays_hit.h
  src/MaterialReflecting.cpp
  src/MaterialReflecting.h
  src/MaterialReflecting_hit.h
  src/differentialRayTracing/MaterialReflecting_DiffRays.cpp
  src/differentialRayTracing/MaterialReflecting_DiffRays.h
  src/differentialRayTracing/MaterialReflecting_DiffRays_hit.h
  src/MaterialReflecting_CovGlass.cpp
  src/MaterialReflecting_CovGlass.h
  src/MaterialReflecting_CovGlass_hit.h
  src/MaterialRefracting.cpp
  src/MaterialRefracting.h
  src/MaterialRefracting_hit.h
  src/differentialRayTracing/MaterialRefracting_DiffRays.cpp
  src/differentialRayTracing/MaterialRefracting_DiffRays.h
  src/differentialRayTracing/MaterialRefracting_DiffRays_hit.h
  src/MaterialDiffracting.cpp
  src/MaterialDiffracting.h
  src/MaterialDiffracting_hit.h
  src/differentialRayTracing/MaterialDiffracting_DiffRays.cpp
  src/differentialRayTracing/MaterialDiffracting_DiffRays.h
  src/differentialRayTracing/MaterialDiffracting_DiffRays_hit.h
  src/MaterialAbsorbing.cpp
  src/MaterialAbsorbing.h
  src/MaterialAbsorbing_hit.h
  src/MaterialVolumeAbsorbing.cpp
  src/MaterialVolumeAbsorbing.h
  src/MaterialVolumeAbsorbing_hit.h
  src/differentialRayTracing/MaterialAbsorbing_DiffRays.cpp
  src/differentialRayTracing/MaterialAbsorbing_DiffRays.h
  src/differentialRayTracing/MaterialAbsorbing_DiffRays_hit.h
  src/MaterialDOE.cpp
  src/MaterialDOE.h
  src/MaterialDOE_hit.h
  src/MaterialLinearGrating1D.cpp
  src/MaterialLinearGrating1D.h
  src/MaterialLinearGrating1D_hit.h
  src/differentialRayTracing/MaterialLinearGrating1D_DiffRays.cpp
  src/differentialRayTracing/MaterialLinearGrating1D_DiffRays.h
  src/differentialRayTracing/MaterialLinearGrating1D_DiffRays_hit.h
  src/Scatter.cpp
  src/Scatter.h
  src/Scatter_hit.h
  src/Scatter_NoScatter.cpp
  src/Scatter_NoScatter.h
  
  src/differentialRayTracing/Scatter_DiffRays.cpp
  src/differentialRayTracing/Scatter_DiffRays.h
  src/differentialRayTracing/Scatter_DiffRays_hit.h
  src/Scatter_TorranceSparrow1D.cpp
  src/Scatter_TorranceSparrow1D.h
  src/Scatter_TorranceSparrow1D_hit.h
  src/Scatter_TorranceSparrow2D.cpp
  src/Scatter_TorranceSparrow2D.h
  src/Scatter_TorranceSparrow2D_hit.h
  src/Scatter_TorranceSparrow2D_PathTrace.cpp
  src/Scatter_TorranceSparrow2D_PathTrace.h
  src/Scatter_TorranceSparrow2D_PathTrace_hit.h
  
  src/differentialRayTracing/Scatter_TorranceSparrow1D_DiffRays.cpp
  src/differentialRayTracing/Scatter_TorranceSparrow1D_DiffRays.h
  src/differentialRayTracing/Scatter_TorranceSparrow1D_DiffRays_hit.h
  src/Scatter_DoubleCauchy1D.cpp
  src/Scatter_DoubleCauchy1D.h
  src/Scatter_DoubleCauchy1D_hit.h
  src/Scatter_DispersiveDoubleCauchy1D.cpp
  src/Scatter_DispersiveDoubleCauchy1D.h
  src/Scatter_DispersiveDoubleCauchy1D_hit.h
  src/differentialRayTracing/Scatter_DoubleCauchy1D_DiffRays.cpp
  src/differentialRayTracing/Scatter_DoubleCauchy1D_DiffRays.h
  src/differentialRayTracing/Scatter_DoubleCauchy1D_DiffRays_hit.h
  src/Scatter_Lambert2D.cpp
  src/Scatter_Lambert2D.h
  src/Scatter_Lambert2D_hit.h
  src/differentialRayTracing/Scatter_Lambert2D_DiffRays.cpp
  src/differentialRayTracing/Scatter_Lambert2D_DiffRays.h
  src/differentialRayTracing/Scatter_Lambert2D_DiffRays_hit.h
  src/microLensArray.cpp
  src/microLensArray.h
  src/microLensArraySurface.cpp
  src/microLensArraySurface.h
  src/microLensArraySurface_intersect.h
  src/microLensArraySurface.cu
  src/SphericalSurface.cpp
  src/SphericalSurface.h
  src/SphericalSurface_Intersect.h
  src/differentialRayTracing/SphericalSurface_DiffRays.cpp
  src/differentialRayTracing/SphericalSurface_DiffRays.h
  src/differentialRayTracing/SphericalSurface_DiffRays_Intersect.h
  src/SimAssistant.cpp
  src/SimAssistant.h
  src/SimAssistant_SingleSim.cpp
  src/SimAssistant_SingleSim.h
  src/SimAssistant_ParamSweep.cpp
  src/SimAssistant_ParamSweep.h
  src/SimAssistant_Layout.cpp
  src/SimAssistant_Layout.h
  src/SimAssistantLib.h
  src/rayTracingMath.h
  src/wavefrontIn.cpp
  src/wavefrontIn.h
  src/rayData.h
  src/GlobalConstants.h
  src/TopObject.h
  src/IlluminationPupilParams.h
  src/PupilParams.h
  src/PupilLib.h
  src/Pupil_RayAim_FarField.cpp
  src/Pupil_RayAim_FarField.h
  src/Pupil_RayAim_FarField_aim.h
  src/Pupil.cpp
  src/Pupil.h
  src/Pupil_aim.h
  
  
  src/missFunction.cu
  src/hitFunctionAbsorbing.cu
  src/differentialRayTracing/hitFunctionAbsorbing_DiffRays.cu
  src/hitFunctionAbsorbing_Scatter_TorranceSparrow1D.cu
  src/differentialRayTracing/hitFunctionAbsorbing_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  src/hitFunctionAbsorbing_CoatNumCoeffs.cu
  src/hitFunctionAbsorbing_CoatFresnelCoeffs.cu
  src/differentialRayTracing/hitFunctionAbsorbing_DiffRays_CoatNumCoeffs_DiffRays.cu
  src/hitFunctionAbsorbing_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  src/differentialRayTracing/hitFunctionAbsorbing_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu
  src/hitFunctionAbsorbing_Scatter_Lambert2D.cu
  src/differentialRayTracing/hitFunctionAbsorbing_DiffRays_Scatter_Lambert2D_DiffRays.cu
  src/hitFunctionAbsorbing_Scatter_DoubleCauchy1D_CoatNumCoeffs.cu
  
  src/hitFunctionRefracting.cu
  src/differentialRayTracing/hitFunctionRefracting_DiffRays.cu
  src/hitFunctionRefracting_Scatter_TorranceSparrow1D.cu
  src/differentialRayTracing/hitFunctionRefracting_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  src/hitFunctionRefracting_CoatNumCoeffs.cu
  src/differentialRayTracing/hitFunctionRefracting_DiffRays_CoatNumCoeffs_DiffRays.cu
  src/hitFunctionRefracting_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  src/differentialRayTracing/hitFunctionRefracting_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu
  
  src/hitFunctionFilter.cu
  src/hitFunctionFilter_CoatNumCoeffs.cu
  
  src/hitFunctionVolumeScatter.cu
  src/hitFunctionVolumeScatterBox.cu
  
  src/hitFunctionDOE.cu
  
  src/hitFunctionDiffracting.cu
  src/differentialRayTracing/hitFunctionDiffracting_DiffRays.cu
  
  src/hitFunctionReflecting.cu
  src/differentialRayTracing/hitFunctionReflecting_DiffRays.cu
  src/hitFunctionReflecting_Scatter_TorranceSparrow1D.cu
  src/differentialRayTracing/hitFunctionReflecting_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  src/hitFunctionReflecting_CoatNumCoeffs.cu
  src/differentialRayTracing/hitFunctionReflecting_DiffRays_CoatNumCoeffs_DiffRays.cu
  src/hitFunctionReflecting_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  src/differentialRayTracing/hitFunctionReflecting_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu
  src/hitFunctionReflecting_Scatter_Lambert2D.cu
  
  src/hitFunctionReflecting_Scatter_TorranceSparrow2D.cu
  src/hitFunctionReflecting_Scatter_TorranceSparrow2D_PathTrace.cu
  
  src/hitFunctionLinGrat1D.cu
  src/differentialRayTracing/hitFunctionLinGrat1D_DiffRays.cu
  src/hitFunctionLinGrat1D_Scatter_TorranceSparrow1D.cu
  src/differentialRayTracing/hitFunctionLinGrat1D_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  src/hitFunctionLinGrat1D_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  src/differentialRayTracing/hitFunctionLinGrat1D_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu
  src/hitFunctionLinGrat1D_Scatter_DoubleCauchy1D.cu
  src/differentialRayTracing/hitFunctionLinGrat1D_DiffRays_Scatter_DoubleCauchy1D_DiffRays.cu  
  
  src/hitFunctionIdealLense.cu
  src/differentialRayTracing/hitFunctionIdealLense_DiffRays.cu
  src/hitFunctionIdealLense_Scatter_TorranceSparrow1D.cu
  src/differentialRayTracing/hitFunctionIdealLense_DiffRays_Scatter_TorranceSparrow1D_DiffRays.cu
  src/hitFunctionIdealLense_CoatNumCoeffs.cu
  src/differentialRayTracing/hitFunctionIdealLense_DiffRays_CoatNumCoeffs_DiffRays.cu
  src/hitFunctionIdealLense_Scatter_TorranceSparrow1D_CoatNumCoeffs.cu
  src/differentialRayTracing/hitFunctionIdealLense_DiffRays_Scatter_TorranceSparrow1D_DiffRays_CoatNumCoeffs_DiffRays.cu

  src/hitFunctionReflecting_CovGlass.cu  
  
  src/hitFunctionPathTraceSource.cu
      
  src/rayGeneration.cu
  src/rayGenerationGeomRender.cu
  src/rayGeneration_PathTracing.cu
  src/differentialRayTracing/rayGenerationDiffRayField.cu
  #src/differentialRayTracing/rayGenerationDiffRayField_Freeform.cu
  src/rayGenerationDiffRayField_RayAiming_Holo.cu
  src/rayGenerationDiffRayField_RayAiming_Holo_test.cu
  src/rayGenerationDiffRayField_RayAiming.cu
  src/rayGeneration_checkVisibility.cu
  src/rayGenerationPathIntTissueRayField.cu
  
  src/planeSurface.cu
  src/differentialRayTracing/planeSurface_DiffRays.cu
  src/planeSurface_GaussBeam.cu
  src/sinusNormalSurface.cu
  src/differentialRayTracing/sinusNormalSurface_DiffRays.cu
  src/IdealLense.cu
  src/differentialRayTracing/IdealLense_DiffRays.cu
  src/IdealLense_GaussBeam.cu
  src/Selector_Visit_Program.cu
  src/sphericalSurface.cu
  src/differentialRayTracing/sphericalSurface_DiffRays.cu
  src/sphericalSurface_GaussBeam.cu
  src/apertureStop.cu
  src/differentialRayTracing/apertureStop_DiffRays.cu
  src/asphericalSurface.cu
  src/differentialRayTracing/asphericalSurface_DiffRays.cu
  src/asphericalSurface_GaussBeam.cu
  src/cylPipe.cu
  src/differentialRayTracing/cylPipe_DiffRays.cu
  src/cylLenseSurface.cu
  src/cylPipe_GaussBeam.cu
  src/conePipe.cu
  src/differentialRayTracing/conePipe_DiffRays.cu
  src/volumeScattererBox.cu
  src/conePipe_GaussBeam.cu
  src/cadObject.cu
  src/boundingBox.cu
  src/exception.cu
  src/cadObject.cu
  
  # This is an extra option to compile to sm_13 (use sm_20 for Tesla?).  It is parsed out in the
  # OPTIX_add_sample_executable macro.
  OPTIONS -arch sm_20 #--use_fast_math
  
  )

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

