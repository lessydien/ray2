SET (target_name MacroSim)

project(${target_name})

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_SHARED_LIBS "Build shared library." ON)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)
OPTION(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET (CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
SET (ITOM_LANGUAGES "de" CACHE STRING "semicolon separated list of languages that should be created (en must not be given since it is the default)")

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${ITOM_SDK_DIR})

if (BUILD_TARGET64)
   set(CMAKE_SIZEOF_VOID_P 8)
else (BUILD_TARGET64)
   set(CMAKE_SIZEOF_VOID_P 4)
endif (BUILD_TARGET64)

IF(BUILD_SHARED_LIBS)
	SET(LIBRARY_TYPE SHARED)
ELSE(BUILD_SHARED_LIBS)
	SET(LIBRARY_TYPE STATIC)
ENDIF(BUILD_SHARED_LIBS)

find_package(OpenCV REQUIRED)
#find_package(Glut REQUIRED)
find_package(ITOM_SDK REQUIRED)
find_package(FFTW 3)
find_package(CUDA 4)
find_package(GLEW)
#SET(VTK_INSTALL_EXPORT_NAME 1) #avoid that vtk directly links to an absolute Qt-library, that is located somewhere on their computer configuration.
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

IF(CUDA_FOUND)

    include("${ITOM_SDK_DIR}/ItomBuildMacros.cmake")

    ADD_DEFINITIONS(${QT_DEFINITIONS})
    IF (BUILD_UNICODE)
        ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
    ENDIF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DCMAKE)

    # enable some qt stuff
    SET (QT_USE_QTOPENGL TRUE)
    SET (QT_USE_QTXML TRUE)

    # default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
    IF (DEFINED CMAKE_BUILD_TYPE)
        SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
    ELSE(CMAKE_BUILD_TYPE)
        SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
    ENDIF (DEFINED CMAKE_BUILD_TYPE)

    message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

    if ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
        SET(Tracer_LIB_DIR "Tracer/lib32")
        #SET(freeglut_LIB_DIR "freeglut/lib")
		SET(Tracer_PTX_DIR "Tracer/build/lib/ptx")
        SET(bit_dest "32")
    else ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
        SET(Tracer_LIB_DIR "Tracer/lib")
        #SET(freeglut_LIB_DIR "freeglut/lib/x64")
		SET(Tracer_PTX_DIR "Tracer/build64/lib/ptx")
        SET(bit_dest "")
    endif ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
	
	#on MSVC enable build using OpenMP for compiling
	if(MSVC)
		ADD_DEFINITIONS(/MP)

		# set some optimization compiler flags
		# i.e.:
		#   - Ox full optimization (replaces standard O2 set by cmake)
		#	- Oi enable intrinsic functions
		#	- Ot favor fast code
		#	- Oy omit frame pointers
		#	- GL whole program optimization
		# 	- GT fibre safe optimization
		#	- openmp enable openmp support, isn't enabled globally here as it breaks opencv
		SET ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /Oy /GL /openmp" )
		SET ( CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
	endif (MSVC)	

	set(PTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_PTX_DIR} CACHE PATH "Path to where the samples look for the PTX code.")
	
    find_library(MacroSim_Tracer_LIBD ITO-MacroSimd HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    find_library(MacroSim_Tracer_LIBR ITO-MacroSim HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    set(MacroSim_Tracer_LIB debug ${MacroSim_Tracer_LIBD} optimized ${MacroSim_Tracer_LIBR}  CACHE STRING "" FORCE)

    find_library(MacroSim_Propagation_LIBD PropagationMathd HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    find_library(MacroSim_Propagation_LIBR PropagationMath HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    set(MacroSim_Propagation_LIB debug ${MacroSim_Propagation_LIBD} optimized ${MacroSim_Propagation_LIBR}  CACHE STRING "" FORCE)

	find_library(MacroSim_nvCommon_LIBD nvCommond HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    find_library(MacroSim_nvCommon_LIBR nvCommon HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    set(MacroSim_nvCommon_LIB debug ${MacroSim_nvCommon_LIBD} optimized ${MacroSim_nvCommon_LIBR}  CACHE STRING "" FORCE)

	find_library(MacroSim_nvModel_LIBD nvModeld HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    find_library(MacroSim_nvModel_LIBR nvModel HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    set(MacroSim_nvModel_LIB debug ${MacroSim_nvModel_LIBD} optimized ${MacroSim_nvModel_LIBR}  CACHE STRING "" FORCE)
	
    find_library(MacroSim_OptiX_LIBR optix.1 HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    find_library(MacroSim_OptiXu_LIBR optixu.1 HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR})
    set(MacroSim_OptiX_LIB debug ${MacroSim_OptiX_LIBR} optimized ${MacroSim_OptiX_LIBR}  CACHE STRING "" FORCE)
    set(MacroSim_OptiXu_LIB debug ${MacroSim_OptiXu_LIBR} optimized ${MacroSim_OptiXu_LIBR}  CACHE STRING "" FORCE)

    #find_library(Freeglut_LIBR freeglut HINTS ${CMAKE_CURRENT_SOURCE_DIR}/${freeglut_LIB_DIR})
    #set(Freeglut_LIB debug ${Freeglut_LIBR} optimized ${Freeglut_LIBR}  CACHE STRING "" FORCE)


    set(MacroSim_Tracer_LIBS ${MacroSim_Tracer_LIBS} ${MacroSim_Tracer_LIB} ${MacroSim_OptiX_LIB} ${MacroSim_OptiXu_LIB} ${MacroSim_Propagation_LIB} ${MacroSim_nvModel_LIB} ${MacroSim_nvCommon_LIB} ${GLEW_LIBRARIES})

    INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${QT_QTCORE_INCLUDE_DIR}
        ${OpenCV_DIR}/include
        ${ITOM_SDK_INCLUDE_DIR}
        #${CMAKE_CURRENT_SOURCE_DIR}/Tracer/include
		${CMAKE_CURRENT_SOURCE_DIR}/Tracer/src/nvcommon/include
		${CMAKE_CURRENT_SOURCE_DIR}/Tracer/src/nvcommon/nvModel/include
		${CMAKE_CURRENT_SOURCE_DIR}/Tracer/src/PropagationMath/include
		${CMAKE_CURRENT_SOURCE_DIR}/Tracer/src/ito_tracer/include
        #${CMAKE_CURRENT_SOURCE_DIR}/freeglut/include/GL
        ${CUDA_INCLUDE_DIRS}
		${FFTW_INCLUDE_DIR}
		${GLEW_INCLUDE_PATH}
#		${VTK_SOURCE_DIR}/GUISupport/Qt
        ${VTK_INCLUDE_DIRS}
    )
    

    LINK_DIRECTORIES(
        ${OpenCV_DIR}/lib
    #	${CMAKE_CURRENT_SOURCE_DIR}/Tracer/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/${Tracer_LIB_DIR}
        #${CMAKE_CURRENT_SOURCE_DIR}/${freeglut_LIB_DIR}
        ${VTK_LIBRARY_DIRS}
    )

    INCLUDE(${QT_USE_FILE})

    set(plugin_HEADERS
        ${ITOM_SDK_INCLUDE_DIR}/common/addInInterface.h
        ${ITOM_SDK_INCLUDE_DIR}/common/helperCommon.h
        ${ITOM_SDK_INCLUDE_DIR}/common/sharedStructures.h
        ${ITOM_SDK_INCLUDE_DIR}/common/sharedStructuresQt.h
        ${ITOM_SDK_INCLUDE_DIR}/common/typeDefs.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/ColorCombo.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/CustomTypes.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/EnumProperty.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Property.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/QPropertyEditorWidget.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/QPropertyModel.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/QVariantDelegate.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec2dProperty.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec2iProperty.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec3dProperty.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec3fProperty.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec9dProperty.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec9siProperty.h
        ${CMAKE_CURRENT_SOURCE_DIR}/abstractItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/coatingItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/coatingItemLib.h
        ${CMAKE_CURRENT_SOURCE_DIR}/coatingNoItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/coatingNumCoeffsItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/consoleStream.h
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorFieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorIntensityItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/detectorVolumeIntensityItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorItemLib.h
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorLibraryContainer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorRayDataItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/dockWidget_Console.h
        ${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetMacroSim.h
        ${CMAKE_CURRENT_SOURCE_DIR}/fieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/fieldItemLib.h
        ${CMAKE_CURRENT_SOURCE_DIR}/fieldLibraryContainer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomApertureStopItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomAsphericalSurfaceItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomConePipeItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomCylLenseSurfaceItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomCylPipeItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geometryItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geometryItemLib.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geometryLibraryContainer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomIdealLenseItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomMicroLensArrayItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomPlaneSurfaceItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomParabolicSurfaceItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomRayFieldItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/geomRenderFieldItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/geomRayFieldItem_PseudoBandwidth.h
		${CMAKE_CURRENT_SOURCE_DIR}/diffRayFieldItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/diffRayField_RayAiming_Item.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomSphericalLenseItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/geomSphericalSurfaceItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/geomApertureArrayItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/geomSubstrateItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/geomStopArrayItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/geomVolumeScattererItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/intensityFieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/macroSim.h
        ${CMAKE_CURRENT_SOURCE_DIR}/macrosim_librarymodel.h
        ${CMAKE_CURRENT_SOURCE_DIR}/macrosim_scenemodel.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mainWinMacroSim.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialAbsorbingItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/materialVolumeAbsorbingItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialDiffractingItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialFilterItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialIdealLenseItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialItemLib.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialLinearGrating1DItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialPathTraceSourceItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialReflectingCovGlassItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialReflectingItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialRefractingItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/materialDOEItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/materialRenderLightItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/materialVolumeScatterItem.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/MyDelegate.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/myGraphicsScene.h
        ${CMAKE_CURRENT_SOURCE_DIR}/rayFieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/renderFuncs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarFieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarGaussianFieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarPlaneFieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarSphericalFieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarUserFieldItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterDispersiveDoubleCauchy1DItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterDoubleCauchy1DItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterItemLib.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterLambert2DItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterNoItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterTorranceSparrow1DItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterTorranceSparrow2DItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterTorranceSparrow2DPathTraceItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TracerThread.h
        ${CMAKE_CURRENT_SOURCE_DIR}/trackball.h
        ${CMAKE_CURRENT_SOURCE_DIR}/oGLdrawWidget.h
		${CMAKE_CURRENT_SOURCE_DIR}/pathIntTissueFieldItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/geomCadObjectItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/geomGroupItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/miscLibraryContainer.h
		${CMAKE_CURRENT_SOURCE_DIR}/miscItemLib.h
		${CMAKE_CURRENT_SOURCE_DIR}/miscItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/myVtkWindow.h
        ${CMAKE_CURRENT_SOURCE_DIR}/pluginVersion.h
    )

    QT4_WRAP_CPP_ITOM(plugin_HEADERS_MOC ${plugin_HEADERS})

    set(plugin_ui
        ${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetMacroSim.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/mainWinMacroSim.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/renderOptionsDialog.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/simConfigDialog.ui
		${CMAKE_CURRENT_SOURCE_DIR}/imageSavingIdleDialog.ui
    )

    QT4_WRAP_UI_ITOM(plugin_ui_MOC ${plugin_ui})

    set(plugin_SOURCES 
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/ColorCombo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/CustomTypes.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/EnumProperty.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Property.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/QPropertyEditorWidget.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/QPropertyModel.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/QVariantDelegate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec2dProperty.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec2iProperty.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec3dProperty.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec3fProperty.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec9dProperty.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/Vec9siProperty.cpp	
        ${CMAKE_CURRENT_SOURCE_DIR}/abstractItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/coatingItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/coatingItemLib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/coatingNoItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/coatingNumCoeffsItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/consoleStream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorFieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorIntensityItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/detectorVolumeIntensityItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorItemLib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorLibraryContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detectorRayDataItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dockWidget_Console.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetMacroSim.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fieldItemLib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fieldLibraryContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomApertureStopItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomAsphericalSurfaceItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomConePipeItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomCylLenseSurfaceItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomCylPipeItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geometryItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geometryItemLib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geometryLibraryContainer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomIdealLenseItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomMicroLensArrayItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomPlaneSurfaceItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomParabolicSurfaceItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomRayFieldItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/geomRenderFieldItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/geomRayFieldItem_PseudoBandwidth.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/diffRayFieldItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/diffRayField_RayAiming_Item.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomSphericalLenseItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/geomSphericalSurfaceItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/geomApertureArrayItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/geomSubstrateItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/geomStopArrayItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/geomVolumeScattererItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/intensityFieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/macroSim.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/macrosimmodel_librarymodel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/macrosimmodel_scenemodel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mainWinMacroSim.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialAbsorbinglItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/materialVolumeAbsorbingItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialDiffractingItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialFilterItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialIdealLenseItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialItemLib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialLinearGrating1DItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialPathTraceSourceItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialReflectingCovGlassItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialReflectingItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/materialRefractingItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/materialDOEItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/materialRenderLightItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/materialVolumeScatterItem.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/MyDelegate.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/myGraphicsScene.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rayFieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/renderFuncs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarFieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarGaussianFieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarPlaneFieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarSphericalFieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scalarUserFieldItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterDispersiveDoubleCauchy1DItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterDoubleCauchy1DItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterItemLib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterLambert2DItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterNoItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterNoItemTorranceSparrow2D.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterNoItemTorranceSparrow2DPathTrace.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterTorranceSparrow1DItem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TracerThread.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trackball.cpp	
        ${CMAKE_CURRENT_SOURCE_DIR}/oGLdrawWidget.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/pathIntTissueFieldItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/geomCadObjectItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/geomGroupItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/miscLibraryContainer.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/miscItemLib.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/miscItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/myVtkWindow.cpp
    )
	
    #Add version information to the plugIn-dll unter MSVC
    if(MSVC)
        list(APPEND plugin_SOURCES ${ITOM_SDK_INCLUDE_DIR}/../pluginLibraryVersion.rc)
    endif(MSVC)    
    
	set(qitom_rcc
		${CMAKE_CURRENT_SOURCE_DIR}/macroSimRes.qrc
	)
	QT4_ADD_RESOURCES(plugin_rcc_MOC ${qitom_rcc})
	
	#handle translations BEGIN STEP 1
	set (FILES_TO_TRANSLATE ${plugin_SOURCES} ${plugin_HEADERS} ${plugin_ui})
	set (TRANSLATION_OUTPUT_FILES)
	set (TRANSLATIONS_FILES)

	file (GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
	#handle translations END STEP 1

    ADD_LIBRARY(${target_name} ${LIBRARY_TYPE} ${plugin_SOURCES} ${plugin_RESOURCES} ${plugin_HEADERS} ${plugin_HEADERS_MOC} ${plugin_ui_MOC} ${plugin_rcc_MOC} ${EXISTING_TRANSLATION_FILES})
	if (VTK_LIBRARIES)
		TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${QT_QTUITOOLS_LIBRARY} ${QT_QTXML_LIBRARY} ${OpenCV_LIBS} ${MacroSim_Tracer_LIBS} ${ITOM_SDK_dataobject_LIBRARY} ${ITOM_SDK_qpropertyeditor_LIBRARY} ${FFTW_LIBRARY_D} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES} ${ITOM_SDK_itomCommonLib_LIBRARY} ${ITOM_SDK_itomCommonQtLib_LIBRARY} ${VTK_LIBRARIES} QVTK)
	else ()
		TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${QT_QTUITOOLS_LIBRARY} ${QT_QTXML_LIBRARY} ${OpenCV_LIBS} ${MacroSim_Tracer_LIBS} ${ITOM_SDK_dataobject_LIBRARY} ${ITOM_SDK_qpropertyeditor_LIBRARY} ${FFTW_LIBRARY_D} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES} ${ITOM_SDK_itomCommonLib_LIBRARY} ${ITOM_SDK_itomCommonQtLib_LIBRARY} vtkHybrid QVTK vtkViews)
	endif()
	#handle translations BEGIN STEP 2
	if (UPDATE_TRANSLATIONS)
		QT4_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target_name} ITOM_LANGUAGES ${FILES_TO_TRANSLATE} ${TRANSLATIONS_FILES})
		
		add_custom_target (_${target_name}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
		add_dependencies(${target_name} _${target_name}_translation)
		
		QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${TRANSLATIONS_FILES})
	else (UPDATE_TRANSLATIONS)
		QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${EXISTING_TRANSLATION_FILES})
	endif (UPDATE_TRANSLATIONS)
	#handle translations END STEP 2

    # COPY SECTION
    set(COPY_SOURCES "")
    set(COPY_DESTINATIONS "")
    ADD_PLUGINLIBRARY_TO_COPY_LIST(${target_name} COPY_SOURCES COPY_DESTINATIONS)
	ADD_QM_FILES_TO_COPY_LIST(${target_name} QM_FILES COPY_SOURCES COPY_DESTINATIONS)
    POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)

    # copy dlls in itoms lib folder
    file(GLOB dlls "${CMAKE_CURRENT_SOURCE_DIR}/Tracer/bin${bit_dest}/*.dll")
	message(STATUS ${dlls})
    POST_BUILD_COPY_FILE_TO_LIB_FOLDER(${target_name} dlls)
	POST_BUILD_COPY_FILE_TO_LIB_FOLDER(${target_name} GLEW_RUNTIME_LIBRARIES)
		
	ADD_CUSTOM_COMMAND(TARGET ${target_name} POST_BUILD                 # Adds a post-build event to MyTest
		COMMAND ${CMAKE_COMMAND} -E copy_directory
			"${PTX_DIR}"
			"${ITOM_APP_DIR}/plugins/${target_name}/ptx"
			)	

ELSE(CUDA_FOUND)
    message(WARNING "CUDA could not be found. ${target_name} will not be build")
ENDIF(CUDA_FOUND)