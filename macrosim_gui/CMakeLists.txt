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

find_package(ITOM_SDK COMPONENTS dataobject itomCommonLib itomCommonQtLib qpropertyeditor REQUIRED)
include("${ITOM_SDK_DIR}/ItomBuildMacros.cmake")
FIND_PACKAGE_QT(ON Core Widgets Xml LinguistTools OpenGL)
find_package(OpenCV COMPONENTS core REQUIRED)
find_package(CUDA 4)
#SET(VTK_INSTALL_EXPORT_NAME 1) #avoid that vtk directly links to an absolute Qt-library, that is located somewhere on their computer configuration.
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

IF(CUDA_FOUND)

    include("${ITOM_SDK_DIR}/ItomBuildMacros.cmake")

    ADD_DEFINITIONS(${QT_DEFINITIONS})
    IF (BUILD_UNICODE)
        ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
    ENDIF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DCMAKE)

    # default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
    IF (DEFINED CMAKE_BUILD_TYPE)
        SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
    ELSE(CMAKE_BUILD_TYPE)
        SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
    ENDIF (DEFINED CMAKE_BUILD_TYPE)

    message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

    if ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
        SET(Tracer_LIB_DIR "Tracer/lib32")
        SET(Tracer_PTX_DIR "Tracer/build/lib/ptx")
        SET(bit_dest "32")
    else ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
        SET(Tracer_LIB_DIR "Tracer/lib")
        SET(Tracer_PTX_DIR "Tracer/build64/lib/ptx")
        SET(bit_dest "")
    endif ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
    
    #on MSVC enable build using OpenMP for compiling
    if(MSVC)
        ADD_DEFINITIONS(/MP)

        # set some optimization compiler flags
        # i.e.:
        #   - Ox full optimization (replaces standard O2 set by cmake)
        #    - Oi enable intrinsic functions
        #    - Ot favor fast code
        #    - Oy omit frame pointers
        #    - GL whole program optimization
        #     - GT fibre safe optimization
        #    - openmp enable openmp support, isn't enabled globally here as it breaks opencv
        SET ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /Oy /GL /openmp" )
        SET ( CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    endif (MSVC)    

    INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${QT_QTCORE_INCLUDE_DIR}
        ${OpenCV_DIR}/include
        ${ITOM_SDK_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../macrosim_tracer/nvcommon/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../macrosim_tracer/nvcommon/nvModel/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../macrosim_tracer/PropagationMath/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../macrosim_tracer/ito_tracer/include
        ${CUDA_INCLUDE_DIRS}
        ${VTK_INCLUDE_DIRS}
    )
    

    LINK_DIRECTORIES(
        ${OpenCV_DIR}/lib
        ${VTK_LIBRARY_DIRS}
    )

    set(plugin_HEADERS
        ${ITOM_SDK_INCLUDE_DIR}/common/addInInterface.h
        ${ITOM_SDK_INCLUDE_DIR}/common/helperCommon.h
        ${ITOM_SDK_INCLUDE_DIR}/common/sharedStructures.h
        ${ITOM_SDK_INCLUDE_DIR}/common/sharedStructuresQt.h
        ${ITOM_SDK_INCLUDE_DIR}/common/typeDefs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/CustomTypes.h
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
        ${CMAKE_CURRENT_SOURCE_DIR}/materialRenderFringeProjItem.h
        ${CMAKE_CURRENT_SOURCE_DIR}/materialVolumeScatterItem.h
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
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterPhongItem.h
		${CMAKE_CURRENT_SOURCE_DIR}/scatterCookTorranceItem.h
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

        set(plugin_SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/QPropertyEditor/CustomTypes.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/materialRenderFringeProjItem.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/scatterPhongItem.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/scatterCookTorranceItem.cpp
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
    
    set(plugin_RCC
        ${CMAKE_CURRENT_SOURCE_DIR}/macroSimRes.qrc
    )

    set(plugin_UI
        ${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetMacroSim.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/mainWinMacroSim.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/renderOptionsDialog.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/simConfigDialog.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/imageSavingIdleDialog.ui
    )

    if (QT5_FOUND)
        #if automoc if OFF, you also need to call QT5_WRAP_CPP here
        QT5_WRAP_UI(plugin_UI_MOC ${plugin_UI})
        QT5_ADD_RESOURCES(plugin_RCC_MOC ${plugin_RCC})
    else (QT5_FOUND)
        QT4_WRAP_CPP_ITOM(plugin_HEADERS_MOC ${plugin_HEADERS})
        QT4_WRAP_UI_ITOM(plugin_UI_MOC ${plugin_UI})
        QT4_ADD_RESOURCES(plugin_RCC_MOC ${plugin_RCC})
    endif (QT5_FOUND)


    file (GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")

    ADD_LIBRARY(${target_name} ${LIBRARY_TYPE} ${plugin_SOURCES} ${plugin_RESOURCES} ${plugin_HEADERS} ${plugin_HEADERS_MOC} ${plugin_UI_MOC} ${plugin_RCC_MOC} ${EXISTING_TRANSLATION_FILES})

    if (QT5_FOUND)
        TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${OpenCV_LIBS} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES} ${ITOM_SDK_LIBRARIES} ${VTK_LIBRARIES} ${GLEW_LIBRARIES} nvCommon nvModel macrosim_tracer PropagationMath)
        qt5_use_modules(${target_name} ${QT_COMPONENTS})
    else (QT5_FOUND)
        TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${OpenCV_LIBS} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES} ${ITOM_SDK_LIBRARIES} ${VTK_LIBRARIES} QVTK ${GLEW_LIBRARIES} nvCommon nvModel macrosim_tracer PropagationMath)
    endif (QT5_FOUND)      
    
    #translation
    set (FILES_TO_TRANSLATE ${plugin_SOURCES} ${plugin_HEADERS} ${plugin_UI})
    PLUGIN_TRANSLATION(QM_FILES ${target_name} ${UPDATE_TRANSLATIONS} "${EXISTING_TRANSLATION_FILES}" ITOM_LANGUAGES "${FILES_TO_TRANSLATE}")

    # COPY SECTION
    set(COPY_SOURCES "")
    set(COPY_DESTINATIONS "")
    ADD_PLUGINLIBRARY_TO_COPY_LIST(${target_name} COPY_SOURCES COPY_DESTINATIONS)
    ADD_QM_FILES_TO_COPY_LIST(${target_name} QM_FILES COPY_SOURCES COPY_DESTINATIONS)
    POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)

    # copy dlls in itoms lib folder
    #file(GLOB dlls "${CMAKE_CURRENT_SOURCE_DIR}/Tracer/bin${bit_dest}/*.dll")
    #message(STATUS ${dlls})
	SET(OPTIX_DLLs ${optix_DLL} ${optixu_DLL})
    POST_BUILD_COPY_FILE_TO_LIB_FOLDER(${target_name} OPTIX_DLLs)
        
	#add another sub-project that does nothing more but copying the cuda output files (*.ptx)
	#to the itom/plugins/macrosim/ptx folder. This sub-project is always outdated, such that
	#compiling ALL_BUILD or POST_COPY_TO_ITOM_PLUGINS will always copy the ptx files.
	add_custom_target(
		POST_COPY_TO_ITOM_PLUGINS ALL
		COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PTX_DIR}"
            "${ITOM_APP_DIR}/plugins/${target_name}/ptx"
		DEPENDS macrosim_tracer
    ) 

ELSE(CUDA_FOUND)
    message(WARNING "CUDA could not be found. ${target_name} will not be build")
ENDIF(CUDA_FOUND)