/***********************************************************************
 This file is part of ITO-MacroSim.

    ITO-MacroSim is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ITO-MacroSim is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
************************************************************************/

/**\file Parser.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef PARSER_H
#define PARSER_H



#include "stdio.h"
#include "FlexZemax.h"
#include "Group.h"
//#include "Geometry.h"
#include "GeometryLib.h"
#include "MaterialLib.h"
#include "differentialRayTracing/Material_DiffRays_Lib.h"
#include "ScatterLib.h"
#include "FieldLib.h"
//#include "Scatter.h"
#include "CoatingLib.h"
//#include "Coating.h"
//#include "RayField.h"
//#include "GeometricRayField.h"
//#include "ScalarLightField.h"
//#include "VectorLightField.h"
#include "wavefrontIn.h"
#include "inputOutput.h"
#include "GlobalConstants.h"
//#include "Detector.h"
#include "DetectorLib.h"
#include <stdlib.h>
#include <iostream>

bool createGeometricSceneFromZemax(Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, TraceMode mode);
bool createPathTracingSceneFromZemax(Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, TraceMode mode);
bool createDifferentialSceneFromZemax(Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, TraceMode mode);
bool createSceneFromZemax(Group *oGroupPtr, FILE *hfile, RayField ***rayFieldPtrPtr, long *sourceNumberPtr, Detector ***detPtrPtr, long *detNumberPtr, TraceMode mode);

/**
 * \detail parseImpArea_Material 
 *
 * parses the importance area for a material
 *
 * \param[in] parseResultStruct* parseResults, int k
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
inline void parseImpArea_Material(parseResultStruct* parseResults, int k)
{
	// if we have an importance area defined, parse it
	if (parseResults->geometryParams[k].materialParams.importanceArea)
	{
		// if we have an importance object defined, we read its aperture data and use it for our importance area
		// we need to do this here and not in the materials processParseResulst() function as we need to have acces to the aperture data of the other objects in the scene
		if (parseResults->geometryParams[k].materialParams.importanceObjNr > -1 )
		{
			parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].apertureHalfWidth1;
			parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].root;
			parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].tilt;
			parseResults->geometryParams[k].materialParams.importanceAreaApertureType=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].aperture;
		}
		else //if we have a cone angle defined, we calculate our importance area accordingly
		{
			// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
			parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.x=(tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.x)-tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.x))/2;
			parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.y=(tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.y)-tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.y))/2;
			parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[k].tilt;
			// calc the tilt angle of the centre of the cone
			double3 rayAngleCentre=make_double3((parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.x+parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.x)/2,(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.y+parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.y)/2,0);
			double3 dirImpAreaCentre=make_double3(0,0,1);
			rotateRay(&dirImpAreaCentre, rayAngleCentre);
			parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[k].root+dirImpAreaCentre/dot(parseResults->geometryParams[k].normal, dirImpAreaCentre);
			parseResults->geometryParams[k].materialParams.importanceAreaApertureType=AT_ELLIPT;
		}
	}
	else // if we don't have an importance area, we set type to infty to signal it and we set an importance area that corresponds to the full hemisphere
	{
		parseResults->geometryParams[k].materialParams.importanceAreaApertureType=AT_INFTY;
		// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
		parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.x=DOUBLE_MAX;
		parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.y=DOUBLE_MAX;
		parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[k].root+parseResults->geometryParams[k].normal;
		parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[k].tilt;
	}

};

/**
 * \detail parseImpArea_Source 
 *
 * parses the importance area for a source
 *
 * \param[in] parseResultStruct* parseResults, int k
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
inline void parseImpArea_Source(parseResultStruct* parseResults, int k)
{
	// if we have an importance area defined, parse it
	if (parseResults->sourceParams[k].importanceArea)
	{
		if (parseResults->sourceParams[k].rayDirDistr == RAYDIR_RAND_RECT)
			parseResults->sourceParams[k].rayDirDistr = RAYDIR_RANDIMPAREA;
		// if we have an importance object defined, we read its aperture data and use it for our importance area
		// we need to do this here and not in the materials processParseResulst() function as we need to have acces to the aperture data of the other objects in the scene
		if (parseResults->sourceParams[k].importanceObjNr > -1 )
		{
			parseResults->sourceParams[k].importanceAreaHalfWidth=parseResults->geometryParams[parseResults->sourceParams[k].importanceObjNr].apertureHalfWidth1;
			parseResults->sourceParams[k].importanceAreaRoot=parseResults->geometryParams[parseResults->sourceParams[k].importanceObjNr].root;
			parseResults->sourceParams[k].importanceAreaTilt=parseResults->geometryParams[parseResults->sourceParams[k].importanceObjNr].tilt;
			parseResults->sourceParams[k].importanceAreaApertureType=parseResults->geometryParams[parseResults->sourceParams[k].importanceObjNr].aperture;
		}
		else //if we have a cone angle defined, we calculate our importance area accordingly
		{
			// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
			parseResults->sourceParams[k].importanceAreaHalfWidth.x=(tan(parseResults->sourceParams[k].importanceConeAlphaMax.x)-tan(parseResults->sourceParams[k].importanceConeAlphaMin.x))/2;
			parseResults->sourceParams[k].importanceAreaHalfWidth.y=(tan(parseResults->sourceParams[k].importanceConeAlphaMax.y)-tan(parseResults->sourceParams[k].importanceConeAlphaMin.y))/2;
			parseResults->sourceParams[k].importanceAreaTilt=parseResults->geometryParams[k].tilt;
			// calc the tilt angle of the centre of the cone
			double3 rayAngleCentre=make_double3((parseResults->sourceParams[k].importanceConeAlphaMax.x+parseResults->sourceParams[k].importanceConeAlphaMin.x)/2,(parseResults->sourceParams[k].importanceConeAlphaMax.y+parseResults->sourceParams[k].importanceConeAlphaMin.y)/2,0);
			double3 dirImpAreaCentre=make_double3(0,0,1);
			rotateRay(&dirImpAreaCentre, rayAngleCentre);
			parseResults->sourceParams[k].importanceAreaRoot=parseResults->sourceParams[k].root+dirImpAreaCentre/dot(parseResults->sourceParams[k].normal, dirImpAreaCentre);
			parseResults->sourceParams[k].importanceAreaApertureType=AT_ELLIPT;
		}
	}
	else // if we don't have an importance area, we set type to infty to signal it and we set an importance area that corresponds to the full hemisphere
	{
		parseResults->sourceParams[k].importanceAreaApertureType=AT_INFTY;
		// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
		parseResults->sourceParams[k].importanceAreaHalfWidth.x=DOUBLE_MAX;
		parseResults->sourceParams[k].importanceAreaHalfWidth.y=DOUBLE_MAX;
		parseResults->sourceParams[k].importanceAreaRoot=parseResults->sourceParams[k].root+parseResults->sourceParams[k].normal;
		parseResults->sourceParams[k].importanceAreaTilt=parseResults->sourceParams[k].tilt;
	}

};

/**
 * \detail parseImpArea_Det 
 *
 * parses the importance area for a detector
 *
 * \param[in] parseResultStruct* parseResults, int k
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
inline void parseImpArea_Det(parseResultStruct* parseResults, int k)
{
	// if we have an importance area defined, parse it
	if (parseResults->detectorParams[k].importanceArea)
	{
		if (parseResults->detectorParams[k].rayDirDistr == RAYDIR_RAND_RECT)
			parseResults->detectorParams[k].rayDirDistr = RAYDIR_RANDIMPAREA;
		// if we have an importance object defined, we read its aperture data and use it for our importance area
		// we need to do this here and not in the materials processParseResulst() function as we need to have acces to the aperture data of the other objects in the scene
		if (parseResults->detectorParams[k].importanceObjNr > -1 )
		{
			parseResults->detectorParams[k].importanceAreaHalfWidth=parseResults->geometryParams[parseResults->detectorParams[k].importanceObjNr].apertureHalfWidth1;
			parseResults->detectorParams[k].importanceAreaRoot=parseResults->geometryParams[parseResults->detectorParams[k].importanceObjNr].root;
			parseResults->detectorParams[k].importanceAreaTilt=parseResults->geometryParams[parseResults->detectorParams[k].importanceObjNr].tilt;
			parseResults->detectorParams[k].importanceAreaApertureType=parseResults->geometryParams[parseResults->detectorParams[k].importanceObjNr].aperture;
		}
		else //if we have a cone angle defined, we calculate our importance area accordingly
		{
			// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
			parseResults->detectorParams[k].importanceAreaHalfWidth.x=(tan(parseResults->detectorParams[k].importanceConeAlphaMax.x)-tan(parseResults->detectorParams[k].importanceConeAlphaMin.x))/2;
			parseResults->detectorParams[k].importanceAreaHalfWidth.y=(tan(parseResults->detectorParams[k].importanceConeAlphaMax.y)-tan(parseResults->detectorParams[k].importanceConeAlphaMin.y))/2;
			parseResults->detectorParams[k].importanceAreaTilt=parseResults->geometryParams[k].tilt;
			// calc the tilt angle of the centre of the cone
			double3 rayAngleCentre=make_double3((parseResults->detectorParams[k].importanceConeAlphaMax.x+parseResults->detectorParams[k].importanceConeAlphaMin.x)/2,(parseResults->detectorParams[k].importanceConeAlphaMax.y+parseResults->detectorParams[k].importanceConeAlphaMin.y)/2,0);
			double3 dirImpAreaCentre=make_double3(0,0,1);
			rotateRay(&dirImpAreaCentre, rayAngleCentre);
			parseResults->detectorParams[k].importanceAreaRoot=parseResults->detectorParams[k].root+dirImpAreaCentre/dot(parseResults->detectorParams[k].normal, dirImpAreaCentre);
			parseResults->detectorParams[k].importanceAreaApertureType=AT_ELLIPT;
		}
	}
	else // if we don't have an importance area, we set type to infty to signal it and we set an importance area that corresponds to the full hemisphere
	{
		parseResults->detectorParams[k].importanceAreaApertureType=AT_INFTY;
		// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
		parseResults->detectorParams[k].importanceAreaHalfWidth.x=DOUBLE_MAX;
		parseResults->detectorParams[k].importanceAreaHalfWidth.y=DOUBLE_MAX;
		parseResults->detectorParams[k].importanceAreaRoot=parseResults->detectorParams[k].root+parseResults->detectorParams[k].normal;
		parseResults->detectorParams[k].importanceAreaTilt=parseResults->detectorParams[k].tilt;
	}

};

#endif