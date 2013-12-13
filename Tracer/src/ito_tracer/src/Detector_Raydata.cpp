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

/**\file Detector_Raydata.cpp
* \brief detector that is detecting the raydata representing the lightfield
* 
*           
* \author Mauch
*/

#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include <ctime>
#include "Detector_Raydata.h"
#include "Parser_XML.h"

/**
 * \detail getDetParamsPtr 
 *
 * writes field representation to file
 *
 * \param[in] void
 * 
 * \return detRaydataParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
void DetectorRaydata::setDetParamsPtr(detRaydataParams *paramsPtr)
{
	this->detParamsPtr=paramsPtr;
};

/**
 * \detail getDetParamsPtr 
 *
 * writes field representation to file
 *
 * \param[in] void
 * 
 * \return detRaydataParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
detRaydataParams* DetectorRaydata::getDetParamsPtr(void)
{
	return this->detParamsPtr;
};

/**
 * \detail detect2TextFile 
 *
 * writes field representation to file
 *
 * \param[in] FILE* hFile, RayField* rayFieldPtr
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
detError DetectorRaydata::detect2TextFile(FILE* hFile, RayField* rayFieldPtr)
{
	//rayDataOutParams outParams;
	//	outParams.reducedData=this->detParamsPtr->reduceData;
	//	outParams.ID=this->detParamsPtr->geomID;
	//	if (FIELD_NO_ERR != rayFieldPtr->writeData2File(hFile, outParams) )
	//		return DET_ERROR;
	return DET_NO_ERROR;
};

/**
 * \detail detect 

 *
 * \param[in] RayField* rayFieldPtr
 * \param[out] Field *imagePtr
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
detError DetectorRaydata::detect(Field* rayFieldPtr, Field **imagePtrPtr)
{
	if (FIELD_NO_ERR!=rayFieldPtr->convert2RayData(imagePtrPtr, *(this->detParamsPtr)) )
	{
		std::cout << "error in DetectorRaydata.detect(): convert2RayData() returned an error" << std::endl;
		return DET_ERROR;
	}
	//GeometricRayField* l_RayFieldPtr=dynamic_cast<GeometricRayField*>(imagePtrPtr);

	char filepath[512];
	sprintf(filepath, "%s", OUTPUT_FILEPATH);
	if (FIELD_NO_ERR != rayFieldPtr->write2File(filepath, *(this->detParamsPtr)) )
	{
		std::cout << "error in DetectorRaydata.detect(): Field.write2File() returned an error" << std::endl;
		return DET_ERROR;
	}

	return DET_NO_ERROR;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] DetectorParseParamStruct &parseResults_Det
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
detError DetectorRaydata::processParseResults(DetectorParseParamStruct &parseResults_Det)
{
	//this->detParamsPtr=new detRaydataParams;
//	this->detParamsPtr->apertureHalfWidth=parseResults_Det.apertureHalfWidth;
//	this->detParamsPtr->detPixel=parseResults_Det.detPixel;
	this->detParamsPtr->normal=parseResults_Det.normal;
	this->detParamsPtr->root=parseResults_Det.root;
//	this->detParamsPtr->rotNormal=parseResults_Det.rotNormal;
	this->detParamsPtr->outFormat=DET_OUT_TEXT;
	this->detParamsPtr->MTransform=createTransformationMatrix(parseResults_Det.tilt, this->detParamsPtr->root);
	switch (parseResults_Det.detectorType)
	{
		case DET_RAYDATA:
			// set params that are specific here
			this->detParamsPtr->reduceData=false;
			this->detParamsPtr->geomID=parseResults_Det.geomID;
			break;
		case DET_RAYDATA_GLOBAL:
			this->detParamsPtr->reduceData=false;
			this->detParamsPtr->geomID=-1;
			break;
		case DET_RAYDATA_RED:
			this->detParamsPtr->reduceData=true;
			this->detParamsPtr->geomID=parseResults_Det.geomID;
			break;
		case DET_RAYDATA_RED_GLOBAL:
			this->detParamsPtr->reduceData=true;
			this->detParamsPtr->geomID=-1;
			break;
		default:
			std::cout << "error in DetectorRaydata.processParseResults(): unknown detector type" << std::endl;
			return DET_ERROR;
			break;
	}

	return DET_NO_ERROR;
};

/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &det
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
detError DetectorRaydata::parseXml(pugi::xml_node &det, vector<Detector*> &detVec)
{
	// parse base class
	if (DET_NO_ERROR != Detector::parseXml(det, detVec))
	{
		std::cout << "error in PlaneSurface.parseXml(): Geometry.parseXml() returned an error." << std::endl;
		return DET_ERROR;
	}

	Parser_XML l_parser;

	bool l_listAllRays;
	if (!this->checkParserError(l_parser.attrByNameToBool(det, "listAllRays", l_listAllRays)))
		return DET_ERROR;
	if (l_listAllRays)
	{
		this->getDetParamsPtr()->geomID=-1;
	}
	else
	{
		std::cout << "error in Detector.parseXml(): listAllRays equal to false is not implemented yet..." << std::endl;
		return DET_ERROR;
	}

	if (!this->checkParserError(l_parser.attrByNameToBool(det, "reduceData", this->getDetParamsPtr()->reduceData)))
		return DET_ERROR;

	return DET_NO_ERROR;
};