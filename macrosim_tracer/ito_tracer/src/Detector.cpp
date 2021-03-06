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

/**\file Detector.cpp
* \brief base class of detectors
* 
*           
* \author Mauch
*/

#include "Detector.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"
#include "randomGenerator.h"
#include <ctime>

#include "Parser_XML.h"

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
detError Detector::processParseResults(DetectorParseParamStruct &parseResults_Det)
{
	std::cout << "error in Detector.processParseResults(): not defined for the given detector" << "...\n";
	return DET_ERROR;
};

/**
 * \detail setPathToPtx 
 *
 * sets the path to the ptx file that the .cu file defining the behaviour of the Detector on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Detector::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the Detector on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return const char*
 * \sa 
 * \remarks 
 * \author Mauch
 */
const char* Detector::getPathToPtx(void)
{
	return this->path_to_ptx;
};

/**
 * \detail setDetParamsPtr 
 *
 * \param[in] detParams *paramsPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Detector::setDetParamsPtr(detParams *paramsPtr)
{
	this->detParamsPtr=paramsPtr;
};

/**
 * \detail getDetParamsPtr 
 *
 * \param[in] void
 * 
 * \return detParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
detParams* Detector::getDetParamsPtr(void)
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
detError Detector::detect2TextFile(FILE* hFile, Field* rayFieldPtr)
{
	std::cout << "error in Detector.detect2TextFile(): not defined for the given detector" << "...\n";
	return DET_ERROR;
};

/**
 * \detail detect 
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
detError Detector::detect(Field* FieldPtr, Field **imagePtrPtr)
{
	std::cout << "error in Detector.detect(): not defined for the given detector" << "...\n";
	return DET_ERROR;
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
detError Detector::parseXml(pugi::xml_node &det, vector<Detector*> &detVec)
{

	Parser_XML l_parser;
	const char* l_fileName=l_parser.attrValByName(det, "fileName");
	if (l_fileName==NULL)
	{
		std::cout << "error in Detector.parseXml(): fileName is not defined" << "...\n";
		return DET_ERROR;
	}
	this->getDetParamsPtr()->filenamePtr=(char*)malloc(512*sizeof(char));
	//memcpy(&(this->getDetParamsPtr()->filename[0]),l_fileName, 512*sizeof(char)); 
	//sprintf(this->getDetParamsPtr()->filename, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, l_fileName);
	sprintf(this->getDetParamsPtr()->filenamePtr, "%s", l_fileName);
	//sprintf(this->getDetParamsPtr()->filenamePtr, "%s", "test.txt");
	if (!this->checkParserError(l_parser.attrByNameToDouble(det, "root.x", this->getDetParamsPtr()->root.x)))
		return DET_ERROR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(det, "root.y", this->getDetParamsPtr()->root.y)))
		return DET_ERROR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(det, "root.z", this->getDetParamsPtr()->root.z)))
		return DET_ERROR;
	double l_tilt;
	if (!this->checkParserError(l_parser.attrByNameToDouble(det, "tilt.y", l_tilt)))
		return DET_ERROR;
	this->getDetParamsPtr()->tilt.y=l_tilt/360*2*M_PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(det, "tilt.z", l_tilt)))
		return DET_ERROR;
	this->getDetParamsPtr()->tilt.z=l_tilt/360*2*M_PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(det, "tilt.x", l_tilt)))
		return DET_ERROR;
	this->getDetParamsPtr()->tilt.x=l_tilt/360*2*M_PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(det, "apertureHalfWidth.x", this->getDetParamsPtr()->apertureHalfWidth.x)))
		return DET_ERROR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(det, "apertureHalfWidth.y", this->getDetParamsPtr()->apertureHalfWidth.y)))
		return DET_ERROR;
	this->getDetParamsPtr()->apertureHalfWidth.z=0.0005;
	if (!this->checkParserError(l_parser.attrByNameToDetOutFormat(det, "detOutFormat", this->getDetParamsPtr()->outFormat)))
		return DET_ERROR;
	this->getDetParamsPtr()->MTransform=createTransformationMatrix(this->getDetParamsPtr()->tilt, this->getDetParamsPtr()->root);
	double3 l_vec=make_double3(0,0,1);
	rotateRay(&l_vec,this->getDetParamsPtr()->tilt);
	this->getDetParamsPtr()->normal=l_vec;
	return DET_NO_ERROR;
};

/**
 * \detail checks wether parseing was succesfull and assembles the error message if it was not
 *
 * returns the coordinates of the minimum corner of the bounding box of the surface
 *
 * \param[in] char *msg
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Detector::checkParserError(char *msg)
{
	if (msg==NULL)
		return true;
	else
	{
		cout << "error in Detector.parseXML(): " << msg << endl;
		delete msg;
		msg=NULL;
		return false;
	}
};