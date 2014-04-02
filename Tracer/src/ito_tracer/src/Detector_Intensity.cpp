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

/**\file Detector_Inetensity.cpp
* \brief Detector that is detecting the intensity of the light field
* 
*           
* \author Mauch
*/

#include "Detector_Intensity.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include <ctime>
#include "Parser_XML.h"

/**
 * \detail setDetParamsPtr
 *
 * \param[in] detIntensityParams *paramsPtr
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void DetectorIntensity::setDetParamsPtr(detIntensityParams *paramsPtr)
{
	this->detParamsPtr=paramsPtr;
};

/**
 * \detail getDetParamsPtr
 *
 * \param[in] void
 * 
 * \return detIntensityParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
detIntensityParams* DetectorIntensity::getDetParamsPtr(void)
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
detError DetectorIntensity::detect2TextFile(FILE* hFile, RayField* rayFieldPtr)
{
	Field *l_imagePtr=new IntensityField();
	Field **l_imagePtrPtr=&l_imagePtr;
		
	if ( DET_NO_ERROR != this->detect(rayFieldPtr, l_imagePtrPtr) )
	{
		std::cout << "error in DetectorIntensity.detect2TextFile(): detect() returned an error" << "...\n";
		return DET_ERROR;
	}
	IntensityField *l_IntensityImagePtr=dynamic_cast<IntensityField*>(l_imagePtr);
	if ( IO_NO_ERR != writeIntensityField2File(hFile, l_IntensityImagePtr) )
	{
		std::cout << "error in DetectorIntensity.detect2TextFile(): writeIntensityFIeld2File() returned an error" << "...\n";
		return DET_ERROR;
	}
	// clean up
	delete *l_imagePtrPtr;
	(*l_imagePtrPtr)=NULL;
	delete l_imagePtrPtr;
	l_imagePtrPtr=NULL;

	return DET_NO_ERROR;
};

/**
 * \detail detect 
 *
 * converts rayfield to Intensity
 *
 * \param[in] RayField* rayFieldPtr
 * \param[out] IntensityField *imagePtr
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
detError DetectorIntensity::detect(Field* fieldPtr, Field **imagePtrPtr)
{
	// check wether there is already an image
	if (*imagePtrPtr != NULL)
	{
		// if the params of the image do not agree with the params of the Detector we have to raise an error
		if ( ((*imagePtrPtr)->getParamsPtr()->nrPixels.x != this->detParamsPtr->detPixel.x)
			|| ((*imagePtrPtr)->getParamsPtr()->nrPixels.y != this->detParamsPtr->detPixel.y)
			|| ((*imagePtrPtr)->getParamsPtr()->nrPixels.z != fieldPtr->getParamsPtr()->nrPseudoLambdas) // we code spectral information in the z-index of the resulting field
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m11 != this->detParamsPtr->MTransform.m11)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m12 != this->detParamsPtr->MTransform.m12)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m13 != this->detParamsPtr->MTransform.m13)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m14 != this->detParamsPtr->MTransform.m14)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m21 != this->detParamsPtr->MTransform.m21)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m22 != this->detParamsPtr->MTransform.m22)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m23 != this->detParamsPtr->MTransform.m23)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m24 != this->detParamsPtr->MTransform.m24) )
		{
			std::cout << "error in DetectorIntensity.detect: given IntensityField does not match parameters of detector" << "...\n";
			return DET_ERROR;
		}
	}
	else
	{
		// create new Intensity Field
		fieldParams imageParams;
		imageParams.MTransform=this->detParamsPtr->MTransform;
		imageParams.lambda=fieldPtr->getParamsPtr()->lambda;
		//imageParams.nrPixels=make_long3(this->detParamsPtr->detPixel.x, this->detParamsPtr->detPixel.y ,1);
		imageParams.nrPixels=make_long3(this->detParamsPtr->detPixel.x, this->detParamsPtr->detPixel.y , fieldPtr->getParamsPtr()->nrPseudoLambdas);
		if (imageParams.nrPixels.x<1)
		{
			std::cout << "error in DetectorIntensity.detect: pixel number smaller than 1 in x is not allowed" << "...\n";
			return DET_ERROR;
		}
		else
			imageParams.scale.x=2*this->detParamsPtr->apertureHalfWidth.x/(imageParams.nrPixels.x);
		if (imageParams.nrPixels.y<1)
		{
			std::cout << "error in DetectorIntensity.detect: pixel number smaller than one in y is not allowed" << "...\n";
			return DET_ERROR;
		}
		else
			imageParams.scale.y=2*this->detParamsPtr->apertureHalfWidth.y/(imageParams.nrPixels.y);
		if (imageParams.nrPixels.z<1)
		{
			std::cout << "error in DetectorIntensity.detect: pixel number smaller than one in z is not allowed" << "...\n";
			return DET_ERROR;
		}
		else
		{
			if (imageParams.nrPixels.z!=fieldPtr->getParamsPtr()->nrPseudoLambdas)
			{
				// 3 dimensional IntensityFields are not implemented yet !!!
				std::cout << "error in DetectorIntensity.detect: 3dimensional fields are not implemented yet" << "...\n";
				return DET_ERROR;
			}
			imageParams.scale.z=fieldPtr->getParamsPtr()->pseudoBandwidth/fieldPtr->getParamsPtr()->nrPseudoLambdas; // calc spectral scaling
		}
		imageParams.units.x=metric_mm;
		imageParams.units.y=metric_mm;
		imageParams.units.z=metric_mm;
		imageParams.unitLambda=metric_mm;
		*imagePtrPtr=new IntensityField(imageParams);
	}
	if ( FIELD_NO_ERR != fieldPtr->convert2Intensity(*imagePtrPtr, *(this->detParamsPtr)) )
	{
		std::cout << "error in Detector_Intensity.detect(): convert2Intensity() returned an error" << "...\n";
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
detError DetectorIntensity::processParseResults(DetectorParseParamStruct &parseResults_Det)
{
	//this->detParamsPtr=new detIntensityParams;
//	this->detParamsPtr->apertureHalfWidth=parseResults_Det.apertureHalfWidth;//make_double2(0.5,0.5);//
//	this->detParamsPtr->detPixel=parseResults_Det.detPixel;
	this->detParamsPtr->normal=parseResults_Det.normal;
	this->detParamsPtr->root=parseResults_Det.root;
//	this->detParamsPtr->rotNormal=parseResults_Det.rotNormal;
	this->detParamsPtr->outFormat=DET_OUT_MAT;
	this->detParamsPtr->MTransform=createTransformationMatrix(parseResults_Det.tilt, this->detParamsPtr->root);
	this->detParamsPtr->tilt=parseResults_Det.tilt;

	return DET_NO_ERROR;
}

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
detError DetectorIntensity::parseXml(pugi::xml_node &det, vector<Detector*> &detVec)
{
	// parse base class
	if (DET_NO_ERROR != Detector::parseXml(det, detVec))
	{
		std::cout << "error in PlaneSurface.parseXml(): Geometry.parseXml() returned an error." << "...\n";
		return DET_ERROR;
	}

	Parser_XML l_parser;

	if (!this->checkParserError(l_parser.attrByNameToLong(det, "detPixel.x", this->getDetParamsPtr()->detPixel.x)))
		return DET_ERROR;
	if (!this->checkParserError(l_parser.attrByNameToLong(det, "detPixel.y", this->getDetParamsPtr()->detPixel.y)))
		return DET_ERROR;
	if (!this->checkParserError(l_parser.attrByNameToInt(det, "ignoreDepth", this->getDetParamsPtr()->ignoreDepth)))
		return DET_ERROR;

	return DET_NO_ERROR;
};