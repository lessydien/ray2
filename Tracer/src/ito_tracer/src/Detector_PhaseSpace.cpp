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

/**\file Detector_PhaseSpace.cpp
* \brief detector that is detecting the full field representation of the light field
* 
*           
* \author Mauch
*/

#include "Detector_PhaseSpace.h"
#include "ScalarLightField.h"
#include "Field.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include <ctime>
#include <iostream>

/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &det, vector<Detector*> &detVec
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
detError  DetectorPhaseSpace::parseXml(pugi::xml_node &det, vector<Detector*> &detVec)
{
	// call base class function
	if (DET_NO_ERROR != Detector::parseXml(det, detVec))
	{
		std::cout << "error in DetectorPhaseSpace.parseXml(): Detector.parseXml()  returned an error." << std::endl;
		return DET_ERROR;
	}
	std::cout << "error in DetectorPhaseSpace.parseXml(): not implemented yet." << std::endl;
	return DET_ERROR;
};

/**
 * \detail setDetParamsPtr
 *
 * \param[in] detPhaseSpaceParams *paramsPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void DetectorPhaseSpace::setDetParamsPtr(detPhaseSpaceParams *paramsPtr)
{
	this->detParamsPtr=paramsPtr;
};

/**
 * \detail getDetParamsPtr
 *
 * \param[in] void
 * 
 * \return detPhaseSpaceParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
detPhaseSpaceParams* DetectorPhaseSpace::getDetParamsPtr(void)
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
detError DetectorPhaseSpace::detect2TextFile(FILE* hFile, RayField* rayFieldPtr)
{
	Field *l_imagePtr=new ScalarLightField();
	Field **l_imagePtrPtr=&l_imagePtr;
	if (FIELD_NO_ERR != this->detect(rayFieldPtr, l_imagePtrPtr) )
	{
		std::cout << "error in DetectorPhaseSpace.detect2TextFile(): detect() returned an error" << std::endl;
		return DET_ERROR;
	}
	ScalarLightField *l_ScalarImagePtr=dynamic_cast<ScalarLightField*>(l_imagePtr);
	if ( FIELD_NO_ERR != writeScalarField2File(hFile, l_ScalarImagePtr) )
	{
		std::cout << "error in DetectorIntensity.detect2TextFile(): writeIntensityFIeld2File() returned an error" << std::endl;
		return DET_ERROR;
	}

	return DET_NO_ERROR;
};

/**
 * \detail detect 
 *
 * converts rayfield to scalar Field
 *
 * \param[in] RayField* rayFieldPtr
 * \param[out] Field *imagePtr
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
detError DetectorPhaseSpace::detect(Field* rayFieldPtr, Field **imagePtrPtr)
{
	// check wether there is already an image
	if (*imagePtrPtr != NULL)
	{
		// if the params of the image do not agree with the params of the Detector we have to raise an error
		if ( ((*imagePtrPtr)->getParamsPtr()->nrPixels.x != this->detParamsPtr->detPixel.x)
			|| ((*imagePtrPtr)->getParamsPtr()->nrPixels.y != this->detParamsPtr->detPixel.y)
			|| ((*imagePtrPtr)->getParamsPtr()->nrPixels.z != 1)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m11 != this->detParamsPtr->MTransform.m11)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m12 != this->detParamsPtr->MTransform.m12)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m13 != this->detParamsPtr->MTransform.m13)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m14 != this->detParamsPtr->MTransform.m14)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m21 != this->detParamsPtr->MTransform.m21)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m22 != this->detParamsPtr->MTransform.m22)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m23 != this->detParamsPtr->MTransform.m23)
			|| ((*imagePtrPtr)->getParamsPtr()->MTransform.m24 != this->detParamsPtr->MTransform.m24) )
		{
			std::cout << "error in DetectorPhaseSpace.detect: given ScalarField does not match parameters of detector" << std::endl;
			return DET_ERROR;
		}
	}
	else
	{
		// create new Intensity Field
		phaseSpaceParams imageParams;
		imageParams.MTransform=this->detParamsPtr->MTransform;
		imageParams.lambda=rayFieldPtr->getParamsPtr()->lambda;
		//imageParams.nrPixels=make_long3(this->detParamsPtr->apertureHalfWidth.x, this->detParamsPtr->apertureHalfWidth.y ,1);
		imageParams.nrPixels=make_long3(this->detParamsPtr->detPixel.x, this->detParamsPtr->detPixel.y ,1);
		imageParams.nrPixels_PhaseSpace=this->detParamsPtr->detPixel_PhaseSpace;
		imageParams.dirHalfWidth=this->detParamsPtr->dirHalfWidth;
		imageParams.dirHalfWidth=this->detParamsPtr->dirHalfWidth;
		if (imageParams.nrPixels.x<1)
		{
			std::cout << "error in DetectorPhaseSpace.detect: pixel number smaller than 1 in x is not allowed" << std::endl;
			return DET_ERROR;
		}
		else
			imageParams.scale.x=2*this->detParamsPtr->apertureHalfWidth.x/(imageParams.nrPixels.x);
		if (imageParams.nrPixels.y<1)
		{
			std::cout << "error in DetectorPhaseSpace.detect: pixel number smaller than one in y is not allowed" << std::endl;
			return DET_ERROR;
		}
		else
			imageParams.scale.y=2*this->detParamsPtr->apertureHalfWidth.y/(imageParams.nrPixels.y);
		if (imageParams.nrPixels.z<1)
		{
			std::cout << "error in DetectorPhaseSpace.detect: pixel number smaller than one in z is not allowed" << std::endl;
			return DET_ERROR;
		}
		else
		{
			if (imageParams.nrPixels.z!=1)
			{
				// 3 dimensional IntensityFields are not implemented yet !!!
				imageParams.scale.z=0.02/(imageParams.nrPixels.z); // we calculate a 2dimensional field here anyway
				std::cout << "error in DetectorPhaseSpace.detect: 3dimensional fields are not implemented yet" << std::endl;
				return DET_ERROR;
			}
			imageParams.scale.z=0.02; // we calculate a 2dimensional field here anyway
		}
		if (imageParams.nrPixels_PhaseSpace.x<1)
		{
			std::cout << "error in DetectorPhaseSpace.detect: pixel number smaller than 1 in dirX is not allowed" << std::endl;
			return DET_ERROR;
		}
		else
			imageParams.scale_dir.x=2*imageParams.dirHalfWidth.x/(imageParams.nrPixels_PhaseSpace.x);
		if (imageParams.nrPixels_PhaseSpace.y<1)
		{
			std::cout << "error in DetectorPhaseSpace.detect: pixel number smaller than 1 in dirY is not allowed" << std::endl;
			return DET_ERROR;
		}
		else
			imageParams.scale_dir.y=2*imageParams.dirHalfWidth.y/(imageParams.nrPixels_PhaseSpace.y);

		imageParams.units.x=metric_mm;
		imageParams.units.y=metric_mm;
		imageParams.units.z=metric_mm;
		imageParams.unitLambda=metric_mm;
		*imagePtrPtr=new PhaseSpaceField(imageParams);
	}
		
	if ( FIELD_NO_ERR != rayFieldPtr->convert2PhaseSpace(*imagePtrPtr, *(this->detParamsPtr)) )
	{
		std::cout << "error in Detector_PhaseSpace.detect(): convert2PhaseSpace() returned an error" << std::endl;
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
detError DetectorPhaseSpace::processParseResults(DetectorParseParamStruct &parseResults_Det)
{
	//this->detParamsPtr=new detPhaseSpaceParams;
//	this->detParamsPtr->apertureHalfWidth=parseResults_Det.apertureHalfWidth;
//	this->detParamsPtr->detPixel=parseResults_Det.detPixel;
	this->detParamsPtr->normal=parseResults_Det.normal;
	this->detParamsPtr->root=parseResults_Det.root;
//	this->detParamsPtr->rotNormal=parseResults_Det.rotNormal;
	this->detParamsPtr->outFormat=DET_OUT_MAT;
	this->detParamsPtr->MTransform=createTransformationMatrix(parseResults_Det.tilt, this->detParamsPtr->root);
	this->detParamsPtr->detPixel_PhaseSpace=parseResults_Det.detPixel_PhaseSpace;
	this->detParamsPtr->dirHalfWidth=parseResults_Det.dirHalfWidth;

	return DET_NO_ERROR;
}