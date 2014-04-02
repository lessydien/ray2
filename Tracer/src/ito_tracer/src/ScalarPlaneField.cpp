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

/**\file ScalarLightField.cpp
* \brief scalar representation of light field
* 
*           
* \author Mauch
*/

#include "ScalarPlaneField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"
#ifdef _MATSUPPORT
	#include "mat.h"
#endif
#include "Parser_XML.h"


fieldParams* ScalarPlaneField::getParamsPtr()
{
	return (this->paramsPtr);
};

/**
 * \detail initGPUSubset 
 *
 * \param[in] RTcontext &context
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarPlaneField::initGPUSubset(RTcontext &context)
{
	std::cout << "error in ScalarLightField.initGPUSubset(): not defined for the given field representation" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail initCPUSubset 
 *
 * \param[in] 
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarPlaneField::initCPUSubset()
{
//	std::cout << "error in ScalarLightField.initCPUSubset(): not defined for the given field representation" << "...\n";
	return FIELD_NO_ERR;
};

/**
 * \detail createCPUSimInstance 
 *
 * \param[in] 
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarPlaneField::createCPUSimInstance()
{
	if (ScalarLightField::U == NULL)
	{
		fftw_complex *in=(fftw_complex*) fftw_malloc(sizeof(fftw_complex) * paramsPtr->nrPixels.x*paramsPtr->nrPixels.y*paramsPtr->nrPixels.z);
		if (!in)
		{
			std::cout << "error in ScalarPlaneField.createCPUSimInstance(): memory could not be allocated" << "...\n";
			return FIELD_ERR;
		}
		ScalarLightField::U=reinterpret_cast<complex<double>*>(in);
		double x=-(paramsPtr->nrPixels.x-1)/2*paramsPtr->scale.x;
		double y=-(paramsPtr->nrPixels.y-1)/2*paramsPtr->scale.y;
		double z=-(paramsPtr->nrPixels.z-1)/2*paramsPtr->scale.z;
		// init to zero
		for (unsigned long jx=0;jx<paramsPtr->nrPixels.x;jx++)
		{
			y=-(paramsPtr->nrPixels.y-1)/2*paramsPtr->scale.y;
			x=x+paramsPtr->scale.x;
			for (unsigned long jy=0;jy<paramsPtr->nrPixels.y;jy++)
			{
				y=y+paramsPtr->scale.y;
				for (unsigned long jz=0;jz<paramsPtr->nrPixels.z;jz++)
				{
					if ( (abs(x)<=this->paramsPtr->fieldWidth.x) && (abs(y)<=this->paramsPtr->fieldWidth.y) )
						ScalarLightField::U[jx+jy*paramsPtr->nrPixels.x+jz*paramsPtr->nrPixels.y]=paramsPtr->amplMax;
					else
						ScalarLightField::U[jx+jy*paramsPtr->nrPixels.x+jz*paramsPtr->nrPixels.y]=0;
				}
			}
		}
	}
};

/**
 * \detail createOptixInstance 
 *
 * \param[in] RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarPlaneField::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	std::cout << "error in ScalarLightField.createOptixInstance(): not defined for the given field representation" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail initSimulation 
 *
 * \param[in] Group &oGroup, simAssParams &params
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarPlaneField::initSimulation(Group &oGroup, simAssParams &params)
{
	if (params.RunOnCPU)
	{
		if (FIELD_NO_ERR!=this->createCPUSimInstance())
		{
			std::cout <<"error in ScalarPlaneField.createOptixInstance(): create CPUSimInstance() returned an error." << "...\n";
			return FIELD_ERR;
		}
		if (GROUP_NO_ERR != oGroup.createCPUSimInstance(this->getParamsPtr()->lambda, params.simParams) )
		{
			std::cout << "error in RayField.initSimulation(): group.createCPUSimInstance() returned an error" << "...\n";
			return FIELD_ERR;
		}
	}
	else
	{
		//if (FIELD_NO_ERR != this->createOptiXContext())
		//{
		//	std::cout << "error in RayField.initSimulation(): createOptiXInstance() returned an error" << "...\n";
		//	return FIELD_ERR;
		//}
		//// convert geometry to GPU code
		//if ( GROUP_NO_ERR != oGroup.createOptixInstance(context, params.mode, this->getParamsPtr()->lambda) )
		//{
		//	std::cout << "error in RayField.initSimulation(): group.createOptixInstance returned an error" << "...\n";
		//	return ( FIELD_ERR );
		//}
		//	// convert rayfield to GPU code
		//	if ( FIELD_NO_ERR != this->createOptixInstance(context, output_buffer_obj, seed_buffer_obj) )
		//	{
		//		std::cout << "error in RayField.initSimulation(): SourceList[i]->createOptixInstance returned an error at index:" << 0 << "...\n";
		//		return ( FIELD_ERR );
		//	}
		//	if (!RT_CHECK_ERROR_NOEXIT( rtContextValidate( context ), context ))
		//		return FIELD_ERR;
		//	if (!RT_CHECK_ERROR_NOEXIT( rtContextCompile( context ), context ))
		//		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
}

/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &field, vector<Field*> &fieldVec
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  ScalarPlaneField::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams)
{
	// call base class function
	if (FIELD_NO_ERR != ScalarLightField::parseXml(field, fieldVec, simParams))
	{
		std::cout << "error in ScalarPlaneField.parseXml(): ScalarLightField.parseXml()  returned an error." << "...\n";
		return FIELD_ERR;
	}
	Parser_XML l_parser;
	if (!l_parser.attrByNameToDouble(field, "fieldWidth.x", this->paramsPtr->fieldWidth.x))
	{
		std::cout << "error in ScalarGaussianField.parseXml(): fieldWidth.x is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "fieldWidth.y", this->paramsPtr->fieldWidth.y))
	{
		std::cout << "error in ScalarGaussianField.parseXml(): fieldWidth.y is not defined" << "...\n";
		return FIELD_ERR;
	}

	return FIELD_NO_ERR;
};