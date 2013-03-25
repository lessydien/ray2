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

/**\file GaussBeamRayField.cpp
* \brief Rayfield for gaussian beam tracing
* 
*           
* \author Mauch
*/

#include "GaussBeamRayField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Geometry.h"
#include "math.h"

/**
 * \detail getRayListLength 
 *
 * \param[in] void
 * 
 * \return unsigned long long
 * \sa 
 * \remarks 
 * \author Mauch
 */
unsigned long long GaussBeamRayField::getRayListLength(void)
{
	return this->rayListLength;
};

/**
 * \detail setRay 
 *
 * \param[in] gaussBeamRayStruct ray, unsigned long long index
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GaussBeamRayField::setRay(gaussBeamRayStruct ray, unsigned long long index)
{
	if (index <= this->rayListLength)
	{
		rayList[index]=ray;
		return FIELD_NO_ERR;
	}
	else
	{
		return FIELD_INDEXOUTOFRANGE_ERR;
	}
};

/**
 * \detail getRay 
 *
 * \param[in] unsigned long long index
 * 
 * \return gaussBeamRayStruct*
 * \sa 
 * \remarks 
 * \author Mauch
 */
gaussBeamRayStruct* GaussBeamRayField::getRay(unsigned long long index)
{
	if (index <= this->rayListLength)
	{
		return &rayList[index];	
	}
	else
	{
		return 0;
	}
};

/**
 * \detail getRayList 
 *
 * \param[in] void
 * 
 * \return gaussBeamRayStruct*
 * \sa 
 * \remarks 
 * \author Mauch
 */
gaussBeamRayStruct* GaussBeamRayField::getRayList(void)
{
	return &rayList[0];	
};

/* functions for GPU usage */

//void GaussBeamRayField::setPathToPtx(char* path)
//{
//	memcpy(this->path_to_ptx_rayGeneration, path, sizeof(this->path_to_ptx_rayGeneration));
//};
//
//const char* GaussBeamRayField::getPathToPtx(void)
//{
//	return this->path_to_ptx_rayGeneration;
//};

/**
 * \detail createOptixInstance 
 *
 * \param[in] RTcontext &context, simMode mode, double lambda
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GaussBeamRayField::createOptixInstance(RTcontext &context, simMode mode, double lambda)
{
//    RTvariable radiance_ray_type;
    RTvariable epsilon;
	RTvariable max_depth;
	RTvariable min_flux;
	RTprogram  ray_gen_program;
    /* variables for ray gen program */
    RTvariable origin_max;
	RTvariable origin_min;
	RTvariable number;
	RTvariable rayDir;

	/* Ray generation program */
    RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_rayGeneration, "rayGeneration", &ray_gen_program ), context );
    RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( ray_gen_program, "origin_max", &origin_max ), context );
	RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( ray_gen_program, "origin_min", &origin_min ), context );
	RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( ray_gen_program, "number", &number ), context );
	RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( ray_gen_program, "rayDir", &rayDir ), context );
	double3 rayDirVar, origin_maxVar, origin_minVar;
	unsigned int numberVar;

	origin_maxVar= this->rayParamsPtr->rayPosEnd;	
	origin_minVar=this->rayParamsPtr->rayPosStart;	

	rayDirVar=this->rayParamsPtr->rayDirection;

	numberVar = this->rayParamsPtr->width*this->rayParamsPtr->height;//(unsigned int)rayListLength/100;

//	RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "radiance_ray_type", &radiance_ray_type ) );
    RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "scene_epsilon", &epsilon ), context );
	RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "max_depth", &max_depth ), context );
	RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "min_flux", &min_flux ), context );

//    RT_CHECK_ERROR_NOEXIT( rtVariableSet1ui( radiance_ray_type, 0u ) );
    RT_CHECK_ERROR_NOEXIT( rtVariableSet1f( epsilon, 1.e-4f ), context );
	RT_CHECK_ERROR_NOEXIT( rtVariableSet1i( max_depth, MAX_DEPTH_CPU ), context );
	RT_CHECK_ERROR_NOEXIT( rtVariableSet1f( min_flux, MIN_FLUX_CPU ), context );


	RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(origin_max, sizeof(double3), &origin_maxVar), context );
	RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(origin_min, sizeof(double3), &origin_minVar), context );
	RT_CHECK_ERROR_NOEXIT( rtVariableSet1ui(number, numberVar ), context );

	RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(rayDir, sizeof(double3), &rayDirVar), context );

    RT_CHECK_ERROR_NOEXIT( rtContextSetRayGenerationProgram( context,0, ray_gen_program ), context );

	this->update=false;

	return FIELD_NO_ERR;
};

/**
 * \detail createCPUSimInstance 
 *
 * \param[in] double lambda
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void GaussBeamRayField::createCPUSimInstance(double lambda)
{
	this->update=false;
}

/**
 * \detail traceScene 
 *
 * \param[in] Group &oGroup
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GaussBeamRayField::traceScene(Group &oGroup)
{
	long long index;
	for (index=0; index < this->rayListLength; index++)
	{
	    for(;;) // iterative tracing
		{
			if(!this->rayList[index].running) 
			    break;
			oGroup.trace(rayList[index]);
		}
	}

	return FIELD_NO_ERR;
};

/**
 * \detail writeData2File 
 *
 * \param[in] FILE *hFile, rayDataOutParams outParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GaussBeamRayField::writeData2File(FILE *hFile, rayDataOutParams outParams)
{
	//writeGeomRayData2File(hFile, this->rayList, this->rayListLength, outParams);
	return FIELD_NO_ERR;
};

/**
 * \detail convert2Intensity 
 *
 * \param[in] IntensityField* imagePtr
 * 
 * \return fieldError
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
fieldError GaussBeamRayField::convert2Intensity(IntensityField* imagePtr)
{
	return FIELD_NO_ERR;
};

/**
 * \detail convert2ScalarField 
 *
 * \param[in] ScalarLightField* imagePtr
 * 
 * \return fieldError
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
fieldError GaussBeamRayField::convert2ScalarField(ScalarLightField* imagePtr)
{
	return FIELD_NO_ERR;
};

/**
 * \detail convert2VecField 
 *
 * \param[in] VectorLightField* imagePtr
 * 
 * \return fieldError
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
fieldError GaussBeamRayField::convert2VecField(VectorLightField* imagePtr)
{
	return FIELD_NO_ERR;
};