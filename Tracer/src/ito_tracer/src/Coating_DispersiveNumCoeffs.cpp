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

/**\file Coating_DispersiveNumCoeffs.cpp
* \brief Coating with predefined coefficients for transmissione and reflection
* 
*           
* \author Mauch
*/

#include "Coating_DispersiveNumCoeffs.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <string.h>
#include <iostream>

/**
 * \detail createCPUInstance
 *
 * \param[in] double lambda
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_DispersiveNumCoeffs::createCPUSimInstance(double lambda)
{
	double l_lambda=lambda*1e3; // we need lambda in um here...
	this->reducedParamsPtr->t=this->fullParamsPtr->a_t*l_lambda*l_lambda+this->fullParamsPtr->c_t;
	this->reducedParamsPtr->r=this->fullParamsPtr->a_r*l_lambda*l_lambda+this->fullParamsPtr->c_r;
	this->update=false;
	return COAT_NO_ERROR;	
};

CoatingError Coating_DispersiveNumCoeffs::setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_coatingParamsPtr)
{
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( *closest_hit_programPtr, "coating_params", l_coatingParamsPtr ), context) )
		return COAT_ERROR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(*l_coatingParamsPtr, sizeof(Coating_DispersiveNumCoeffs_ReducedParams), (this->reducedParamsPtr)), context) )
		return COAT_ERROR;

	return COAT_NO_ERROR;
};

/**
 * \detail setFullParams
 *
 * \param[in] Coating_DispersiveNumCoeffs_FullParams* ptrIn
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_DispersiveNumCoeffs::setFullParams(Coating_DispersiveNumCoeffs_FullParams* ptrIn)
{
	this->fullParamsPtr=ptrIn;
	return COAT_NO_ERROR;
};

/**
 * \detail getFullParams
 *
 * \param[in] void
 * 
 * \return Coating_DispersiveNumCoeffs_FullParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_DispersiveNumCoeffs_FullParams* Coating_DispersiveNumCoeffs::getFullParams(void)
{
	return this->fullParamsPtr;
};

/**
 * \detail getReducedParams
 *
 * \param[in] void
 * 
 * \return Coating_DispersiveNumCoeffs_ReducedParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_DispersiveNumCoeffs_ReducedParams* Coating_DispersiveNumCoeffs::getReducedParams(void)
{
	return this->reducedParamsPtr;
};

/**
 * \detail calcCoatingCoeffs
 *
 * calc the transmission and reflection coefficients of the Coating for the given wavelentgh and the incident ray
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
//CoatingError Coating_DispersiveNumCoeffs::calcCoatingCoeffs(double lambda, double3 normal, double3 direction)
//{
//	return COAT_NO_ERROR;
//};

/**
 * \detail hit function of the Coating for geometric rays
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool  Coating_DispersiveNumCoeffs::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	return hitCoatingNumCoeff(ray, hitParams, *this->reducedParamsPtr);
}

/**
 * \detail hit function of the Coating for gaussian beam rays
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Coating_DispersiveNumCoeffs::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
{
	// dummy function to be overwritten by child class
	return true;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] MaterialParseParamStruct &parseResults_Geom
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_DispersiveNumCoeffs::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	this->fullParamsPtr->type=CT_DISPNUMCOEFFS;
//	this->fullParamsPtr->r=parseResults_Mat.coating_r;
//	this->fullParamsPtr->t=parseResults_Mat.coating_t;
	this->fullParamsPtr->a_r=parseResults_Mat.coating_a_r;
	this->fullParamsPtr->c_r=parseResults_Mat.coating_c_r;
	this->fullParamsPtr->a_t=parseResults_Mat.coating_a_t;
	this->fullParamsPtr->c_t=parseResults_Mat.coating_c_t;
	return COAT_NO_ERROR;
}
