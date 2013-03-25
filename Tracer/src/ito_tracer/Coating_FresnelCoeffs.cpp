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

/**\file Coating_FresnelCoeffs.cpp
* \brief Coating with predefined coefficients for transmissione and reflection
* 
*           
* \author Mauch
*/

#include "Coating_FresnelCoeffs.h"
#include "myUtil.h"
#include <iostream>
#include "sampleConfig.h"
#include <string.h>

/**
 * \detail calcRefrIndices 
 *
 * \param[in] double lambda
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_FresnelCoeffs::calcRefrIndices(double lambda)
{
	double l_lambda=lambda*1e3; // within the glass catalog lambda is assumed to be in microns. In our trace we have lambda in mm. We calculate a local lambda here in microns so we can use the coefficients from the glass catalog without rescaling...
	// calc refractive index of glass material for current wavelength
	if ( (l_lambda<this->fullParamsPtr->glassDispersionParamsPtr->lambdaMin)||(l_lambda>this->fullParamsPtr->glassDispersionParamsPtr->lambdaMax) )
	{
		std::cout << "error in Coating_FresnelCoeffs.calcRefrIndices(): lambda outside of range of glass definition at lambda=" << lambda << std::endl;
		return COAT_ERROR;
	}
	switch (this->fullParamsPtr->glassDispersionParamsPtr->dispersionFormula)
	{
		case MAT_DISPFORMULA_SCHOTT:
			this->reducedParamsPtr->n1=sqrt(this->fullParamsPtr->glassDispersionParamsPtr->paramsNom[0]+this->fullParamsPtr->glassDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)+this->fullParamsPtr->glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,-2)+this->fullParamsPtr->glassDispersionParamsPtr->paramsNom[3]*pow(l_lambda,-4)+this->fullParamsPtr->glassDispersionParamsPtr->paramsNom[4]*pow(l_lambda,-6)+this->fullParamsPtr->glassDispersionParamsPtr->paramsDenom[0]*pow(l_lambda,-8));
			break;
		case MAT_DISPFORMULA_SELLMEIER1:
			this->reducedParamsPtr->n1=sqrt(this->fullParamsPtr->glassDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-this->fullParamsPtr->glassDispersionParamsPtr->paramsDenom[0])+this->fullParamsPtr->glassDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)/(pow(l_lambda,2)-this->fullParamsPtr->glassDispersionParamsPtr->paramsDenom[1])+this->fullParamsPtr->glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-this->fullParamsPtr->glassDispersionParamsPtr->paramsDenom[2]));
			break;
		case MAT_DISPFORMULA_NODISP:
			// no dispersions comes with user defined glasses for which the correct values are already set for n1 and n2...
			this->reducedParamsPtr->n1=this->fullParamsPtr->n1;
			break;
		default:
			this->reducedParamsPtr->n1=1;
			break;
	}	
	// do the same for the immersion medium
	if ( (l_lambda<this->fullParamsPtr->immersionDispersionParamsPtr->lambdaMin)||(l_lambda>this->fullParamsPtr->immersionDispersionParamsPtr->lambdaMax) )
	{
		std::cout << "error in Coating_FresnelCoeffs.calcRefrIndices(): lambda outside of range of immersion definition at lambda=" << lambda << std::endl;
		return COAT_ERROR;
	}
	switch (this->fullParamsPtr->immersionDispersionParamsPtr->dispersionFormula)
	{
		case MAT_DISPFORMULA_SCHOTT:
			this->reducedParamsPtr->n2=sqrt(this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom[0]+this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)+this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom[2]*pow(l_lambda,-2)+this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom[3]*pow(l_lambda,-4)+this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom[4]*pow(l_lambda,-6)+this->fullParamsPtr->immersionDispersionParamsPtr->paramsDenom[0]*pow(l_lambda,-8));
			break;
		case MAT_DISPFORMULA_SELLMEIER1:
			this->reducedParamsPtr->n2=sqrt(this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-this->fullParamsPtr->immersionDispersionParamsPtr->paramsDenom[0])+this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)/(pow(l_lambda,2)-this->fullParamsPtr->immersionDispersionParamsPtr->paramsDenom[1])+this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-this->fullParamsPtr->immersionDispersionParamsPtr->paramsDenom[2]));
			break;
		case MAT_DISPFORMULA_NODISP:
			// no dispersions comes with user defined glasses for which the correct values are already set for n1 and n2...
			this->reducedParamsPtr->n2=this->fullParamsPtr->n2;
			break;
		default:
			this->reducedParamsPtr->n2=1;
			break;
	}	

	return COAT_NO_ERROR;
};

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
CoatingError Coating_FresnelCoeffs::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	if (COAT_NO_ERROR!=this->calcRefrIndices(lambda))
	{
		std::cout << "error in Coating_FresnelCoeffs.createCPUSimInstance(): calcRefrIndices() returned an error."<< std::endl;
		return COAT_ERROR;
	}
	this->update=false;
	return COAT_NO_ERROR;	
};

CoatingError Coating_FresnelCoeffs::setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_coatingParamsPtr)
{
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( *closest_hit_programPtr, "coating_params", l_coatingParamsPtr ), context) )
		return COAT_ERROR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(*l_coatingParamsPtr, sizeof(Coating_FresnelCoeffs_ReducedParams), (this->reducedParamsPtr)), context) )
		return COAT_ERROR;

	return COAT_NO_ERROR;
};

/**
 * \detail setFullParams
 *
 * \param[in] Coating_FresnelCoeffs_ReducedParams* ptrIn
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_FresnelCoeffs::setFullParams(Coating_FresnelCoeffs_FullParams* ptrIn)
{
	this->fullParamsPtr=ptrIn;
	return COAT_NO_ERROR;
};

/**
 * \detail getFullParams
 *
 * \param[in] void
 * 
 * \return Coating_FresnelCoeffs_ReducedParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_FresnelCoeffs_FullParams* Coating_FresnelCoeffs::getFullParams(void)
{
	return this->fullParamsPtr;
};

/**
 * \detail getReducedParams
 *
 * \param[in] void
 * 
 * \return Coating_FresnelCoeffs_ReducedParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_FresnelCoeffs_ReducedParams* Coating_FresnelCoeffs::getReducedParams(void)
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
//CoatingError Coating_FresnelCoeffs::calcCoatingCoeffs(double lambda, double3 normal, double3 direction)
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
bool  Coating_FresnelCoeffs::hit(rayStruct &ray, Mat_hitParams hitParams)
{
	return hitCoatingFresnelCoeff(ray, hitParams, *this->reducedParamsPtr);
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
bool Coating_FresnelCoeffs::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
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
CoatingError Coating_FresnelCoeffs::processParseResults(MaterialParseParamStruct &parseResults_Mat, parseGlassResultStruct* parseResults_GlassPtr, parseGlassResultStruct* parseResults_ImmPtr)
{
	this->fullParamsPtr->glassDispersionParamsPtr=new MatRefracting_DispersionParams;
	this->fullParamsPtr->immersionDispersionParamsPtr=new MatRefracting_DispersionParams;
	// if we have a user defined glass we simply take the values of n1 and n2 defined in the prescription file and set no dispersion
	if (!strcmp(parseResults_Mat.glassName, "USERDEFINED"))
	{
		this->fullParamsPtr->n1=parseResults_Mat.nRefr.x;
		this->fullParamsPtr->glassDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
		this->fullParamsPtr->glassDispersionParamsPtr->lambdaMin=0;
		this->fullParamsPtr->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
	}
	else
	{
		this->fullParamsPtr->glassDispersionParamsPtr->lambdaMax=parseResults_GlassPtr->lambdaMax;
		this->fullParamsPtr->glassDispersionParamsPtr->lambdaMin=parseResults_GlassPtr->lambdaMin;
		memcpy(this->fullParamsPtr->glassDispersionParamsPtr->paramsDenom, parseResults_GlassPtr->paramsDenom, 5*sizeof(double));
		memcpy(this->fullParamsPtr->glassDispersionParamsPtr->paramsNom, parseResults_GlassPtr->paramsNom, 5*sizeof(double));
		switch (parseResults_GlassPtr->dispersionFormulaIndex)
		{
			case 1:
				this->fullParamsPtr->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
				break;
			case 2:
				this->fullParamsPtr->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
				break;
			default:
				this->fullParamsPtr->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
				std::cout <<"error in Coating_FresnelCoeffs.processParseResults(): unknown material dispersion formula" << std::endl;
				return COAT_ERROR;
				break;
		}
	}
	// if  we have a user defined immersion medium set it
	if (!strcmp(parseResults_Mat.immersionName,"USERDEFINED"))
	{
		this->fullParamsPtr->n2=parseResults_Mat.nRefr.y;
		this->fullParamsPtr->immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
		this->fullParamsPtr->immersionDispersionParamsPtr->lambdaMin=0;
		this->fullParamsPtr->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
	}
	else
	{
		// if we have no immersion medium specified, set it to n=1
		if (!strcmp(parseResults_Mat.immersionName,"STANDARD"))
		{
			this->fullParamsPtr->n2=1;
			this->fullParamsPtr->immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
			this->fullParamsPtr->immersionDispersionParamsPtr->lambdaMin=0;
			this->fullParamsPtr->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
		}
		else // if we have a immersion medium specified, parse for it in the glass catalog
		{
			this->fullParamsPtr->immersionDispersionParamsPtr->lambdaMax=parseResults_ImmPtr->lambdaMax;
			this->fullParamsPtr->immersionDispersionParamsPtr->lambdaMin=parseResults_ImmPtr->lambdaMin;
			memcpy(this->fullParamsPtr->immersionDispersionParamsPtr->paramsDenom, parseResults_ImmPtr->paramsDenom, 5*sizeof(double));
			memcpy(this->fullParamsPtr->immersionDispersionParamsPtr->paramsNom, parseResults_ImmPtr->paramsNom, 5*sizeof(double));
			switch (parseResults_ImmPtr->dispersionFormulaIndex)
			{
				case 1:
					this->fullParamsPtr->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
					break;
				case 2:
					this->fullParamsPtr->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
					break;
				default:
					this->fullParamsPtr->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
					std::cout <<"error in Coating_FresnelCoeffs.processParseResults(): unknown material dispersion formula" << std::endl;
					return COAT_ERROR;
					break;
			}
		} // end parsing for immersion medium
	} // end of "if immersion medium is user defined"
	this->fullParamsPtr->type=CT_FRESNELCOEFFS;
	return COAT_NO_ERROR;
}
