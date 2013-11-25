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

/**\file MaterialDiffracting.cpp
* \brief refracting material
* 
*           
* \author Mauch
*/

#include "MaterialDiffracting.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "rayTracingMath.h"


/**
 * \detail hit function of material for geometric rays
 *
 * we call the hit function of the coating first. Then we call the hit function of the material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialDiffracting::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	
//	extern Group oGroup;
	bool coat_reflected=false;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);

	if ( hitDiffracting(ray, hitParams, this->params, t_hit, geometryID, coat_reflected) )
	{
		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
			this->scatterPtr->hit(ray, hitParams);
		//if (ray.depth<MAX_DEPTH_CPU && ray.flux>MIN_FLUX_CPU)
		//{			
		//	oGroup.trace(ray);
		//}
		if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
			ray.running=false;//stop ray

	}
}


/**
 * \detail setParams 
 *
 * \param[in] MatDiffracting_params params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialDiffracting::setParams(MatDiffracting_params params)
{
	this->update=true;
	this->params=params;
}

/**
 * \detail calcRefrIndices 
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialDiffracting::calcRefrIndices(double lambda)
{
	double l_lambda=lambda*1e3; // within the glass catalog lambda is assumed to be in microns. In our trace we have lambda in mm. We calculate a local lambda here in microns so we can use the coefficients from the glass catalog without rescaling...
	// calc refractive index of glass material for current wavelength
	if ( (l_lambda<fullParamsPtr->lambdaMin)||(l_lambda>fullParamsPtr->lambdaMax) )
	{
		std::cout << "error in MaterialRefracting.calcRefrIndices(): lambda outside of range of glass definition at lambda=" << lambda << std::endl;
		return MAT_ERR;
	}
	switch (this->fullParamsPtr->dispersionFormula)
	{
		case MAT_DISPFORMULA_SCHOTT:
			this->params.n1=sqrt(fullParamsPtr->paramsNom[0]+fullParamsPtr->paramsNom[1]*pow(l_lambda,2)+fullParamsPtr->paramsNom[2]*pow(l_lambda,-2)+fullParamsPtr->paramsNom[3]*pow(l_lambda,-4)+fullParamsPtr->paramsNom[4]*pow(l_lambda,-6)+fullParamsPtr->paramsDenom[0]*pow(l_lambda,-8));
			break;
		case MAT_DISPFORMULA_SELLMEIER1:
			this->params.n1=sqrt(fullParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-fullParamsPtr->paramsDenom[0])+fullParamsPtr->paramsNom[1]*pow(l_lambda,2)/(pow(l_lambda,2)-fullParamsPtr->paramsDenom[1])+fullParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-fullParamsPtr->paramsDenom[2]));
			break;
		case MAT_DISPFORMULA_NODISP:
			// no dispersions comes with user defined glasses for which the correct values are already set for n1 and n2...
			this->params.n1=this->params.n1;
			break;
		default:
			this->params.n1=1;
			break;
	}	
	// do the same for the immersion medium
	if ( (l_lambda<immersionDispersionParamsPtr->lambdaMin)||(l_lambda>immersionDispersionParamsPtr->lambdaMax) )
	{
		std::cout << "error in MaterialRefracting.calcRefrIndices(): lambda outside of range of immersion definition at lambda=" << lambda << std::endl;
		return MAT_ERR;
	}
	switch (this->immersionDispersionParamsPtr->dispersionFormula)
	{
		case MAT_DISPFORMULA_SCHOTT:
			this->params.n2=sqrt(immersionDispersionParamsPtr->paramsNom[0]+immersionDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)+immersionDispersionParamsPtr->paramsNom[2]*pow(l_lambda,-2)+immersionDispersionParamsPtr->paramsNom[3]*pow(l_lambda,-4)+immersionDispersionParamsPtr->paramsNom[4]*pow(l_lambda,-6)+immersionDispersionParamsPtr->paramsDenom[0]*pow(l_lambda,-8));
			break;
		case MAT_DISPFORMULA_SELLMEIER1:
			this->params.n2=sqrt(immersionDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsDenom[0])+immersionDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsDenom[1])+immersionDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsDenom[2]));
			break;
		case MAT_DISPFORMULA_NODISP:
			// no dispersions comes with user defined glasses for which the correct values are already set for n1 and n2...
			this->params.n2=this->params.n2;
			break;
		default:
			this->params.n2=1;
			break;
	}	

	return MAT_NO_ERR;
};

/**
 * \detail createOptiXInstance
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks we have a seperate .cu file for each combination of material, scatter and coating. Therefore we set the path to that ptx file that corresponds to the combination present in the current instance
 * \author Mauch
 */
MaterialError MaterialDiffracting::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	lambda_old=lambda; // when creating the OptiXInstance we need to do this

	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialDiffracting.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR != calcRefrIndices(lambda))
	{
		std::cout << "error in MaterialDiffracting.createOptiXInstance(): calcRefrIndeices() returned an error" << std::endl;
		return MAT_ERR;
	}
	params.importanceAreaHalfWidth=this->fullParamsPtr->importanceAreaHalfWidth;
	params.importanceAreaRoot=this->fullParamsPtr->importanceAreaRoot;
	params.importanceAreaTilt=this->fullParamsPtr->importanceAreaTilt;

	/* set the variables of the geometry */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatDiffracting_params), &(this->params)), context) )
		return MAT_ERR;

	return MAT_NO_ERR;	
};

/**
 * \detail updateOptiXInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialDiffracting::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if ( (this->update)||(this->lambda_old!=lambda) )
	{
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "params", &l_params ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatDiffracting_params), &(this->params)), context) )
			return MAT_ERR;

		// calc the refractive indices at current wavelength
		if (MAT_NO_ERR != calcRefrIndices(lambda))
		{
			std::cout << "error in MaterialLinearGrating1D.updateOptiXInstance(): calcRefrIndeices() returned an error" << std::endl;
			return MAT_ERR;
		}
		// if we have no importance object and all cone angles are zero, we set importance cone to full hemisphere
		params.importanceAreaHalfWidth=this->fullParamsPtr->importanceAreaHalfWidth;
		params.importanceAreaRoot=this->fullParamsPtr->importanceAreaRoot;
		params.importanceAreaTilt=this->fullParamsPtr->importanceAreaTilt;

		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatDiffracting_params), &(this->params)), context) )
			return MAT_ERR;

		this->update=false;
	}

	if (MAT_NO_ERR != Material::updateOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialDiffracting.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}
	return MAT_NO_ERR;	
};

/**
 * \detail updateOptiXInstance 
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialDiffracting::updateOptiXInstance(double lambda)
{
	if ( (this->update)||(this->lambda_old!=lambda) )
	{
		// calc the refractive indices at current wavelength
		if (MAT_NO_ERR != calcRefrIndices(lambda))
		{
			std::cout << "error in MaterialLinearGrating1D.updateOptiXInstance(): calcRefrIndeices() returned an error" << std::endl;
			return MAT_ERR;
		}
		params.importanceAreaHalfWidth=this->fullParamsPtr->importanceAreaHalfWidth;
		params.importanceAreaRoot=this->fullParamsPtr->importanceAreaRoot;
		params.importanceAreaTilt=this->fullParamsPtr->importanceAreaTilt;
		this->update=false;
	}
	if ( (this->coatingPtr->update)||(this->lambda_old!=lambda) )
	{
		lambda_old=lambda;
		// create simulation instance of coating
		this->coatingPtr->createCPUSimInstance(lambda);

		this->coatingPtr->update=false;
	}
	if ( (this->scatterPtr->update)||(this->lambda_old!=lambda) )
	{
		lambda_old=lambda;
		// create simulation instance of scatter
		this->scatterPtr->createCPUSimInstance(lambda);

		this->scatterPtr->update=false;
	}

	return MAT_NO_ERR;	
};


/**
 * \detail getParams 
 *
 * \param[in] void
 * 
 * \return MatDiffractingParams
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatDiffracting_params MaterialDiffracting::getParams(void)
{
	return this->params;
}

/**
 * \detail setDispersionParams 
 *
 * \param[in] MatDiffracting_DispersionParams* paramsInPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialDiffracting::setDispersionParams(MatDiffracting_DispersionParams* paramsInPtr)
{
	this->fullParamsPtr=paramsInPtr;
}

/**
 * \detail getGlassDispersionParams 
 *
 * \param[in] void
 * 
 * \return MatDiffracting_DispersionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatDiffracting_DispersionParams* MaterialDiffracting::getFullParams(void)
{
	return this->fullParamsPtr;
}

/**
 * \detail getImmersionDispersionParams 
 *
 * \param[in] void
 * 
 * \return MatDiffracting_DispersionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatDiffracting_DispersionParams* MaterialDiffracting::getImmersionDispersionParams(void)
{
	return this->immersionDispersionParamsPtr;
}

/**
 * \detail setImmersionDispersionParams 
 *
 * \param[in] MatDiffracting_DispersionParams*
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialDiffracting::setImmersionDispersionParams(MatDiffracting_DispersionParams* paramsInPtr)
{
	this->immersionDispersionParamsPtr=paramsInPtr;
}

/**
 * \detail createCPUSimInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialDiffracting::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR != calcRefrIndices(lambda))
	{
		std::cout << "error in MaterialLinearGrating1D.createOptiXInstance(): calcRefrIndeices() returned an error" << std::endl;
		return MAT_ERR;
	}
	params.importanceAreaHalfWidth=this->fullParamsPtr->importanceAreaHalfWidth;
	params.importanceAreaRoot=this->fullParamsPtr->importanceAreaRoot;
	params.importanceAreaTilt=this->fullParamsPtr->importanceAreaTilt;

	// create simulation instance of coating
	if (MAT_NO_ERR != Material::createCPUSimInstance(lambda) )
	{
		std::cout << "error in MaterialDiffracting.createCPUSimInstance(): Material.createCPUSimInstance() returned an error." << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] MaterialParseParamStruct &parseResults_Geom, parseGlassResultStruct &parseResults_Glass
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialDiffracting::processParseResults(MaterialParseParamStruct &parseResults_Mat, parseGlassResultStruct* parseResults_GlassPtr)
{
	this->fullParamsPtr=new MatDiffracting_DispersionParams;
	this->immersionDispersionParamsPtr=new MatDiffracting_DispersionParams;
	// set 
	this->fullParamsPtr->lambdaMax=DOUBLE_MAX;
	this->fullParamsPtr->lambdaMin=0;
	this->fullParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
	this->params.n1=1;
	
	this->fullParamsPtr->importanceAreaHalfWidth=parseResults_Mat.importanceAreaHalfWidth;
	this->fullParamsPtr->importanceAreaRoot=parseResults_Mat.importanceAreaRoot;
	this->fullParamsPtr->importanceAreaTilt=parseResults_Mat.importanceAreaTilt;
	// if  we have a user defined immersion medium set it
	if (!strcmp(parseResults_Mat.immersionName,"USERDEFINED"))
	{
		//refrParams.n2=parseResults_Mat.nRefr.y;
		immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
		immersionDispersionParamsPtr->lambdaMin=0;
		//oMaterialDiffractingPtr->setParams(refrParams);
		immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
	}
	else
	{
		// if we have no immersion medium specified, set it to n=1
		if (!strcmp(parseResults_Mat.immersionName,"STANDARD"))
		{
			this->params.n2=1;
			immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
			immersionDispersionParamsPtr->lambdaMin=0;
			immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
		}
		else // if we have a immersion medium specified, we use the parse results
		{
			immersionDispersionParamsPtr->lambdaMax=parseResults_GlassPtr->lambdaMax;
			immersionDispersionParamsPtr->lambdaMin=parseResults_GlassPtr->lambdaMin;
			memcpy(immersionDispersionParamsPtr->paramsDenom, parseResults_GlassPtr->paramsDenom, 5*sizeof(double));
			memcpy(immersionDispersionParamsPtr->paramsNom, parseResults_GlassPtr->paramsNom, 5*sizeof(double));
			switch (parseResults_GlassPtr->dispersionFormulaIndex)
			{
				case 1:
					immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
					break;
				case 2:
					immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
					break;
				default:
					immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
					std::cout <<"error in MaterialDiffracting.processParseResults(): unknown dispersion formula" << std::endl;
					return MAT_ERR;
					break;
			}
		} // end parsing for immersion medium
	} // end of "if immersion medium is user defined"

	return MAT_NO_ERR;
}
