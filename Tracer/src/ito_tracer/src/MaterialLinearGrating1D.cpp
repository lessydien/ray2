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

/**\file MaterialLinearGrating1D.cpp
* \brief material of a linear line grating
* 
*           
* \author Mauch
*/

#include "MaterialLinearGrating1D.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>

#include "Parser_XML.h"
#include "Parser.h"


/**
 * \detail getGlassDispersionParams 
 *
 * \param[in] void
 * 
 * \return MatRefracting_DispersionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatRefracting_DispersionParams* MaterialLinearGrating1D::getGlassDispersionParams(void)
{
	return this->glassDispersionParamsPtr;
};

/**
 * \detail setGlassDispersionParams 
 *
 * \param[in] MatRefracting_DispersionParams* dispersionParamsInPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialLinearGrating1D::setGlassDispersionParams(MatRefracting_DispersionParams* dispersionParamsInPtr)
{
	this->glassDispersionParamsPtr=dispersionParamsInPtr;
};

/**
 * \detail getImmersionDispersionParams 
 *
 * \param[in] void
 * 
 * \return MatRefracting_DispersionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatRefracting_DispersionParams* MaterialLinearGrating1D::getImmersionDispersionParams(void)
{
	return this->immersionDispersionParamsPtr;
};

/**
 * \detail setImmersionDispersionParams 
 *
 * \param[in] MatRefracting_DispersionParams* dispersionParamsInPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialLinearGrating1D::setImmersionDispersionParams(MatRefracting_DispersionParams* dispersionParamsInPtr)
{
	this->immersionDispersionParamsPtr=dispersionParamsInPtr;
};

/**
 * \detail getDiffractionParams 
 *
 * \param[in] void
 * 
 * \return MatLinearGrating1D_DiffractionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatLinearGrating1D_DiffractionParams* MaterialLinearGrating1D::getDiffractionParams(void)
{
	return this->diffractionParamsPtr;
};

/**
 * \detail setDiffractionParams 
 *
 * \param[in] MatLinearGrating1D_DiffractionParams*
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialLinearGrating1D::setDiffractionParams(MatLinearGrating1D_DiffractionParams* diffractionParamsInPtr)
{
	this->diffractionParamsPtr=diffractionParamsInPtr;	
};

/**
 * \detail hit function of material for geometric rays
 *
 * Here we need to call the hit function of the coating first as the grating diffracts different for reflection and for transmission. Then we call hitLinearGrating1D that describes the interaction of the ray with the material and can be called from GPU as well. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialLinearGrating1D::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	bool reflected=false; // init flag indicating reflection
	if (this->params.nRefr1==0)
		reflected =true; // check wether we had a reflective grating or not
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		reflected=this->coatingPtr->hit(ray, hitParams); // now see wether the coating wants reflection or refraction

	if (hitLinearGrating1D(ray, hitParams, this->params, t_hit, geometryID, reflected) )
	{
		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
			this->scatterPtr->hit(ray, hitParams);

//		if (ray.depth<MAX_DEPTH_CPU && ray.flux>MIN_FLUX_CPU)
//		{			
//			oGroup.trace(ray);
//		}
		if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
			ray.running=false;//stop ray
	}
};

/**
 * \detail hit function of the material for gaussian beam rays
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID
 * 
 * \return void
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
void MaterialLinearGrating1D::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID)
{
		extern Group oGroup;
		// reflect all the rays making up the gaussian beam
		ray.baseRay.direction=reflect(ray.baseRay.direction,normal.normal_baseRay);
		ray.waistRayX.direction=reflect(ray.waistRayX.direction,normal.normal_waistRayX);
		ray.waistRayY.direction=reflect(ray.waistRayY.direction,normal.normal_waistRayY);
		ray.divRayX.direction=reflect(ray.divRayX.direction,normal.normal_divRayX);
		ray.divRayY.direction=reflect(ray.divRayY.direction,normal.normal_divRayY);
		ray.baseRay.currentGeometryID=geometryID;
		if (ray.baseRay.depth<MAX_DEPTH_CPU && ray.baseRay.flux>MIN_FLUX_CPU)
		{			
			oGroup.trace(ray);
		}
};

/**
 * \detail calcRefrIndices
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks we have a seperate .cu file for each combination of material, scatter and coating. Therefore we set the path to that ptx file that corresponds to the combination present in the current instance
 * \author Mauch
 */
MaterialError MaterialLinearGrating1D::calcRefrIndices(double lambda)
{
	double l_lambda=lambda*1e3; // within the glass catalog lambda is assumed to be in microns. In our trace we have lambda in mm. We calculate a local lambda here in microns so we can use the coefficients from the glass catalog without rescaling...
	// calc refractive index of glass material for current wavelength
	if ( (l_lambda<glassDispersionParamsPtr->lambdaMin)||(l_lambda>glassDispersionParamsPtr->lambdaMax) )
	{
		std::cout << "error in MaterialLinearGrating1D.calcRefrIndices(): lambda outside definition range of glass at" << lambda << std::endl;
		return MAT_ERR;
	}
	switch (this->glassDispersionParamsPtr->dispersionFormula)
	{
		case MAT_DISPFORMULA_SCHOTT:
			this->params.nRefr1 =sqrt(glassDispersionParamsPtr->paramsNom[0]+glassDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)+glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,-2)+glassDispersionParamsPtr->paramsNom[3]*pow(l_lambda,-4)+glassDispersionParamsPtr->paramsNom[4]*pow(l_lambda,-6)+glassDispersionParamsPtr->paramsDenom[0]*pow(l_lambda,-8));
			break;
		case MAT_DISPFORMULA_SELLMEIER1:
			this->params.nRefr1=sqrt(glassDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[0])+glassDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[1])+glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[2]));
			break;
		case MAT_DISPFORMULA_NODISP:
			// no dispersions comes with user defined glasses for which the correct values are already set for n1 and n2...
			this->params.nRefr1=this->params.nRefr1;
			break;
		default:
			this->params.nRefr1=1;
			break;
	}	
	// do the same for the immersion medium
	if ( (l_lambda<immersionDispersionParamsPtr->lambdaMin)||(l_lambda>immersionDispersionParamsPtr->lambdaMax) )
	{
		std::cout << "error in MaterialLinearGrating1D.calcRefrIndices(): lambda outside definition range of immersion at" << lambda << std::endl;
		return MAT_ERR;
	}
	switch (this->immersionDispersionParamsPtr->dispersionFormula)
	{
		case MAT_DISPFORMULA_SCHOTT:
			this->params.nRefr2=sqrt(immersionDispersionParamsPtr->paramsNom[0]+glassDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)+glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,-2)+glassDispersionParamsPtr->paramsNom[3]*pow(l_lambda,-4)+glassDispersionParamsPtr->paramsNom[4]*pow(l_lambda,-6)+glassDispersionParamsPtr->paramsDenom[0]*pow(l_lambda,-8));
			break;
		case MAT_DISPFORMULA_SELLMEIER1:
			this->params.nRefr2=sqrt(immersionDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[0])+glassDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[1])+glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[2]));
			break;
		case MAT_DISPFORMULA_NODISP:
			// no dispersions comes with user defined glasses for which the correct values are already set for n1 and n2...
			this->params.nRefr2=this->params.nRefr2;
			break;
		default:
			this->params.nRefr2=1;
			break;
	}	

	return MAT_NO_ERR;
};

/**
 * \detail calcDiffEffs
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks we have a seperate .cu file for each combination of material, scatter and coating. Therefore we set the path to that ptx file that corresponds to the combination present in the current instance
 * \author Mauch
 */
MaterialError MaterialLinearGrating1D::calcDiffEffs(double lambda)
{
	this->params.g=this->diffractionParamsPtr->g;
	this->params.nrDiffOrders=this->diffractionParamsPtr->nrOrdersSim;
	this->params.diffAxis=this->diffractionParamsPtr->diffAxis;
	memcpy(this->params.diffOrderNr, this->diffractionParamsPtr->diffOrdersPtr, this->params.nrDiffOrders*sizeof(short));
	// find the indices of the wavelengths that are next to the current simulation wavelength
	int i;
	int indexLambdaPlus=this->diffractionParamsPtr->nrWavelengths-1;
	int indexLambdaMin=this->diffractionParamsPtr->nrWavelengths-1;
	for (i=0;i<this->diffractionParamsPtr->nrWavelengths;i++)
	{
		if (this->diffractionParamsPtr->lambdaPtr[i]>lambda)
		{
			indexLambdaPlus=min(i,this->diffractionParamsPtr->nrWavelengths-1);
			indexLambdaMin=max(0,i-1);
			break;
		}
	}
	// interpolate efficiencies for given wavelength
	// do we need to copy all four efficiencies to the hit-params ?!?
	double avgEffInterpol;
	double avgEffPlus;
	double avgEffMin;
	int indexOrder;
	for (i=0;i<this->params.nrDiffOrders;i++)
	{
		// find the index for the efficiency of the current diffraction order
		indexOrder=(int)((this->diffractionParamsPtr->nrOrders+1)/2-1+this->params.diffOrderNr[i])*this->diffractionParamsPtr->nrWavelengths;
		// copy the efficiency of the order specified in params.diffOrderNr. Remember that RTP01 is a 2-dimensional field in wavelnegth and diffraction order. The indexing of RTP01Ptr is wavelength first !!
		// for unpolarized ray tracing we calculate an average efficiency
		avgEffPlus=(this->diffractionParamsPtr->RTP01Ptr[indexOrder+indexLambdaPlus]+this->diffractionParamsPtr->RTP10Ptr[indexOrder+indexLambdaPlus]+this->diffractionParamsPtr->RTS01Ptr[indexOrder+indexLambdaPlus]+this->diffractionParamsPtr->RTS10Ptr[indexOrder+indexLambdaPlus])/2;
		// if the current wavelength is outside the given range, we take the value of the last efficiency in range
		if (indexLambdaPlus==indexLambdaMin)
			this->params.eff[i]=avgEffPlus;
		else
		{
			// do the interpolation
			double test=this->diffractionParamsPtr->RTP10Ptr[indexOrder+indexLambdaMin];
			avgEffMin=(this->diffractionParamsPtr->RTP01Ptr[indexOrder+indexLambdaMin]+this->diffractionParamsPtr->RTP10Ptr[indexOrder+indexLambdaMin]+this->diffractionParamsPtr->RTS01Ptr[indexOrder+indexLambdaMin]+this->diffractionParamsPtr->RTS10Ptr[indexOrder+indexLambdaMin])/2;
			avgEffInterpol=avgEffMin+(avgEffPlus-avgEffMin)/(this->diffractionParamsPtr->lambdaPtr[indexLambdaPlus]-this->diffractionParamsPtr->lambdaPtr[indexLambdaMin])*(lambda-this->diffractionParamsPtr->lambdaPtr[indexLambdaMin]);
			this->params.eff[i]=avgEffInterpol;
		}
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
MaterialError MaterialLinearGrating1D::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR != calcRefrIndices(lambda))
	{
		std::cout << "error in MaterialLinearGrating1D.createOptiXInstance(): calcRefrIndices() returned an error" << std::endl;
		return MAT_ERR;
	}
	// calc the diffraction efficiencies at the current wavelength
	if (MAT_NO_ERR != calcDiffEffs(lambda))
	{
		std::cout << "error in MaterialLinearGrating1D.createOptiXInstance(): calcDiffEffs() returned an error" << std::endl;
		return MAT_ERR;
	}

	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialLinearGrating1D.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	lambda_old=lambda; // when creating the instance we need to do this

	// set material params
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatLinearGrating1D_params), &(this->params)), context) )
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
MaterialError MaterialLinearGrating1D::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if ( (this->update)||(this->lambda_old!=lambda) )
	{
		// calc the refractive indices at current wavelength
		if (MAT_NO_ERR != calcRefrIndices(lambda))
		{
			std::cout << "error in MaterialLinearGrating1D.updateOptiXInstance(): calcRefrIndices() returned an error" << std::endl;
			return MAT_ERR;
		}
		// calc the diffraction efficiencies at the current wavelength
		if (MAT_NO_ERR != calcDiffEffs(lambda))
		{
			std::cout << "error in MaterialLinearGrating1D.createOptiXInstance(): calcDiffEffs() returned an error" << std::endl;
			return MAT_ERR;
		}
		this->update=false;
	}

	if (MAT_NO_ERR != Material::updateOptiXInstance(context, instance, index, simParams, lambda) )
	{
		std::cout << "error in MaterialLinearGrating1D.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;	
};

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
MaterialError MaterialLinearGrating1D::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR != calcRefrIndices(lambda))
	{
		std::cout << "error in MaterialLinearGrating1D.createCPUSimInstance(): calcRefrIndices() returned an error" << std::endl;
		return MAT_ERR;
	}

	if (MAT_NO_ERR != calcDiffEffs(lambda))
	{
		std::cout << "error in MaterialLinearGrating1D.createCPUSimInstance(): calcDiffEffs() returned an error" << std::endl;
		return MAT_ERR;
	}

	// create simulation instance of coating
	if (MAT_NO_ERR != Material::createCPUSimInstance(lambda) )
	{
		std::cout << "error in MaterialLinearGrating1D.createCPUSimInstance(): Material.createCPUSimInstance() returned an error." << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;
};

/**
 * \detail setParams 
 *
 * \param[in] MatLinearGrating1D_params paramsIn
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialLinearGrating1D::setParams(MatLinearGrating1D_params paramsIn)
{
	this->update=true;
	this->params=paramsIn;
};

/**
 * \detail setParams 
 *
 * \param[in] void
 * 
 * \return MaterialLinearGrating1D
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatLinearGrating1D_params MaterialLinearGrating1D::getParams(void)
{
	return this->params;
};

/**
 * \detail updateCPUSimInstance 
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialLinearGrating1D::updateCPUSimInstance(double lambda)
{
	if ( (this->update)||(this->lambda_old!=lambda) )
	{
		lambda_old=lambda;
		// calc the refractive indices at current wavelength
		if (MAT_NO_ERR != calcRefrIndices(lambda))
		{
			std::cout << "error in MaterialLinearGrating1D.updateCPUSimInstance(): calcRefrIndices() returned an error" << std::endl;
			return MAT_ERR;
		}
		// calc the diffraction efficiencies at the current wavelength
		if (MAT_NO_ERR != calcDiffEffs(lambda))
		{
			std::cout << "error in MaterialLinearGrating1D.updateCPUSimInstance(): calcDiffEffs() returned an error" << std::endl;
			return MAT_ERR;
		}
		this->update=false;
	}
	if ( (this->scatterPtr->update)||(this->lambda_old!=lambda) )
	{
		if (SCAT_NO_ERROR != this->scatterPtr->createCPUSimInstance(lambda) )
		{
			std::cout << "error in MaterialLinearGrating1D.updateCPUSimInstance(): Scatter.createCPUSimInstance() returned an error" << std::endl;
			return MAT_ERR;
		}
	}
	if ( (this->coatingPtr->update)||(this->lambda_old!=lambda) )
	{
		// create simulation instance of coating
		if (COAT_NO_ERROR != this->coatingPtr->createCPUSimInstance(lambda) )
		{
			std::cout << "error in MaterialLinearGrating1D.updateCPUSimInstance(): Coating.createCPUSimInstance() returned an error" << std::endl;
			return MAT_ERR;
		}
		this->coatingPtr->update=false;
	}
	return MAT_NO_ERR;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] MaterialParseParamStruct &parseResults_Mat
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialLinearGrating1D::processParseResults(MaterialParseParamStruct &parseResults_Mat, parseGlassResultStruct* parseResults_GlassPtr, parseGlassResultStruct* parseResults_ImmPtr, ParseGratingResultStruct* parseResults_GratPtr)
{
	this->glassDispersionParamsPtr=new MatRefracting_DispersionParams;
	this->immersionDispersionParamsPtr=new MatRefracting_DispersionParams;
	this->diffractionParamsPtr=new MatLinearGrating1D_DiffractionParams;
	if ( !strcmp(parseResults_Mat.glassName,"MIRROR") )
	{
		//a mirror is mimicked with a refracting material with refractive index 0. So we fill the dispersion parameters with values that produce n=0 for all lambdas
		this->glassDispersionParamsPtr->lambdaMax=1000;
		this->glassDispersionParamsPtr->lambdaMin=0;
		this->glassDispersionParamsPtr->paramsNom[0]=0;
		this->glassDispersionParamsPtr->paramsNom[1]=0;
		this->glassDispersionParamsPtr->paramsNom[2]=0;
		this->glassDispersionParamsPtr->paramsNom[3]=0;
		this->glassDispersionParamsPtr->paramsNom[4]=0;
		this->glassDispersionParamsPtr->paramsDenom[0]=0;
		this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
	}
	else
	{
		// if we have a user defined glass we simply take the values of n1 and n2 defined in the prescription file and set no dispersion
		if (!strcmp(parseResults_Mat.glassName, "USERDEFINED"))
		{
			params.nRefr1=parseResults_Mat.nRefr.x;
			//params.n2=parseResults_Mat.nRefr.y;
			this->glassDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
			this->glassDispersionParamsPtr->lambdaMin=0;
			this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
		}
		else
		{
			this->glassDispersionParamsPtr->lambdaMax=parseResults_GlassPtr->lambdaMax;
			this->glassDispersionParamsPtr->lambdaMin=parseResults_GlassPtr->lambdaMin;
			memcpy(this->glassDispersionParamsPtr->paramsDenom, parseResults_GlassPtr->paramsDenom, 5*sizeof(double));
			memcpy(this->glassDispersionParamsPtr->paramsNom, parseResults_GlassPtr->paramsNom, 5*sizeof(double));
			switch (parseResults_GlassPtr->dispersionFormulaIndex)
			{
				case 1:
					this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
					break;
				case 2:
					glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
					break;
				default:
					std::cout <<"error in MaterialLinearGrating1D.processParseResults(): unknown dispersion formula" << std::endl;
					glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
					return MAT_ERR;
					break;
			}
		}
	}
	// if  we have a user defined immersion medium set it
	if (!strcmp(parseResults_Mat.immersionName,"USERDEFINED"))
	{
		params.nRefr2=parseResults_Mat.nRefr.y;
		this->immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
		this->immersionDispersionParamsPtr->lambdaMin=0;
		this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
	}
	else
	{
		// if we have no immersion medium specified, set it to n=1
		if (!strcmp(parseResults_Mat.immersionName,"STANDARD"))
		{
			params.nRefr2=1;
			this->immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
			this->immersionDispersionParamsPtr->lambdaMin=0;
			this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
		}
		else // if we have a immersion medium specified, parse for it in the glass catalog
		{
			this->immersionDispersionParamsPtr->lambdaMax=parseResults_ImmPtr->lambdaMax;
			this->immersionDispersionParamsPtr->lambdaMin=parseResults_ImmPtr->lambdaMin;
			memcpy(this->immersionDispersionParamsPtr->paramsDenom, parseResults_ImmPtr->paramsDenom, 5*sizeof(double));
			memcpy(this->immersionDispersionParamsPtr->paramsNom, parseResults_ImmPtr->paramsNom, 5*sizeof(double));
			switch (parseResults_ImmPtr->dispersionFormulaIndex)
			{
				case 1:
					this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
					break;
				case 2:
					this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
					break;
				default:
					std::cout <<"error in MaterialLinearGrating1D.processParseResults(): unknown material dispersion formula" << std::endl;
					this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
					break;
			}
		} // end parsing for immersion medium
	} // end of "if immersion medium is user defined"

	// diffraction params
	this->diffractionParamsPtr->diffAxis=parseResults_Mat.coatingAxis;
	if ( parseResults_Mat.gratingOrdersFromFile || parseResults_Mat.gratingEffsFromFile )
	{
		// read number of diffraction orders from MicroSim file
		this->diffractionParamsPtr->nrOrders=parseResults_GratPtr->nrOrders;
	}
	else
		this->diffractionParamsPtr->nrOrders=parseResults_Mat.nrDiffOrders;
	if ( parseResults_Mat.gratingLinesFromFile )
		// read grating constant from MicroSim file
		this->diffractionParamsPtr->g=parseResults_GratPtr->g;
	else
		// read grating constant from zemax
		this->diffractionParamsPtr->g=parseResults_Mat.gratingConstant;
	if ( parseResults_Mat.gratingOrdersFromFile )
	{
		// read diffraction orders from MicroSim file
		this->diffractionParamsPtr->nrOrdersSim=parseResults_GratPtr->nrOrders;
		this->diffractionParamsPtr->diffOrdersPtr=(short*) calloc(parseResults_GratPtr->nrOrders, sizeof(short));
		memcpy(this->diffractionParamsPtr->diffOrdersPtr, parseResults_GratPtr->diffOrdersPtr, parseResults_GratPtr->nrOrders*sizeof(short));
	}
	else
	{
		// if the efficiencies are read from Zemax as well, we need to set the number of diffOrders to (2*abs(maximumOrder)+1)
		if ( !parseResults_Mat.gratingEffsFromFile )
		{
			short maxOrder=0;
			// find highest diffraction order that is to be simulated
			for (int j=0;j<parseResults_Mat.nrDiffOrders;j++)
			{
				if (maxOrder<abs(parseResults_Mat.diffOrder[j]))
					maxOrder=parseResults_Mat.diffOrder[j];
			}
			this->diffractionParamsPtr->nrOrders=2*maxOrder+1;
		}
		// read diffraction orders from zemax
		this->diffractionParamsPtr->nrOrdersSim=parseResults_Mat.nrDiffOrders;
		this->diffractionParamsPtr->diffOrdersPtr=(short*) calloc(parseResults_Mat.nrDiffOrders, sizeof(short));
		memcpy(this->diffractionParamsPtr->diffOrdersPtr, &(parseResults_Mat.diffOrder[0]), this->diffractionParamsPtr->nrOrdersSim*sizeof(short));

		// if we have the efficiencies from file, we copy all of them. Later only the ones that are to be simulated are sent to GPU
		if ( parseResults_Mat.gratingEffsFromFile )
		{
			// allocate memory for the efficiencies
			this->diffractionParamsPtr->nrWavelengths=parseResults_GratPtr->nrWavelengths;
			this->diffractionParamsPtr->lambdaPtr=(double*) calloc(parseResults_GratPtr->nrWavelengths, sizeof(double));
			this->diffractionParamsPtr->RTP01Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*parseResults_GratPtr->nrWavelengths,sizeof(double));
			this->diffractionParamsPtr->RTP10Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*parseResults_GratPtr->nrWavelengths,sizeof(double));
			this->diffractionParamsPtr->RTS01Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*parseResults_GratPtr->nrWavelengths,sizeof(double));
			this->diffractionParamsPtr->RTS10Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*parseResults_GratPtr->nrWavelengths,sizeof(double));
			// copy the efficiencies
			memcpy(&(this->diffractionParamsPtr->lambdaPtr[0]), parseResults_GratPtr->lambdaPtr, parseResults_GratPtr->nrWavelengths*sizeof(double));
			memcpy(&(this->diffractionParamsPtr->RTP01Ptr[0]), parseResults_GratPtr->RTP01Ptr, parseResults_GratPtr->nrWavelengths*parseResults_GratPtr->nrOrders*sizeof(double));
			memcpy(&(this->diffractionParamsPtr->RTP10Ptr[0]), parseResults_GratPtr->RTP10Ptr, parseResults_GratPtr->nrWavelengths*parseResults_GratPtr->nrOrders*sizeof(double));
			memcpy(&(this->diffractionParamsPtr->RTS10Ptr[0]), parseResults_GratPtr->RTS10Ptr, parseResults_GratPtr->nrWavelengths*parseResults_GratPtr->nrOrders*sizeof(double));
			memcpy(&(this->diffractionParamsPtr->RTS01Ptr[0]), parseResults_GratPtr->RTS01Ptr, parseResults_GratPtr->nrWavelengths*parseResults_GratPtr->nrOrders*sizeof(double));
		}
		else
		{
			// read efficiencies from zemax
			this->diffractionParamsPtr->nrWavelengths=1;
			this->diffractionParamsPtr->lambdaPtr=(double*) calloc(this->diffractionParamsPtr->nrWavelengths, sizeof(double));
			this->diffractionParamsPtr->RTP01Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*this->diffractionParamsPtr->nrWavelengths,sizeof(double));
			this->diffractionParamsPtr->RTP10Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*this->diffractionParamsPtr->nrWavelengths,sizeof(double));
			this->diffractionParamsPtr->RTS01Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*this->diffractionParamsPtr->nrWavelengths,sizeof(double));
			this->diffractionParamsPtr->RTS10Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*this->diffractionParamsPtr->nrWavelengths,sizeof(double));
			// fill the efficiencies such that all polarisations are equal and no coupling of polarizations occurs
			for (short i=0;i<this->diffractionParamsPtr->nrOrdersSim;i++)
			{
				short index=(short)((this->diffractionParamsPtr->nrOrders+1)/2-1+this->diffractionParamsPtr->diffOrdersPtr[i]);
				this->diffractionParamsPtr->RTP01Ptr[index]=parseResults_Mat.diffEff[i];
				this->diffractionParamsPtr->RTP10Ptr[index]=0;
				this->diffractionParamsPtr->RTS01Ptr[index]=0;
				this->diffractionParamsPtr->RTS10Ptr[index]=parseResults_Mat.diffEff[i];
			}
		}
	} // end if !gratingOrdersFromFile

	return MAT_NO_ERR;
}
	
/**
 * \detail parseXml 
 *
 * sets the parameters of the material according to the given xml node
 *
 * \param[in] xml_node &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialLinearGrating1D::parseXml(pugi::xml_node &material, SimParams simParams)
{
	if (!Material::parseXml(material, simParams))
	{
		std::cout << "error in MaterialLinearGrating1D.parseXml(): Material.parseXml() returned an error." << std::endl;
		return MAT_ERR;
	}

	this->glassDispersionParamsPtr=new MatRefracting_DispersionParams;
	this->immersionDispersionParamsPtr=new MatRefracting_DispersionParams;
	this->diffractionParamsPtr=new MatLinearGrating1D_DiffractionParams;

	Parser_XML l_parser;
	const char* l_glassName=l_parser.attrValByName(material, "glassName");
	if (l_glassName==NULL)
	{
		std::cout << "error in MaterialLinearGrating1D.parseXml(): glassName is not defined" << std::endl;
		return MAT_ERR;
	}
	if ( !strcmp(l_glassName,"MIRROR") )
	{
		//a mirror is mimicked with a refracting material with refractive index 0. So we fill the dispersion parameters with values that produce n=0 for all lambdas
		this->glassDispersionParamsPtr->lambdaMax=1000;
		this->glassDispersionParamsPtr->lambdaMin=0;
		this->glassDispersionParamsPtr->paramsNom[0]=0;
		this->glassDispersionParamsPtr->paramsNom[1]=0;
		this->glassDispersionParamsPtr->paramsNom[2]=0;
		this->glassDispersionParamsPtr->paramsNom[3]=0;
		this->glassDispersionParamsPtr->paramsNom[4]=0;
		this->glassDispersionParamsPtr->paramsDenom[0]=0;
		this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
	}
	else
	{
		// if we have a user defined glass we simply take the values of n1 and n2 defined in the prescription file and set no dispersion
		if (!strcmp(l_glassName,"USERDEFINED"))
		{
			if (!this->checkParserError(l_parser.attrByNameToDouble(material, "n1", this->params.nRefr1)))
				return MAT_ERR;

			this->glassDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
			this->glassDispersionParamsPtr->lambdaMin=0;
			this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
		}
		else // we have to parse the glass catalog
		{
			//char filepath[512];
			///* get handle to the glass catalog file */
			//sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
			//FILE* hfileGlass = fopen( filepath, "r" );
			FILE* hfileGlass = fopen(FILE_GLASSCATALOG, "r");
			if (!hfileGlass)
			{
				std::cout <<"error in MaterialLinearGrating1D.parseXml(): could not open glass catalog at: " << FILE_GLASSCATALOG  << std::endl;
				return MAT_ERR;
			}
			parseGlassResultStruct* parseResultsGlassPtr;
			/* parse Zemax glass catalog */
			if ( PARSER_NO_ERR != parseZemaxGlassCatalog(&parseResultsGlassPtr, hfileGlass, l_glassName) )
			{
				std::cout <<"error in MaterialLinearGrating1D.parseXml(): parseZemaxGlassCatalogOld() returned an error." << std::endl;
				return MAT_ERR;
			}
			this->glassDispersionParamsPtr->lambdaMax=parseResultsGlassPtr->lambdaMax;
			this->glassDispersionParamsPtr->lambdaMin=parseResultsGlassPtr->lambdaMin;
			memcpy(this->glassDispersionParamsPtr->paramsDenom, parseResultsGlassPtr->paramsDenom, 6*sizeof(double));
			memcpy(this->glassDispersionParamsPtr->paramsNom, parseResultsGlassPtr->paramsNom, 6*sizeof(double));
			switch (parseResultsGlassPtr->dispersionFormulaIndex)
			{
				case 1:
					this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
					break;
				case 2:
					this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
					break;
				default:
					this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
					std::cout <<"error in MaterialLinearGrating1D.parseXml(): unknown material dispersion formula" << std::endl;
					return MAT_ERR;
					break;
			}

			fclose(hfileGlass);
			delete parseResultsGlassPtr;
		}
	}

	// parse immersion medium
	const char* l_immersionName=l_parser.attrValByName(material, "immersionName");
	if (l_immersionName==NULL)
	{
		std::cout << "error in MaterialLinearGrating1D.parseXml(): glassName is not defined" << std::endl;
		return MAT_ERR;
	}
	// if we have a user defined glass we simply take the values of n1 and n2 defined in the prescription file and set no dispersion
	if (!strcmp(l_immersionName,"USERDEFINED"))
	{
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "n2", this->params.nRefr2)))
			return MAT_ERR;

		this->immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
		this->immersionDispersionParamsPtr->lambdaMin=0;
		this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
	}
	else
	{
		char filepath[512];
		/* get handle to the glass catalog file */
		//sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
		//FILE* hfileGlass = fopen( filepath, "r" );
		FILE* hfileGlass = fopen(FILE_GLASSCATALOG, "r");
		if (!hfileGlass)
		{
			std::cout <<"error in MaterialLinearGrating1D.parseXml(): could not open glass catalog at: " << filepath  << std::endl;
			return MAT_ERR;
		}
		parseGlassResultStruct* parseResultsGlassPtr;
		/* parse Zemax glass catalog */
		if ( PARSER_NO_ERR != parseZemaxGlassCatalog(&parseResultsGlassPtr, hfileGlass, l_immersionName) )
		{
			std::cout <<"error in MaterialLinearGrating1D.parseXml(): parseZemaxGlassCatalogOld() returned an error." << std::endl;
			return MAT_ERR;
		}
		this->immersionDispersionParamsPtr->lambdaMax=parseResultsGlassPtr->lambdaMax;
		this->immersionDispersionParamsPtr->lambdaMin=parseResultsGlassPtr->lambdaMin;
		memcpy(this->immersionDispersionParamsPtr->paramsDenom, parseResultsGlassPtr->paramsDenom, 6*sizeof(double));
		memcpy(this->immersionDispersionParamsPtr->paramsNom, parseResultsGlassPtr->paramsNom, 6*sizeof(double));
		switch (parseResultsGlassPtr->dispersionFormulaIndex)
		{
			case 1:
				this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
				break;
			case 2:
				this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
				break;
			default:
				this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
				std::cout <<"error in MaterialLinearGrating1D.parseXml(): unknown material dispersion formula" << std::endl;
				return MAT_ERR;
				break;
		}

		fclose(hfileGlass);
		delete parseResultsGlassPtr;
	}

	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffAxis.x", this->diffractionParamsPtr->diffAxis.x)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffAxis.y", this->diffractionParamsPtr->diffAxis.y)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffAxis.z", this->diffractionParamsPtr->diffAxis.z)))
		return MAT_ERR;

	// so far we only allow for exactly 9 diffraction orders
	this->diffractionParamsPtr->nrOrders=9;
	this->diffractionParamsPtr->diffOrdersPtr=(short*)calloc(this->diffractionParamsPtr->nrOrders,sizeof(short));

	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x1", this->diffractionParamsPtr->diffOrdersPtr[0])))
		return MAT_ERR;

	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x2", this->diffractionParamsPtr->diffOrdersPtr[1])))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x3", this->diffractionParamsPtr->diffOrdersPtr[2])))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x4", this->diffractionParamsPtr->diffOrdersPtr[3])))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x5", this->diffractionParamsPtr->diffOrdersPtr[4])))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x6", this->diffractionParamsPtr->diffOrdersPtr[5])))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x7", this->diffractionParamsPtr->diffOrdersPtr[6])))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x8", this->diffractionParamsPtr->diffOrdersPtr[7])))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToShort(material, "diffOrder.x9", this->diffractionParamsPtr->diffOrdersPtr[8])))
		return MAT_ERR;
	// find number of orders that are to be simulated
	// the orders have to be ordered from -max to plus max. Therefore, if we encounter the second zero, we end

	short maxOrder=0;
	bool zeroFound=false;
	this->params.nrDiffOrders=0;
	// find highest diffraction order that is to be simulated
	for (int j=0;j<9;j++)
	{
		if (this->diffractionParamsPtr->diffOrdersPtr[j]==0) 
		{
			if (zeroFound)
				break;
			zeroFound=true;
		}	
		this->params.nrDiffOrders++;

		if (maxOrder<abs(this->diffractionParamsPtr->diffOrdersPtr[j]))
			maxOrder=abs(this->diffractionParamsPtr->diffOrdersPtr[j]);
	}
	this->diffractionParamsPtr->nrOrders=2*maxOrder+1;
	this->diffractionParamsPtr->nrOrdersSim=this->params.nrDiffOrders;

	const char* l_DiffFileName=l_parser.attrValByName(material, "diffFileName");
	if (l_DiffFileName==NULL)
	{
		std::cout << "error in MaterialLinearGrating1D.parseXml(): diffFileName is not defined" << std::endl;
		return MAT_ERR;
	}
	// if we have a user defined glass we simply take the values of n1 and n2 defined in the prescription file and set no dispersion
	if (!strcmp(l_DiffFileName,"USERDEFINED"))
	{
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x1", this->params.eff[0])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x2", this->params.eff[1])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x3", this->params.eff[2])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x4", this->params.eff[3])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x5", this->params.eff[4])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x6", this->params.eff[5])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x7", this->params.eff[6])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x8", this->params.eff[7])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "diffEff.x9", this->params.eff[8])))
			return MAT_ERR;
		if (!this->checkParserError(l_parser.attrByNameToDouble(material, "gratingPeriod", this->diffractionParamsPtr->g)))
			return MAT_ERR;
		this->diffractionParamsPtr->g=this->diffractionParamsPtr->g*1e-3;

		// read efficiencies from zemax
		this->diffractionParamsPtr->nrWavelengths=1;
		this->diffractionParamsPtr->lambdaPtr=(double*) calloc(this->diffractionParamsPtr->nrWavelengths, sizeof(double));
		this->diffractionParamsPtr->RTP01Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*this->diffractionParamsPtr->nrWavelengths,sizeof(double));
		this->diffractionParamsPtr->RTP10Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*this->diffractionParamsPtr->nrWavelengths,sizeof(double));
		this->diffractionParamsPtr->RTS01Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*this->diffractionParamsPtr->nrWavelengths,sizeof(double));
		this->diffractionParamsPtr->RTS10Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*this->diffractionParamsPtr->nrWavelengths,sizeof(double));

		// fill the efficiencies such that all polarisations are equal and no coupling of polarizations occurs
		for (short i=0;i<this->params.nrDiffOrders;i++)
		{
			short index=(short)((this->diffractionParamsPtr->nrOrders+1)/2-1+this->diffractionParamsPtr->diffOrdersPtr[i]);
			this->diffractionParamsPtr->RTP01Ptr[index]=this->params.eff[i];
			this->diffractionParamsPtr->RTP10Ptr[index]=0;
			this->diffractionParamsPtr->RTS01Ptr[index]=0;
			this->diffractionParamsPtr->RTS10Ptr[index]=this->params.eff[i];
		}

	}
	else // we have to parse the grating file
	{
		char l_diffFile[512];
		sprintf(l_diffFile, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH,l_DiffFileName);
		FILE* hfileGlass = fopen(l_diffFile, "r");
		if (!hfileGlass)
		{
			std::cout <<"error in MaterialLinearGrating1D.parseXml(): could not open diff file at: " << l_diffFile  << std::endl;
			return MAT_ERR;
		}
		ParseGratingResultStruct* parseResultsGratingPtr;
		/* parse Zemax glass catalog */
		if ( PARSER_NO_ERR != parseMicroSimGratingData(&parseResultsGratingPtr, hfileGlass) )
		{
			std::cout <<"error in MaterialLinearGrating1D.parseXml(): parseZemaxGlassCatalogOld() returned an error." << std::endl;
			return MAT_ERR;
		}
		this->diffractionParamsPtr->g=parseResultsGratingPtr->g;

		// allocate memory for the efficiencies
		this->diffractionParamsPtr->nrWavelengths=parseResultsGratingPtr->nrWavelengths;
		this->diffractionParamsPtr->lambdaPtr=(double*) calloc(parseResultsGratingPtr->nrWavelengths, sizeof(double));
		this->diffractionParamsPtr->RTP01Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*parseResultsGratingPtr->nrWavelengths,sizeof(double));
		this->diffractionParamsPtr->RTP10Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*parseResultsGratingPtr->nrWavelengths,sizeof(double));
		this->diffractionParamsPtr->RTS01Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*parseResultsGratingPtr->nrWavelengths,sizeof(double));
		this->diffractionParamsPtr->RTS10Ptr=(double*) calloc(this->diffractionParamsPtr->nrOrders*parseResultsGratingPtr->nrWavelengths,sizeof(double));
		// copy the efficiencies
		memcpy(&(this->diffractionParamsPtr->lambdaPtr[0]), parseResultsGratingPtr->lambdaPtr, parseResultsGratingPtr->nrWavelengths*sizeof(double));
		memcpy(&(this->diffractionParamsPtr->RTP01Ptr[0]), parseResultsGratingPtr->RTP01Ptr, parseResultsGratingPtr->nrWavelengths*parseResultsGratingPtr->nrOrders*sizeof(double));
		memcpy(&(this->diffractionParamsPtr->RTP10Ptr[0]), parseResultsGratingPtr->RTP10Ptr, parseResultsGratingPtr->nrWavelengths*parseResultsGratingPtr->nrOrders*sizeof(double));
		memcpy(&(this->diffractionParamsPtr->RTS10Ptr[0]), parseResultsGratingPtr->RTS10Ptr, parseResultsGratingPtr->nrWavelengths*parseResultsGratingPtr->nrOrders*sizeof(double));
		memcpy(&(this->diffractionParamsPtr->RTS01Ptr[0]), parseResultsGratingPtr->RTS01Ptr, parseResultsGratingPtr->nrWavelengths*parseResultsGratingPtr->nrOrders*sizeof(double));


	}

	return MAT_NO_ERR;
}