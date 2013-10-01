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

/**\file MaterialDOE.cpp
* \brief DOE material
* 
*           
* \author Mauch
*/

#include "MaterialDOE.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "rayTracingMath.h"

#include "Parser_XML.h"
#include "Parser.h"


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
void MaterialDOE::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	
//	extern Group oGroup;
	bool coat_reflected=false;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);

	if ( hitDOE(ray, hitParams, this->params, this->coeffVec, this->effLookUpTable, t_hit, geometryID, coat_reflected) )
	{
		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
			this->scatterPtr->hit(ray, hitParams);
//		if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
//			ray.running=false;//stop ray

	}
	else
	{
		std::cout << "error in MaterialDOE.hit: hitDOE() returned an error" << std::endl;
		ray.running=false;
	}

}

/**
 * \detail createCPUSimInstance
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialDOE::createCPUSimInstance(double lambda)
{
	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR != calcRefrIndices(lambda))
	{
		std::cout << "error in MaterialLinearGrating1D.createCPUSimInstance(): calcRefrIndices() returned an error" << std::endl;
		return MAT_ERR;
	}

	// create simulation instance of coating
	if (MAT_NO_ERR != Material::createCPUSimInstance(lambda) )
	{
		std::cout << "error in MaterialLinearGrating1D.createCPUSimInstance(): Material.createCPUSimInstance() returned an error." << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;
}

/**
 * \detail createOptiXInstance
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks we have a seperate .cu file for each combination of material, scatter and coating. Therefore we set the path to that ptx file that corresponds to the combination present in the current instance
 * \author Mauch
 */
MaterialError MaterialDOE::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda)
{
	if (MAT_NO_ERR != Material::createOptiXInstance(context, instance, index, mode, lambda) )
	{
		std::cout << "error in MaterialDOE.createOptiXInstance(): Material.creatOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	lambda_old=lambda; // when creating the OptiXInstance we need to do this

	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR != calcRefrIndices(lambda))
	{
		std::cout << "error in MaterialLinearGrating1D.createOptiXInstance(): calcRefrIndices() returned an error" << std::endl;
		return MAT_ERR;
	}

	/* set the variables of the material */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatDOE_params), &(this->params)), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "coeffVec", &l_coeffVec ), context) )
		return MAT_ERR;
	//if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_coeffVec, this->params.coeffVecLength*sizeof(double), this->coeffVec), context) )
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_coeffVec, sizeof(MatDOE_coeffVec), &(this->coeffVec)), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( closest_hit_program, "effLookUpTable", &l_effLookUpTable ), context) )
		return MAT_ERR;
	//if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_effLookUpTable, this->params.effLookUpTableDims.x*this->params.effLookUpTableDims.y*this->params.effLookUpTableDims.z*sizeof(double), this->effLookUpTable), context) )
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_effLookUpTable, sizeof(MatDOE_lookUp), &(this->effLookUpTable)), context) )
		return MAT_ERR;

	MatDOE_lookUp test;
	RTresult res;
	res=rtVariableGetUserData(l_effLookUpTable, sizeof(MatDOE_lookUp), &test);

	return MAT_NO_ERR;	
};

/**
 * \detail updateOptiXInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialDOE::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda)
{
	if ( (this->update)||(this->lambda_old!=lambda) )
	{
		// calc the refractive indices at current wavelength
		if (MAT_NO_ERR != calcRefrIndices(lambda))
		{
			std::cout << "error in MaterialLinearGrating1D.createOptiXInstance(): calcRefrIndeices() returned an error" << std::endl;
			return MAT_ERR;
		}
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "params", &l_params ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_params, sizeof(MatDOE_params), &(this->params)), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "coeffVec", &l_coeffVec ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_coeffVec, sizeof(MatDOE_coeffVec), &(this->coeffVec)), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "effLookUpTable", &l_effLookUpTable ), context) )
			return MAT_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_effLookUpTable, sizeof(MatDOE_lookUp), &(this->effLookUpTable)), context) )
			return MAT_ERR;


		this->update=false;
	}

	if (MAT_NO_ERR != Material::updateOptiXInstance(context, instance, index, mode, lambda) )
	{
		std::cout << "error in MaterialDOE.updateOptiXInstance(): Material.updateOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	return MAT_NO_ERR;	
};


/**
 * \detail setParams 
 *
 * \param[in] MatDOE_params params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialDOE::setParams(MatDOE_params params)
{
	this->update=true;
	this->params=params;
}

/**
 * \detail getParams 
 *
 * \param[in] void
 * 
 * \return MatDOEParams
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatDOE_params MaterialDOE::getParams(void)
{
	return this->params;
}

/**
 * \detail setGlassDispersionParams 
 *
 * \param[in] MatDOE_DispersionParams* paramsInPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialDOE::setGlassDispersionParams(MatDOE_DispersionParams* paramsInPtr)
{
	this->glassDispersionParamsPtr=paramsInPtr;
}

/**
 * \detail getGlassDispersionParams 
 *
 * \param[in] void
 * 
 * \return MatDOE_DispersionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatDOE_DispersionParams* MaterialDOE::getGlassDispersionParams(void)
{
	return this->glassDispersionParamsPtr;
}

/**
 * \detail getImmersionDispersionParams 
 *
 * \param[in] void
 * 
 * \return MatDOE_DispersionParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatDOE_DispersionParams* MaterialDOE::getImmersionDispersionParams(void)
{
	return this->immersionDispersionParamsPtr;
}

/**
 * \detail setImmersionDispersionParams 
 *
 * \param[in] MatDOE_DispersionParams*
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialDOE::setImmersionDispersionParams(MatDOE_DispersionParams* paramsInPtr)
{
	this->immersionDispersionParamsPtr=paramsInPtr;
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
MaterialError MaterialDOE::calcRefrIndices(double lambda)
{
	double l_lambda=lambda*1e3; // within the glass catalog lambda is assumed to be in microns. In our trace we have lambda in mm. We calculate a local lambda here in microns so we can use the coefficients from the glass catalog without rescaling...
	// calc refractive index of glass material for current wavelength
	if ( (l_lambda<glassDispersionParamsPtr->lambdaMin)||(l_lambda>glassDispersionParamsPtr->lambdaMax) )
	{
		std::cout << "error in MaterialDOE.calcRefrIndices(): lambda outside of range of glass definition at lambda=" << lambda << std::endl;
		return MAT_ERR;
	}
	switch (this->glassDispersionParamsPtr->dispersionFormula)
	{
		case MAT_DISPFORMULA_SCHOTT:
			this->params.n1=sqrt(glassDispersionParamsPtr->paramsNom[0]+glassDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)+glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,-2)+glassDispersionParamsPtr->paramsNom[3]*pow(l_lambda,-4)+glassDispersionParamsPtr->paramsNom[4]*pow(l_lambda,-6)+glassDispersionParamsPtr->paramsDenom[0]*pow(l_lambda,-8));
			break;
		case MAT_DISPFORMULA_SELLMEIER1:
			//this->params.n1=sqrt(glassDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[0])+glassDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[1])+glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsDenom[2]));
			this->params.n1=sqrt(1+glassDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsNom[1])+glassDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsNom[3])+glassDispersionParamsPtr->paramsNom[4]*pow(l_lambda,2)/(pow(l_lambda,2)-glassDispersionParamsPtr->paramsNom[5]));
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
		std::cout << "error in MaterialDOE.calcRefrIndices(): lambda outside of range of immersion definition at lambda=" << lambda << std::endl;
		return MAT_ERR;
	}
	switch (this->immersionDispersionParamsPtr->dispersionFormula)
	{
		case MAT_DISPFORMULA_SCHOTT:
			this->params.n2=sqrt(immersionDispersionParamsPtr->paramsNom[0]+immersionDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)+immersionDispersionParamsPtr->paramsNom[2]*pow(l_lambda,-2)+immersionDispersionParamsPtr->paramsNom[3]*pow(l_lambda,-4)+immersionDispersionParamsPtr->paramsNom[4]*pow(l_lambda,-6)+immersionDispersionParamsPtr->paramsDenom[0]*pow(l_lambda,-8));
			break;
		case MAT_DISPFORMULA_SELLMEIER1:
			//this->params.n2=sqrt(immersionDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsDenom[0])+immersionDispersionParamsPtr->paramsNom[1]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsDenom[1])+immersionDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsDenom[2]));
			this->params.n2=sqrt(1+immersionDispersionParamsPtr->paramsNom[0]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsNom[1])+immersionDispersionParamsPtr->paramsNom[2]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsNom[3])+immersionDispersionParamsPtr->paramsNom[4]*pow(l_lambda,2)/(pow(l_lambda,2)-immersionDispersionParamsPtr->paramsNom[5]));
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
 * \detail calcSourceImmersion 
 *
 * \param[in] double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
double MaterialDOE::calcSourceImmersion(double lambda)
{
	// calc the refractive indices at current wavelength
	if (MAT_NO_ERR!=this->calcRefrIndices(lambda))
	{
		std::cout << "error in MaterialDOE.calcSourceImmersion(): calcRefrIndices() returned an error" << std::endl;
		return 0; // if we encounterd an error we return 0 to indicate this error
	}
	return this->params.n1;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] MaterialParseParamStruct &parseResults_Geom, parseGlassResultStruct &parseResults_Glass, parseGlassResultStruct &parseResults_Imm
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError MaterialDOE::processParseResults(MaterialParseParamStruct &parseResults_Mat, parseGlassResultStruct* parseResults_GlassPtr, parseGlassResultStruct* parseResults_ImmPtr)
{
	this->glassDispersionParamsPtr=new MatDOE_DispersionParams;
	this->immersionDispersionParamsPtr=new MatDOE_DispersionParams;

	// if we have a user defined glass we simply take the values of n1 and n2 defined in the prescription file and set no dispersion
	if (!strcmp(parseResults_Mat.glassName, "USERDEFINED"))
	{
		this->params.n1=parseResults_Mat.nRefr.x;
		this->glassDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
		this->glassDispersionParamsPtr->lambdaMin=0;
		this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
	}
	else
	{
		this->glassDispersionParamsPtr->lambdaMax=parseResults_GlassPtr->lambdaMax;
		this->glassDispersionParamsPtr->lambdaMin=parseResults_GlassPtr->lambdaMin;
		memcpy(this->glassDispersionParamsPtr->paramsDenom, parseResults_GlassPtr->paramsDenom, 6*sizeof(double));
		memcpy(this->glassDispersionParamsPtr->paramsNom, parseResults_GlassPtr->paramsNom, 6*sizeof(double));
		switch (parseResults_GlassPtr->dispersionFormulaIndex)
		{
			case 1:
				this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
				break;
			case 2:
				this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
				break;
			default:
				this->glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
				std::cout <<"error in MaterialDOE.processParseResults(): unknown material dispersion formula" << std::endl;
				return MAT_ERR;
				break;
		}
	}

	// if  we have a user defined immersion medium set it
	if (!strcmp(parseResults_Mat.immersionName,"USERDEFINED"))
	{
		this->params.n2=parseResults_Mat.nRefr.y;
		this->immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
		this->immersionDispersionParamsPtr->lambdaMin=0;
		this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
	}
	else
	{
		// if we have no immersion medium specified, set it to n=1
		if (!strcmp(parseResults_Mat.immersionName,"STANDARD"))
		{
			this->params.n2=1;
			this->immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
			this->immersionDispersionParamsPtr->lambdaMin=0;
			this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
		}
		else // if we have a immersion medium specified, parse for it in the glass catalog
		{
			this->immersionDispersionParamsPtr->lambdaMax=parseResults_ImmPtr->lambdaMax;
			this->immersionDispersionParamsPtr->lambdaMin=parseResults_ImmPtr->lambdaMin;
			memcpy(this->immersionDispersionParamsPtr->paramsDenom, parseResults_ImmPtr->paramsDenom, 6*sizeof(double));
			memcpy(this->immersionDispersionParamsPtr->paramsNom, parseResults_ImmPtr->paramsNom, 6*sizeof(double));
			switch (parseResults_ImmPtr->dispersionFormulaIndex)
			{
				case 1:
					this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SCHOTT;
					break;
				case 2:
					this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_SELLMEIER1;
					break;
				default:
					this->immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_UNKNOWN;
					std::cout <<"error in MaterialDOE.processParseResults(): unknown material dispersion formula" << std::endl;
					return MAT_ERR;
					break;
			}
		} // end parsing for immersion medium
	} // end of "if immersion medium is user defined"


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
MaterialError MaterialDOE::parseXml(pugi::xml_node &material)
{
	if (!Material::parseXml(material))
	{
		std::cout << "error in MaterialDOE.parseXml(): Material.parseXml() returned an error." << std::endl;
		return MAT_ERR;
	}

	this->glassDispersionParamsPtr=new MatDOE_DispersionParams;
	this->immersionDispersionParamsPtr=new MatDOE_DispersionParams;

	Parser_XML l_parser;

	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "stepHeight", this->params.stepHeight)))
		return MAT_ERR;

	if (!this->checkParserError(l_parser.attrByNameToInt(material, "DOEnr", this->params.dOEnr)))
		return MAT_ERR;

	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomRoot.x", this->params.geomRoot.x)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomRoot.y", this->params.geomRoot.y)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomRoot.z", this->params.geomRoot.z)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomTilt.x", this->params.geomTilt.x)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomTilt.y", this->params.geomTilt.y)))
		return MAT_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(material, "geomTilt.z", this->params.geomTilt.z)))
		return MAT_ERR;

	// read DOE efficiencies
	const char* l_DOEEffsBaseFileName=l_parser.attrValByName(material, "filenameBaseDOEEffs");
	if (l_DOEEffsBaseFileName==NULL)
	{
		std::cout << "error in MaterialDOE.parseXml(): filenameBaseDOEEffs is not defined" << std::endl;
		return MAT_ERR;
	}
    char buf[15001];
	// we read data into a local buffer that we preallocate to 100000 elements here.
	// Afterwards we know how many elements were read, then we allocate the respective amount of memory 
	// for the member variable effLookUpTable and copy the data
	// note this field needs to be boigger than the amount of data that we read from the files!!!
	double *l_pLookUp=(double*)malloc(15000*sizeof(double));
	char *sub_string;
	long linIdx=0;
    // Beugungseffizienzen einlesen
    for (int i=-10;i<11;i++)
    // für jede Beugungsordnung eine 2D-Matrix (Perioden, Höhen)
    // einlesen, die dann interpoliert wird
    // In der ersten Zeile stehen die [0 Höhenwerte[relativ zur 2pi-Designhöhe]].
    // In der ersten Spalte stehen die [0 Periodenwerte[mm]].
    // In den restlichen Einträgen stehen die Beugungseffizienzen.
    {
		char l_DOEEffsfile[512];
		sprintf(l_DOEEffsfile, "%s" PATH_SEPARATOR "%s%i.txt", INPUT_FILEPATH, l_DOEEffsBaseFileName, i);

		FILE* hfileDOEEffs = fopen(l_DOEEffsfile, "r");
		if (!hfileDOEEffs)
		{
			std::cout <<"error in MaterialDOE.parseXml(): could not open DOE efficiency file at: " << l_DOEEffsfile  << std::endl;
			return MAT_ERR;
		}

		this->params.effLookUpTableDims.y=0;
		while(fgets(buf, 15000, hfileDOEEffs))
        {
			this->params.effLookUpTableDims.x=1;
            // Tokens lesen
			l_pLookUp[linIdx]= atof(strtok(buf, " "));
			linIdx++;
            while ( (sub_string=strtok(NULL, " ")) != NULL)
            {
				this->params.effLookUpTableDims.x++;
                l_pLookUp[linIdx]=atof(sub_string);
				linIdx++;
            }
            this->params.effLookUpTableDims.y++; // zählt die Zeilen hoch
        }
		fclose(hfileDOEEffs);
		this->params.effLookUpTableDims.z++;
    }

	// effLookUpTable is organized as an 1D-array
	// efficiencies[z][s][i+10] in Prussens old code becomes effLookUpTable[s+z*effLookUpTableDims.x+(i+10)*effLookUpTableDims.x*effLookUpTableDims.y] here...
//	this->effLookUpTable=(double*)malloc(this->params.effLookUpTableDims.x*this->params.effLookUpTableDims.y*this->params.effLookUpTableDims.z*sizeof(double));
//	memcpy(this->effLookUpTable, l_pLookUp, this->params.effLookUpTableDims.x*this->params.effLookUpTableDims.y*this->params.effLookUpTableDims.z*sizeof(double));
	memcpy(&(this->effLookUpTable.data[0]), l_pLookUp, 20*30*21*sizeof(double));
	delete l_pLookUp;

	// read DOE coeffs
	const char* l_DOEFileName=l_parser.attrValByName(material, "filenameDOE");
	if (l_DOEFileName==NULL)
	{
		std::cout << "error in MaterialDOE.parseXml(): filenameDOE is not defined" << std::endl;
		return MAT_ERR;
	}

	char l_DOEfile[512];
	sprintf(l_DOEfile, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, l_DOEFileName);

	FILE* hfileDOE = fopen(l_DOEfile, "r");
	if (!hfileDOE)
	{
		std::cout <<"error in MaterialDOE.parseXml(): could not open DOE file at: " << l_DOEfile  << std::endl;
		return MAT_ERR;
	}

	parseDOEResultStruct* parseResultsDOEPtr;
	/* parse doe coefficient file */
	if ( PARSER_NO_ERR != parseDOEFile(&parseResultsDOEPtr, hfileDOE, this->params.dOEnr) )
	{
		std::cout <<"error in MaterialDOE.parseXml(): parseDOEFile() returned an error." << std::endl;
		return MAT_ERR;
	}
//	this->coeffVec = (double*)malloc(parseResultsDOEPtr->coeffLength*sizeof(double));
//	memcpy(this->coeffVec, &(parseResultsDOEPtr->coeffArray[0]), parseResultsDOEPtr->coeffLength*sizeof(double));
	memcpy(&(this->coeffVec.data[0]), &(parseResultsDOEPtr->coeffArray[0]), min(44,parseResultsDOEPtr->coeffLength)*sizeof(double));
	this->params.coeffVecLength=min(44,parseResultsDOEPtr->coeffLength);

	fclose(hfileDOE);

	const char* l_glassName=l_parser.attrValByName(material, "glassName");
	if (l_glassName==NULL)
	{
		std::cout << "error in MaterialDOE.parseXml(): glassName is not defined" << std::endl;
		return MAT_ERR;
	}
	// if we have a user defined glass we simply take the values of n1 and n2 defined in the prescription file and set no dispersion
	if (!strcmp(l_glassName,"USERDEFINED"))
	{
		if (!l_parser.attrByNameToDouble(material, "n1", this->params.n1))
		{
			std::cout << "error in MaterialDOE.parseXml(): n1 is not defined" << std::endl;
			return MAT_ERR;
		}

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
			std::cout <<"error in MaterialDOE.parseXml(): could not open glass catalog at: " << FILE_GLASSCATALOG  << std::endl;
			return MAT_ERR;
		}
		parseGlassResultStruct* parseResultsGlassPtr;
		/* parse Zemax glass catalog */
		if ( PARSER_NO_ERR != parseZemaxGlassCatalog(&parseResultsGlassPtr, hfileGlass, l_glassName) )
		{
			std::cout <<"error in MaterialDOE.parseXml(): parseZemaxGlassCatalogOld() returned an error." << std::endl;
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
				std::cout <<"error in MaterialDOE.parseXml(): unknown material dispersion formula" << std::endl;
				return MAT_ERR;
				break;
		}

		fclose(hfileGlass);
		delete parseResultsGlassPtr;
	}

	// parse immersion medium
	const char* l_immersionName=l_parser.attrValByName(material, "immersionName");
	if (l_immersionName==NULL)
	{
		std::cout << "error in MaterialDOE.parseXml(): glassName is not defined" << std::endl;
		return MAT_ERR;
	}
	// if we have a user defined glass we simply take the values of n1 and n2 defined in the prescription file and set no dispersion
	if (!strcmp(l_immersionName,"USERDEFINED"))
	{
		if (!l_parser.attrByNameToDouble(material, "n2", this->params.n2))
		{
			std::cout << "error in MaterialDOE.parseXml(): n2 is not defined" << std::endl;
			return MAT_ERR;
		}

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
			std::cout <<"error in MaterialDOE.parseXml(): could not open glass catalog at: " << filepath  << std::endl;
			return MAT_ERR;
		}
		parseGlassResultStruct* parseResultsGlassPtr;
		/* parse Zemax glass catalog */
		if ( PARSER_NO_ERR != parseZemaxGlassCatalog(&parseResultsGlassPtr, hfileGlass, l_immersionName) )
		{
			std::cout <<"error in MaterialDOE.parseXml(): parseZemaxGlassCatalogOld() returned an error." << std::endl;
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
				std::cout <<"error in MaterialDOE.parseXml(): unknown material dispersion formula" << std::endl;
				return MAT_ERR;
				break;
		}

		fclose(hfileGlass);
		delete parseResultsGlassPtr;
	}

	return MAT_NO_ERR;
}