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

/**\file Coating_NumCoeffs.cpp
* \brief Coating with predefined coefficients for transmissione and reflection
* 
*           
* \author Mauch
*/

#include "Coating_NumCoeffs.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <string.h>
#include <iostream>

#include "Parser_XML.h"

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
CoatingError Coating_NumCoeffs::createCPUSimInstance(double lambda)
{
	this->reducedParamsPtr->t=this->fullParamsPtr->t;
	this->reducedParamsPtr->r=this->fullParamsPtr->r;
	this->update=false;
	return COAT_NO_ERROR;	
};

CoatingError Coating_NumCoeffs::setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_coatingParamsPtr)
{
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( *closest_hit_programPtr, "coating_params", l_coatingParamsPtr ), context) )
		return COAT_ERROR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(*l_coatingParamsPtr, sizeof(Coating_NumCoeffs_ReducedParams), (this->reducedParamsPtr)), context) )
		return COAT_ERROR;

	return COAT_NO_ERROR;
};

/**
 * \detail setFullParams
 *
 * \param[in] Coating_NumCoeffs_FullParams* ptrIn
 * 
 * \return CoatingError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_NumCoeffs::setFullParams(Coating_NumCoeffs_FullParams* ptrIn)
{
	this->fullParamsPtr=ptrIn;
	return COAT_NO_ERROR;
};

/**
 * \detail getFullParams
 *
 * \param[in] void
 * 
 * \return Coating_NumCoeffs_FullParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_NumCoeffs_FullParams* Coating_NumCoeffs::getFullParams(void)
{
	return this->fullParamsPtr;
};

/**
 * \detail getReducedParams
 *
 * \param[in] void
 * 
 * \return Coating_NumCoeffs_ReducedParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating_NumCoeffs_ReducedParams* Coating_NumCoeffs::getReducedParams(void)
{
	return this->reducedParamsPtr;
};

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
bool  Coating_NumCoeffs::hit(rayStruct &ray, Mat_hitParams hitParams)
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
bool Coating_NumCoeffs::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal)
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
CoatingError Coating_NumCoeffs::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	this->fullParamsPtr->type=CT_NUMCOEFFS;
	this->fullParamsPtr->r=parseResults_Mat.coating_r;
	this->fullParamsPtr->t=parseResults_Mat.coating_t;
	return COAT_NO_ERROR;
}

/**
 * \detail parseXml 
 *
 * sets the parameters of the coating according to the given xml node
 *
 * \param[in] xml_node &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
CoatingError Coating_NumCoeffs::parseXml(pugi::xml_node &coating)
{
	Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToDouble(coating, "tA", this->fullParamsPtr->t)))
		return COAT_ERROR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(coating, "rA", this->fullParamsPtr->r)))
		return COAT_ERROR;

	return COAT_NO_ERROR;
}

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
bool Coating_NumCoeffs::checkParserError(char *msg)
{
	if (msg==NULL)
		return true;
	else
	{
		cout << "error in Coating_NumCoeffs.parseXML(): " << msg << endl;
		delete msg;
		msg=NULL;
		return false;
	}
};
