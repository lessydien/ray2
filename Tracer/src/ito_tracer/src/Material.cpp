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

/**\file Material.cpp
* \brief base class of all materials that can be applied to the geometries
* 
*           
* \author Mauch
*/

#include "Material.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "ScatterLib.h"
#include "CoatingLib.h"
#include "differentialRayTracing\ScatterLib_DiffRays.h"
#include "differentialRayTracing\CoatingLib_DiffRays.h"

#include "Parser_XML.h"

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
MaterialError Material::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	std::cout << "error in Material.processParseResults(): not defined for the given Field representation" << std::endl;
	return MAT_ERR;
};

/**
 * \detail parseXml 
 *
 * sets the parameters of the detector according to the given xml node
 *
 * \param[in] xml_node &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError Material::parseXml(pugi::xml_node &material, SimParams simParams)
{
	Parser_XML l_parser;
	vector<xml_node>* l_pScatNodes;
	vector<xml_node>* l_pCoatNodes;
	l_pCoatNodes=l_parser.childsByTagName(material,"coating");
	l_pScatNodes=l_parser.childsByTagName(material,"scatter");
	if ((l_pScatNodes->size()!=1) || (l_pCoatNodes->size()!=1))
	{
		std::cout << "error in Material.parseXml(): there must be exactly 1 scatter and 1 coating attached to each material." << std::endl;
		return MAT_ERR;
	}
	ScatterFab* l_pScatFab;
    CoatingFab* l_pCoatFab;
    switch (simParams.simMode)
    {
    case SIM_GEOM_RT:
        l_pScatFab=new ScatterFab();
        l_pCoatFab=new CoatingFab();
        break;
    case SIM_DIFF_RT:
        l_pScatFab=new ScatterFab_DiffRays();
        l_pCoatFab=new CoatingFab_DiffRays();
        break;
    default:
        std::cout << "error in Material.parseXml(): unknown simulation mode." << std::endl;
        return MAT_ERR;
        break;
    }

	Scatter* l_pScatter;
	if (!l_pScatFab->createScatInstFromXML(l_pScatNodes->at(0),l_pScatter, simParams))
	{
		std::cout << "error in Material.parseXml(): ScatFab.createScatInstFromXML() returned an error." << std::endl;
		return MAT_ERR;
	}
	this->setScatter(l_pScatter);

	Coating* l_pCoat;
	if (!l_pCoatFab->createCoatInstFromXML(l_pCoatNodes->at(0),l_pCoat, simParams))
	{
		std::cout << "error in Material.parseXml(): CoatFab.createCoatInstFromXML() returned an error." << std::endl;
		return MAT_ERR;
	}
	this->setCoating(l_pCoat);

	delete l_pScatNodes;
	delete l_pCoatNodes;
    delete l_pCoatFab;
    delete l_pScatFab;

	return MAT_NO_ERR;
}

/**
 * \detail setPathToPtx 
 *
 * sets the path to the ptx file that the .cu file defining the behaviour of the Material on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Material::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the Material on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return char* path
 * \sa 
 * \remarks 
 * \author Mauch
 */
char* Material::getPathToPtx(void)
{
	return this->path_to_ptx;
};

/**
 * \detail createMaterialHitProgramPtx 
 *
 * creates a ptx file from the given path
 *
 * \param[in] RTcontext context
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError Material::createMaterialHitProgramPtx(RTcontext context, SimParams simParams)
{
	if ( (simParams.simMode==SIM_DIFF_RT) )
		strcat(this->path_to_ptx, "_DiffRays");
	strcat(this->path_to_ptx, ".cu.ptx");

	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx, "closestHit", &closest_hit_program ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx, "anyHit", &any_hit_program ), context) )
		return MAT_ERR;

	return MAT_NO_ERR;
}

/**
 * \detail createOptiXInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError Material::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	// create simulation instance of scatter
	if (SCAT_NO_ERROR != this->getScatter()->createOptiXInstance(lambda, &(this->path_to_ptx)) )
	{
		std::cout << "error in Material.createOptiXInstance(): Scatter.createOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}
	// create simulation instance of coating
	if (COAT_NO_ERROR != this->getCoating()->createOptiXInstance(lambda, &(this->path_to_ptx)) )
	{
		std::cout << "error in Material.createOptiXInstance(): Coating.createOptiXInstance() returned an error" << std::endl;
		return MAT_ERR;
	}

	if (MAT_NO_ERR != createMaterialHitProgramPtx(context, simParams))
	{
		std::cout << "error in Material.createOptiXInstance(): createMaterialHitProgramPtx() returned an error" << std::endl;
		return MAT_ERR;
	}

	if ( !RT_CHECK_ERROR_NOEXIT( rtMaterialCreate( context, &OptiXMaterial ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtMaterialSetClosestHitProgram( OptiXMaterial, 0, closest_hit_program ), context) )
		return MAT_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtMaterialSetAnyHitProgram( OptiXMaterial, 0, any_hit_program ), context) )
		return MAT_ERR;

	// set scatter parameters
	this->scatterPtr->setParams2Program( context, &closest_hit_program, &l_scatterParams);
	
	// set coating parameters
	this->coatingPtr->setParams2Program( context, &closest_hit_program, &l_coatingParams);

	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceSetMaterial( instance, index, OptiXMaterial ), context) )
		return MAT_ERR;

	this->update=false;

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
MaterialError Material::updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda)
{
	if ( (this->getCoating()->update)||(this->lambda_old!=lambda) )
	{
		// create simulation instance of coating
		this->getCoating()->createOptiXInstance(lambda, &path_to_ptx);

		// query coating params from program
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "coating_params", &l_coatingParams ), context) )
			return MAT_ERR;

		// set coating parameters
		this->getCoating()->setParams2Program( context, &closest_hit_program, &l_coatingParams);

		this->getCoating()->update=false;
	}
	if ( (this->getScatter()->update)||(this->lambda_old!=lambda) )
	{
		// create simulation instance of scatter
		this->getScatter()->createOptiXInstance(lambda, &path_to_ptx);

		// query scatter params from program
		if ( !RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( closest_hit_program, "scatterParams", &l_scatterParams ), context) )
			return MAT_ERR;

		// set scatter parameters
		this->getScatter()->setParams2Program( context, &closest_hit_program, &l_scatterParams);

		this->getScatter()->update=false;
	}
	lambda_old=lambda;
	this->update=false;

	return MAT_NO_ERR;	};

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
MaterialError Material::createCPUSimInstance(double lambda)
{
	// create simulation instance of coating
	if (COAT_NO_ERROR != this->getCoating()->createCPUSimInstance(lambda) )
	{
		std::cout << "error in Material.createCPUSimInstance(): Coating.createCPUSimInstance() returned an error." << std::endl;
		return MAT_ERR;
	}
	// create simulation instance of scatter
	if (SCAT_NO_ERROR != this->getScatter()->createCPUSimInstance(lambda))
	{
		std::cout << "error in Material.createCPUSimInstance(): Scatter.createCPUSimInstance() returned an error." << std::endl;
		return MAT_ERR;
	}
	this->update=false;
	return MAT_NO_ERR;
};

/**
 * \detail updateCPUSimInstance 
 *
 * \param[in] RTcontext context, RTgeometryinstance &instance, int index, TraceMode mode, double lambda
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError Material::updateCPUSimInstance(double lambda)
{
	std::cout << "error in Material.updateCPUSimInstance(): not defined for the given Material" << std::endl;
	return MAT_ERR;
};

/**
 * \detail calcSourceImmersion 
 *
 * \param[in] double lambda
 * 
 * \return double nRefr
 * \sa 
 * \remarks 
 * \author Mauch
 */
double Material::calcSourceImmersion(double lambda)
{
	std::cout << "error in Material.calcSourceImmersion(): not defined for the given Material" << std::endl;
	return 0;	// if the function is not overwritten by the child class, we return a standard value of one for the refractive index of the immersion material
};

/**
 * \detail setCoating 
 *
 * \param[in] Coating* ptrIn
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError Material::setCoating(Coating* ptrIn)
{
	this->coatingPtr=ptrIn;
	return MAT_NO_ERR;
};

/**
 * \detail getCoating 
 *
 * \param[in] void
 * 
 * \return Coating*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Coating* Material::getCoating(void)
{
	return this->coatingPtr;
};

/**
 * \detail setScatter 
 *
 * \param[in] Scatter* ptrIn
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
MaterialError Material::setScatter(Scatter* ptrIn)
{
	this->scatterPtr=ptrIn;
	return MAT_NO_ERR;
};

/**
 * \detail getScatter 
 *
 * \param[in] void
 * 
 * \return Scatter*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Scatter* Material::getScatter(void)
{
	return this->scatterPtr;
};

/**
 * \detail hit function of the material for geometric rays
 *
 * \param[in] rayStruct &ray, double3 normal, double t, int geometryID
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Material::hit(rayStruct &ray, Mat_hitParams hitParams, double t, int geometryID)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Material.hit(): hit is not yet implemented for geometric rays for the given material. Material_DiffRays is ignored..." << std::endl;
};

/**
 * \detail hit function of the material for differential rays
 *
 * \param[in] diffRayStruct &ray, double3 normal, double t, int geometryID
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Material::hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID)
{
	// dummy function to be overwritten by child class
	std::cout << "error in Material.hit(): hit is not yet implemented for differential rays for the given material. Material is ignored..." << std::endl;
};

/**
 * \detail hit function of the material for gaussian beam rays
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Material::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID)
{
	std::cout << "error in Material_DiffRays.hit(): hit is not yet implemented for differential rays for the given material. Material_DiffRays is ignored..." << std::endl;
	// dummy function to be overwritten by child class
};

/**
 * \detail setGlassDispersionParams
 *
 * \param[in] MatDispersionParams *params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Material::setGlassDispersionParams(MatDispersionParams *params)
{
	std::cout << "error in Material.setGlassDispersionParams(): not defined for the given Material" << std::endl;
};

/**
 * \detail getGlassDispersionParams
 *
 * \param[in] void
 * 
 * \return MatDispersionParams
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatDispersionParams* Material::getGlassDispersionParams(void)
{
	std::cout << "error in Material.getGlassDispersionParams(): not defined for the given Material" << std::endl;
	return NULL;
};

/**
 * \detail setImmersionDispersionParams
 *
 * \param[in] MatDispersionParams *params
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Material::setImmersionDispersionParams(MatDispersionParams *params)
{
	std::cout << "error in Material.setImmersionDispersionParams(): not defined for the given Material" << std::endl;
};

/**
 * \detail getImmersionDispersionParams
 *
 * \param[in] void
 * 
 * \return MatDispersionParams
 * \sa 
 * \remarks 
 * \author Mauch
 */
MatDispersionParams* Material::getImmersionDispersionParams(void)
{
	std::cout << "error in Material.getImmersionDispersionParams(): not defined for the given Material" << std::endl;
	return NULL;
};

/**
 * \detail checks wether parsing was succesfull and assembles the error message if it was not
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
bool Material::checkParserError(char *msg)
{
	if (msg==NULL)
		return true;
	else
	{
		cout << "error in Material.parseXML(): " << msg << endl;
		delete msg;
		msg=NULL;
		return false;
	}
};