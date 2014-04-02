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

/**\file MaterialAbsorbing.cpp
* \brief absorbing material
* 
*           
* \author Mauch
*/

#include "MaterialAbsorbing.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>

#include "Parser_XML.h"


//void MaterialAbsorbing::setPathToPtx(char* path)
//{
//	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	
//};
//
//char* MaterialAbsorbing::getPathToPtx(void)
//{
//	return this->path_to_ptx;
//};

/**
 * \detail hit function of material for geometric rays
 *
 * we call hitAbsorbing that describes the interaction of the ray with the material and can be called from GPU as well. Then we call the hit function of the coatin attached to the Material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialAbsorbing::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	//ray.direction=make_double3(0.0);
	if ( hitAbsorbing(ray, hitParams, t_hit, geometryID) )
	{
		ray.running=false;//stop ray
		bool reflected;
		if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		{
			reflected=this->coatingPtr->hit(ray, hitParams);
			if (reflected)
			{
				ray.direction=reflect(ray.direction, hitParams.normal);
				ray.running=true; // keep ray alive
			}
		}
		else
			ray.running=false; // stop ray

		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
		{
			this->scatterPtr->hit(ray, hitParams);
			ray.running=true; // keep ray alive
		}
	}
}

/**
 * \detail hit function of material for differential rays
 *
 * we call hitAbsorbing that describes the interaction of the ray with the material and can be called from GPU as well. Then we call the hit function of the coatin attached to the Material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialAbsorbing::hit(diffRayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	//ray.direction=make_double3(0.0);
	if ( hitAbsorbing(ray, hitParams, t_hit, geometryID) )
	{
		ray.running=false;//stop ray
		bool reflected;
		if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		{
			reflected=this->coatingPtr->hit(ray, hitParams);
			if (reflected)
			{
				ray.direction=reflect(ray.direction, hitParams.normal);
				ray.running=true; // keep ray alive
			}
		}

		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
		{
			this->scatterPtr->hit(ray, hitParams);
			ray.running=true; // keep ray alive
		}
	}
}

/**
 * \detail hit function of material for gaussian beam rays
 *
 * we repeatedly call hitAbsorbing that describes the interaction of geometric rays with the material and can be called from GPU as well. Then we call the hit function of the coatin attached to the Material. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks not tested yet
 * \author Mauch
 */
void MaterialAbsorbing::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID)
{
	//ray.direction=make_double3(0.0);
	ray.baseRay.running=false;
	ray.baseRay.currentGeometryID=geometryID;
}

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
MaterialError MaterialAbsorbing::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	// Theres really nothing to do here...
	return MAT_NO_ERR;
};

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
MaterialError MaterialAbsorbing::parseXml(pugi::xml_node &geometry, SimParams simParams)
{
	if (!Material::parseXml(geometry, simParams))
	{
		std::cout << "error in MaterialAbsorbing.parseXml(): Material.parseXml() returned an error." << "...\n";
		return MAT_ERR;
	}

	return MAT_NO_ERR;
};