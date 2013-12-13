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

/**\file MaterialVolumeAbsorbing.cpp
* \brief absorbing material
* 
*           
* \author Mauch
*/

#include "MaterialVolumeAbsorbing.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>

#include "Parser_XML.h"


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
void MaterialVolumeAbsorbing::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
	bool coat_reflected=false;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);

	if ( hitVolumeAbsorbing(ray, hitParams, this->params, t_hit, geometryID, coat_reflected) )
	{
		if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
			this->scatterPtr->hit(ray, hitParams);
		if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
			ray.running=false;//stop ray
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
//void MaterialVolumeAbsorbing::hit(diffRayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
//{
//
//}

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
MaterialError MaterialVolumeAbsorbing::parseXml(pugi::xml_node &geometry)
{
	if (!Material::parseXml(geometry))
	{
		std::cout << "error in MaterialVolumeAbsorbing.parseXml(): Material.parseXml() returned an error." << std::endl;
		return MAT_ERR;
	}

	Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "absorbCoeff", this->params.absorbCoeff)))
		return MAT_ERR;


	return MAT_NO_ERR;
};