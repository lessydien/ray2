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

/**\file MaterialReflecting.cpp
* \brief reflecting material
* 
*           
* \author Mauch
*/

#include "MaterialReflecting.h"
#include "GlobalConstants.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>

/**
 * \detail hit function of material for geometric rays
 *
 * Here we need to call the hit function of the coating first. If the coating transmits the ray through the material it passes without any further deflectiosn. Finally we call the hit function of the Scatter attached to the material
 *
 * \param[in] rayStruct &ray, double3 normal, double t_hit, int geometryID
 * 
 * \return MaterialError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void MaterialReflecting::hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID)
{
//	extern Group oGroup;
	double3 n=hitParams.normal;
	bool coat_reflected = true;
	if ( this->coatingPtr->getFullParams()->type != CT_NOCOATING )
		coat_reflected=this->coatingPtr->hit(ray, hitParams);
	// if the coating wants transmission we do not change the ray direction at all !!!
	if (coat_reflected)
		hitReflecting(ray, hitParams, t_hit, geometryID);
	if ( this->scatterPtr->getFullParams()->type != ST_NOSCATTER )
		this->scatterPtr->hit(ray, hitParams);

	if (ray.depth>MAX_DEPTH_CPU || ray.flux<MIN_FLUX_CPU)
		ray.running=false;//stop ray
}

/**
 * \detail hit function of the material for gaussian beam rays
 *
 * \param[in] gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID
 * 
 * \return void
 * \sa 
 * \remarks not tested yet
 * \author Mauch
 */
void MaterialReflecting::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID)
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
MaterialError MaterialReflecting::processParseResults(MaterialParseParamStruct &parseResults_Mat)
{
	// Theres really nothing to do here...
	return MAT_NO_ERR;
};
