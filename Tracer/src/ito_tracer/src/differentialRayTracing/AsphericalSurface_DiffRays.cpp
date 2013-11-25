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

/**\file AsphericalSurface_DiffRays.cpp
* \brief aspheric surface
* 
*           
* \author Mauch
*/

#include "AsphericalSurface_DiffRays.h"
#include "../rayTracingMath.h"
#include <iostream>
#include "../myUtil.h"

/**
 * \detail intersect function for geometric rays
 *
 * \param[in] rayStruct ray
 * 
 * \return double t. That is the factor t for which r=ray.position+t*ray.direction is the intersection point of the ray with the surface
 * \sa 
 * \remarks This is a wrapper that calls the inline function intersectRayAsphere that can be called from GPU as well
 * \author Mauch
 */
double AsphericalSurface_DiffRays::intersect(diffRayStruct *ray)
{
	// we do not allow a ray to hit the asphere twice
	if (ray->currentGeometryID==this->paramsPtr->geometryID)
		return 0;
	else
		return intersectRayAsphere(ray->position,ray->direction,*(this->reducedParamsPtr));
};

/**
 * \detail reduceParams
 *
 * \param[in] void
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError AsphericalSurface_DiffRays::reduceParams(void)
{
	if ( (this->paramsPtr!=NULL) && (this->reducedParamsPtr!=NULL) )
	{
		this->reducedParamsPtr->geometryID=this->paramsPtr->geometryID;
		this->reducedParamsPtr->apertureRadius=this->paramsPtr->apertureRadius;
		this->reducedParamsPtr->apertureType=this->paramsPtr->apertureType;
//		this->reducedParamsPtr->rotNormal=this->paramsPtr->rotNormal;
		this->reducedParamsPtr->c10=this->paramsPtr->c10;
		this->reducedParamsPtr->c8=this->paramsPtr->c8;
		this->reducedParamsPtr->c6=this->paramsPtr->c6;
		this->reducedParamsPtr->c4=this->paramsPtr->c4;
		this->reducedParamsPtr->c2=this->paramsPtr->c2;
		this->reducedParamsPtr->c=this->paramsPtr->c;
		this->reducedParamsPtr->k=this->paramsPtr->k;
		this->reducedParamsPtr->orientation=this->paramsPtr->orientation;
		this->reducedParamsPtr->vertex=this->paramsPtr->vertex;
		this->reducedParamsPtr->tilt=this->paramsPtr->tilt;
	}
	return GEOM_NO_ERR;
};

/**
 * \detail hit function function for geometric rays
 *
 * we calc the normal to the surface in the intersection point. Then we call the hit function of the material that is attached to the surface
 *
 * \param[in] rayStruct ray
 * \param[in] double ray
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError AsphericalSurface_DiffRays::hit(diffRayStruct &ray,double t)
{
	Mat_DiffRays_hitParams hitParams;
	hitParams=calcHitParamsAsphere_DiffRays(ray.position+t*ray.direction, *(this->reducedParamsPtr));

	int i;
	for (i=0;i<this->materialListLength;i++)
	{
		this->getMaterial(i)->hit(ray,hitParams,t,this->paramsPtr->geometryID);
	}
		
	return GEOM_NO_ERR;
};

/**
 * \detail createOptixInstance
 *
 * we create an OptiX instance of the surface and the materials attached to it
 *
 * \param[in] RTcontext &context, RTgeometrygroup &geometrygroup, int index, TraceMode mode, double lambda
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError AsphericalSurface_DiffRays::createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	if (GEOM_NO_ERR != Geometry::createOptixInstance(context, geometrygroup, index, simParams, lambda) )
	{
		std::cout <<"error in AsphericalSurface_DiffRays.createOptixInstance(): Geometry.creatOptiXInstacne() returned an error at geometry: " << this->paramsPtr->geometryID << std::endl;
		return GEOM_ERR;
	}
    /* set geometry variables */
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(AsphericalSurface_DiffRays_ReducedParams), this->paramsPtr), context) )
		return GEOM_ERR;

	return GEOM_NO_ERR;
};

/**
 * \detail updateOptixInstance
 *
 * instead of destroying the OptiX instance of the surface we can change some of its parameters and update it and the materials attached to it
 *
 * \param[in] RTcontext &context, RTgeometrygroup &geometrygroup, int index, TraceMode mode, double lambda
 * 
 * \return geometryError
 * \sa 
 * \remarks maybe we should include means to update only those parameters that have changed instead of updating all parameters at once...
 * \author Mauch
 */
geometryError AsphericalSurface_DiffRays::updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{

	if (this->update)
	{
		if (GEOM_NO_ERR != Geometry::updateOptixInstance(context, geometrygroup, index, simParams, lambda))
		{
			std::cout <<"error in AsphericalSurface_DiffRays.updateOptixInstance(): materialList[i] returned an error at geometry: " << this->getParamsPtr()->geometryID << std::endl;
			return GEOM_ERR;
		}
		/* set geometry variables */
		if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceQueryVariable( instance, "params", &params ), context) )
			return GEOM_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(AsphericalSurface_DiffRays_ReducedParams), this->reducedParamsPtr), context) )
			return GEOM_ERR;

	}
	this->update=false;
	return GEOM_NO_ERR;
};