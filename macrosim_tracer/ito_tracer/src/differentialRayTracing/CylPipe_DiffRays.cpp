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

/**\file CylPipe_DiffRays.cpp
* \brief cylinder surface
* 
*           
* \author Mauch
*/

#include "CylPipe_DiffRays.h"
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
double CylPipe_DiffRays::intersect(diffRayStruct *ray)
{
	double t=intersectRayCylPipe_DiffRays(ray->position,ray->direction,*(this->reducedParamsPtr));
	if (t>MAX_TOLERANCE)
	{
		return t;
	}
	else
	{
		return 0;
	}
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
geometryError CylPipe_DiffRays::hit(diffRayStruct &ray,double t)
{
	Mat_hitParams hitParams;
	hitParams=calcHitParamsCylPipe_DiffRays(ray.position+t*ray.direction, *(this->reducedParamsPtr));

	int i;
	for (i=0;i<this->materialListLength;i++)
	{
		this->getMaterial(i)->hit(ray,hitParams,t,this->paramsPtr->geometryID);
	}

	return GEOM_NO_ERR;
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
geometryError CylPipe_DiffRays::reduceParams(void)
{
	if ( (this->paramsPtr!=NULL) && (this->reducedParamsPtr!=NULL) )
	{
		this->reducedParamsPtr->geometryID=this->paramsPtr->geometryID;
		this->reducedParamsPtr->radius=this->paramsPtr->radius;
		this->reducedParamsPtr->root=this->paramsPtr->root;
		this->reducedParamsPtr->orientation=this->paramsPtr->orientation;
		this->reducedParamsPtr->thickness=this->paramsPtr->thickness;
//		this->reducedParamsPtr->rotNormal=this->paramsPtr->rotNormal;
		this->reducedParamsPtr->tilt=this->paramsPtr->tilt;
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
geometryError CylPipe_DiffRays::createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	if (GEOM_NO_ERR != Geometry::createOptixInstance(context, geometrygroup, index, simParams, lambda) )
	{
		std::cout <<"error in CylPipe_DiffRays.createOptixInstance(): Geometry.creatOptiXInstacne() returned an error at geometry: " << this->paramsPtr->geometryID << "...\n";
		return GEOM_ERR;
	}
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(CylPipe_DiffRays_ReducedParams), this->reducedParamsPtr), context) )
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
geometryError CylPipe_DiffRays::updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	if (this->update)
	{
		if (GEOM_NO_ERR != Geometry::updateOptixInstance(context, geometrygroup, index, simParams, lambda))
		{
			std::cout <<"error in CylPipe_DiffRays.updateOptixInstance(): materialList[i] returned an error at geometry: " << this->getParamsPtr()->geometryID << "...\n";
			return GEOM_ERR;
		}
		if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceQueryVariable( instance, "params", &params ), context) )
			return GEOM_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(CylPipe_DiffRays_ReducedParams), this->reducedParamsPtr), context) )
			return GEOM_ERR;
	}
	this->update=false;
	return GEOM_NO_ERR;
};
