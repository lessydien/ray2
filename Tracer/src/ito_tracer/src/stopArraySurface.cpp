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

/**\file StopArraySurface.cpp
* \brief spherical surface
* 
*           
* \author Mauch
*/

#include "stopArraySurface.h"
#include <iostream>
#include "myUtil.h"

#include "Parser_XML.h"

/**
 * \detail getParamsPtr of Surface
 *
 * \param[in] void
 * 
 * \return Geometry_Params*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Geometry_Params* StopArraySurface::getParamsPtr(void)
{
  return this->paramsPtr;
};

/**
 * \detail setParamsPtr of Surface
 *
 * \param[in] StopArraySurface_Params *paramsInPtr
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError StopArraySurface::setParams(Geometry_Params *paramsIn)//PlaneSurface_Params *paramsIn)
{
	StopArraySurface_params *l_ptr=dynamic_cast<StopArraySurface_params*>(paramsIn);
	// if the incoming pointer has the correct type, copy the params
	if (l_ptr != NULL)
		*(this->paramsPtr)=*l_ptr;
	else
	{
		std::cout << "error in StopArraySurface.setParams(): paramsIn seems to not be of type PlaneSurface_Params" << std::endl;
		return GEOM_ERR;
	}

	this->update=true;
    return GEOM_NO_ERR;
};

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
double StopArraySurface::intersect(rayStruct *ray)
{
	double t=intersectRayStopArraySurface(ray->position,ray->direction,*(this->reducedParamsPtr));
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
geometryError StopArraySurface::hit(rayStruct &ray, double t)
{
	Mat_hitParams hitParams=calcHitParamsStopArraySurface(ray.position+t*ray.direction,*(this->reducedParamsPtr));
	hitParams.normal=this->paramsPtr->normal;
	int i;
	for (i=0;i<this->materialListLength;i++)
	{
		this->getMaterial(i)->hit(ray, hitParams, t, this->paramsPtr->geometryID);
	}
	
	return GEOM_NO_ERR;
};

///**
// * \detail intersect function for gaussian beam rays 
// *
// * \param[in] gaussBeamRayStruct ray
// * 
// * \return double t. That is the factor t for which r=ray.position+t*ray.direction is the intersection point of the ray with the surface
// * \sa 
// * \remarks This is a wrapper that repeatedly calls the inline function intersectRayAsphere for geometric rays that can be called from GPU as well
// * \author Mauch
// */
//gaussBeam_t StopArraySurface::intersect(gaussBeamRayStruct *ray)
//{
//	gaussBeam_t t;
//	t.t_baseRay=intersectRaySphere(ray->baseRay.position, ray->baseRay.direction, *(this->reducedParamsPtr));
//    // set aperture to infinity for waist rays and divergence rays
//	this->paramsPtr->apertureType=AT_INFTY;
//    t.t_waistRayX=intersectRaySphere(ray->waistRayX.position, ray->waistRayX.direction, *(this->reducedParamsPtr));
//    t.t_waistRayY=intersectRaySphere(ray->waistRayY.position, ray->waistRayY.direction, *(this->reducedParamsPtr));
//    t.t_divRayX=intersectRaySphere(ray->divRayX.position, ray->divRayX.direction, *(this->reducedParamsPtr));
//	t.t_divRayY=intersectRaySphere(ray->divRayY.position, ray->divRayY.direction, *(this->reducedParamsPtr));
//
//	return t;//intersectRayPlaneSurface(ray->position,ray->direction,*(this->paramsPtr));
//};

///**
// * \detail hit function for gaussian beam rays
// *
// * we calc the normal to the surface in the intersection point of the base ray. Then we call the hit function of the material that is attached to the surface. If one of the rays making up the gaussian beam missed the geometry we raise an error so far
// *
// * \param[in] gaussBeamRayStruct ray
// * \param[in] gaussBeam_t ray
// * 
// * \return geometryError
// * \sa 
// * \remarks 
// * \author Mauch
// */
//geometryError StopArraySurface::hit(gaussBeamRayStruct &ray, gaussBeam_t t)
//{
//	// check wether all the rays arrived at the same geometry the centre ray did
//	if ( (t.t_waistRayX!=0)&&(t.t_waistRayY!=0)&&(t.t_divRayX!=0)&&(t.t_divRayY != 0) ) 
//	{
//		// update the positions of the rays
//		ray.baseRay.position=ray.baseRay.position+ray.baseRay.direction*t.t_baseRay; //updating the ray
//		ray.baseRay.opl=ray.baseRay.opl+ray.baseRay.nImmersed*t.t_baseRay;
//		ray.baseRay.currentGeometryID=this->paramsPtr->geometryID;
//		ray.waistRayX.position=ray.waistRayX.position+ray.waistRayX.direction*t.t_waistRayX; //updating the ray
//		ray.waistRayX.currentGeometryID=this->paramsPtr->geometryID;
//		ray.waistRayY.position=ray.waistRayY.position+ray.waistRayY.direction*t.t_waistRayY; //updating the ray
//		ray.waistRayY.currentGeometryID=this->paramsPtr->geometryID;
//		ray.divRayX.position=ray.divRayX.position+ray.divRayX.direction*t.t_divRayX; //updating the ray
//		ray.divRayX.currentGeometryID=this->paramsPtr->geometryID;
//		ray.divRayY.position=ray.divRayY.position+ray.divRayY.direction*t.t_divRayY; //updating the ray
//		ray.divRayY.currentGeometryID=this->paramsPtr->geometryID;
//		// The normals at the intersection points are simply the normal to the surface
//		gaussBeam_geometricNormal normal;
//		normal.normal_baseRay=calcHitParamsSphere(ray.baseRay.position,*(this->reducedParamsPtr));
//		normal.normal_waistRayX=calcHitParamsSphere(ray.waistRayX.position,*(this->reducedParamsPtr));
//		normal.normal_waistRayY=calcHitParamsSphere(ray.waistRayY.position,*(this->reducedParamsPtr));
//		normal.normal_divRayX=calcHitParamsSphere(ray.divRayX.position,*(this->reducedParamsPtr));
//		normal.normal_divRayY=calcHitParamsSphere(ray.divRayY.position,*(this->reducedParamsPtr));
//		// call the hit function of the underlying material. 
//		this->getMaterial(0)->hit(ray,normal,this->paramsPtr->geometryID);
//		return GEOM_NO_ERR;
//	}
//	else
//	{
//		std::cout <<"error in StopArraySurface.hit(): rays of gaussian beamlet have inconsistent intersections at geometry:" << this->paramsPtr->geometryID << std::endl;
//		return GEOM_GBINCONSISTENTINTERSECTIONS_ERR;// terminate the ray with an error
//	}
//};

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
geometryError StopArraySurface::reduceParams(void)
{
	if ( (this->paramsPtr!=NULL) && (this->reducedParamsPtr!=NULL) )
	{
		this->reducedParamsPtr->geometryID=this->paramsPtr->geometryID;
		this->reducedParamsPtr->root=this->paramsPtr->root;
		this->reducedParamsPtr->normal=this->paramsPtr->normal;
		this->reducedParamsPtr->microStopRad=this->paramsPtr->microStopRad;
		this->reducedParamsPtr->microStopPitch=this->paramsPtr->microStopPitch;
		this->reducedParamsPtr->microStopType=this->paramsPtr->microStopType;
		this->reducedParamsPtr->apertureType=this->paramsPtr->apertureType;
		this->reducedParamsPtr->tilt=this->paramsPtr->tilt;
		this->reducedParamsPtr->apertureRadius=this->paramsPtr->apertureRadius;
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
geometryError StopArraySurface::createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	if (GEOM_NO_ERR != Geometry::createOptixInstance(context, geometrygroup, index, simParams, lambda) )
	{
		std::cout <<"error in StopArraySurface.createOptixInstance(): Geometry.creatOptiXInstacne() returned an error at geometry: " << this->paramsPtr->geometryID << std::endl;
		return GEOM_ERR;
	}

	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(StopArraySurface_ReducedParams), this->reducedParamsPtr), context) )
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
geometryError StopArraySurface::updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	if (this->update)
	{
		if (GEOM_NO_ERR != this->updateOptixInstance(context, geometrygroup, index, simParams, lambda) )
		{
			std::cout <<"error in StopArraySurface.updateOptixInstance(): Geometry.updateOptiXInstacne() returned an error at geometry: " << this->paramsPtr->geometryID << std::endl;
			return GEOM_ERR;
		}
		/* set geometry variables */
		if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceQueryVariable( instance, "params", &params ), context) )
			return GEOM_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(StopArraySurface_ReducedParams), this->reducedParamsPtr), context) )
			return GEOM_ERR;

	}
	this->update=false;
	return GEOM_NO_ERR;
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
geometryError StopArraySurface::parseXml(pugi::xml_node &geometry, TraceMode mode, vector<Geometry*> &geomVec)
{
	Parser_XML l_parser;

	// call base class function
	if (GEOM_NO_ERR != Geometry::parseXml(geometry, mode, geomVec))
	{
		std::cout << "error in StopArraySurface.parseXml(): Geometry.parseXml() returned an error" << std::endl;
		return GEOM_ERR;
	}

	this->paramsPtr->normal=make_double3(0,0,1);
	rotateRay(&(this->paramsPtr->normal), this->paramsPtr->tilt);

	double3 l_root;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "root.x", l_root.x)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "root.y", l_root.y)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "root.z", l_root.z)))
		return GEOM_ERR;
	this->paramsPtr->root=l_root;


	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "microStopRad.x", this->paramsPtr->microStopRad.x)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "microStopRad.y", this->paramsPtr->microStopRad.y)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "microStopPitch.x", this->paramsPtr->microStopPitch.x)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "microStopPitch.y", this->paramsPtr->microStopPitch.y)))
		return GEOM_ERR;

	const char* str=l_parser.attrValByName(geometry, "microStopType");
	if (str==NULL)
	{
		std::cout << "error in StopArraySurface.parseXml(): microStopType is not defined" << std::endl;
		return GEOM_ERR;
	}

	if (!strcmp(str, "MICRORECTANGULAR"))
		this->paramsPtr->microStopType=AT_RECT;
	else
	{
		if (!strcmp(str, "MICROELLIPTICAL"))
			this->paramsPtr->microStopType=AT_ELLIPT;
		else
		{
			std::cout << "error in StopArraySurface.parseXml(): unknown microStopType" << std::endl;
			return GEOM_ERR;
		}
	}

	geomVec.push_back(this);
	return GEOM_NO_ERR;
};
