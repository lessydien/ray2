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

/**\file VolumeScattererBox.cpp
* \brief plane surface
* 
*           
* \author Mauch
*/

#include "VolumeScattererBox.h"
#include "VolumeScattererBox_intersect.h"
#include <iostream>
#include "myUtil.h"
#include "Parser_XML.h"

/**
 * \detail getParamsPtr of Surface
 *
 * \param[in] void
 * 
 * \return VolumeScattererBox_Params*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Geometry_Params* VolumeScattererBox::getParamsPtr(void)
{
  return this->paramsPtr;
};

/**
 * \detail setParamsPtr of Surface
 *
 * \param[in] VolumeScattererBox_Params *paramsInPtr
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError VolumeScattererBox::setParams(Geometry_Params *paramsIn)//VolumeScattererBox_Params *paramsIn)
{
	VolumeScattererBox_Params *l_ptr=dynamic_cast<VolumeScattererBox_Params*>(paramsIn);
	// if the incoming pointer has the correct type, copy the params
	if (l_ptr != NULL)
		*(this->paramsPtr)=*l_ptr;
	else
	{
		std::cout << "error in VolumeScattererBox.setParams(): paramsIn seems to not be of type VolumeScattererBox_Params" << std::endl;
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
double VolumeScattererBox::intersect(rayStruct *ray)
{
	return intersectRayVolumeScattererBox(ray->position,ray->direction,*(this->reducedParamsPtr));
};

/**
 * \detail intersect function for differential rays
 *
 * \param[in] rayStruct ray
 * 
 * \return double t. That is the factor t for which r=ray.position+t*ray.direction is the intersection point of the ray with the surface
 * \sa 
 * \remarks This is a wrapper that calls the inline function intersectRayAsphere that can be called from GPU as well
 * \author Mauch
 */
//double VolumeScattererBox::intersect(diffRayStruct *ray)
//{
//	return intersectRayVolumeScattererBox(ray->position,ray->direction,*(this->reducedParamsPtr));
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
geometryError VolumeScattererBox::reduceParams(void)
{
	if ( (this->paramsPtr!=NULL) && (this->reducedParamsPtr!=NULL) )
	{
		this->reducedParamsPtr->geometryID=this->paramsPtr->geometryID;
		this->reducedParamsPtr->root=this->paramsPtr->root;
		this->reducedParamsPtr->apertureRadius=this->paramsPtr->apertureRadius;
		this->reducedParamsPtr->normal=this->paramsPtr->normal;
		//this->reducedParamsPtr->rotNormal=this->paramsPtr->rotNormal;
		this->reducedParamsPtr->tilt=this->paramsPtr->tilt;
		this->reducedParamsPtr->thickness=this->paramsPtr->thickness;
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
geometryError VolumeScattererBox::hit(rayStruct &ray, double t)
{
	Mat_hitParams hitParams;
	hitParams=calcHitParamsVolumeScattererBox(ray.position+t*ray.direction, *(this->reducedParamsPtr));
	for (int i=0;i<this->materialListLength;i++)
	{
		this->getMaterial(i)->hit(ray, hitParams, t, this->paramsPtr->geometryID);
	}
	
	return GEOM_NO_ERR;
 };

/**
 * \detail hit function function for differential rays
 *
 * we calc the normal to the surface in the intersection point. Then we call the hit function of the material that is attached to the surface
 *
 * \param[in] diffRayStruct ray
 * \param[in] double ray
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
//geometryError VolumeScattererBox::hit(diffRayStruct &ray, double t)
//{
//	Mat_DiffRays_hitParams hitParams;
//	// calc main directions at intersection point
//	hitParams.mainDirX=rotateRay(make_double3(1,0,0),this->paramsPtr->tilt);
//	hitParams.mainDirY=rotateRay(make_double3(0,1,0),this->paramsPtr->tilt);
//	// calc main radii of curvature at intersection. DOUBLE_MAX is the closest we can get to infinity here...
//	hitParams.mainRad=make_double2(DOUBLE_MAX,DOUBLE_MAX);
//	int i;
//	for (i=0;i<this->materialListLength;i++)
//	{
//		this->getMaterial(i)->hit(ray, hitParams, t, this->paramsPtr->geometryID);
//	}
//	
//	return GEOM_NO_ERR;
// };

/**
 * \detail intersect function for gaussian beam rays 
 *
 * \param[in] gaussBeamRayStruct ray
 * 
 * \return double t. That is the factor t for which r=ray.position+t*ray.direction is the intersection point of the ray with the surface
 * \sa 
 * \remarks This is a wrapper that repeatedly calls the inline function intersectRayAsphere for geometric rays that can be called from GPU as well
 * \author Mauch
 */
gaussBeam_t VolumeScattererBox::intersect(gaussBeamRayStruct *ray)
{
	gaussBeam_t t;
	t.t_baseRay=intersectRayVolumeScattererBox(ray->baseRay.position, ray->baseRay.direction, *(this->reducedParamsPtr));
    // set aperture to infinity for waist rays and divergence rays
	this->paramsPtr->apertureType=AT_INFTY;
    t.t_waistRayX=intersectRayVolumeScattererBox(ray->waistRayX.position, ray->waistRayX.direction, *(this->reducedParamsPtr));
    t.t_waistRayY=intersectRayVolumeScattererBox(ray->waistRayY.position, ray->waistRayY.direction, *(this->reducedParamsPtr));
    t.t_divRayX=intersectRayVolumeScattererBox(ray->divRayX.position, ray->divRayX.direction, *(this->reducedParamsPtr));
	t.t_divRayY=intersectRayVolumeScattererBox(ray->divRayY.position, ray->divRayY.direction, *(this->reducedParamsPtr));

	return t;//intersectRayVolumeScattererBox(ray->position,ray->direction,*(this->paramsPtr));
};

/**
 * \detail hit function for gaussian beam rays
 *
 * we calc the normal to the surface in the intersection point of the base ray. Then we call the hit function of the material that is attached to the surface. If one of the rays making up the gaussian beam missed the geometry we raise an error so far
 *
 * \param[in] gaussBeamRayStruct ray
 * \param[in] gaussBeam_t ray
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError VolumeScattererBox::hit(gaussBeamRayStruct &ray, gaussBeam_t t)
{
	// check wether all the rays hit this geometry
	if ( (t.t_waistRayX!=0)&&(t.t_waistRayY!=0)&&(t.t_divRayX!=0)&&(t.t_divRayY != 0) ) 
	{
		// update the positions of the rays
		ray.baseRay.position=ray.baseRay.position+ray.baseRay.direction*t.t_baseRay; //updating the ray
		ray.baseRay.opl=ray.baseRay.opl+ray.baseRay.nImmersed*t.t_baseRay;
		ray.baseRay.currentGeometryID=this->reducedParamsPtr->geometryID;
		ray.waistRayX.position=ray.waistRayX.position+ray.waistRayX.direction*t.t_waistRayX; //updating the ray
		ray.waistRayX.currentGeometryID=this->reducedParamsPtr->geometryID;
		ray.waistRayY.position=ray.waistRayY.position+ray.waistRayY.direction*t.t_waistRayY; //updating the ray
		ray.waistRayY.currentGeometryID=this->reducedParamsPtr->geometryID;
		ray.divRayX.position=ray.divRayX.position+ray.divRayX.direction*t.t_divRayX; //updating the ray
		ray.divRayX.currentGeometryID=this->reducedParamsPtr->geometryID;
		ray.divRayY.position=ray.divRayY.position+ray.divRayY.direction*t.t_divRayY; //updating the ray
		ray.divRayY.currentGeometryID=this->reducedParamsPtr->geometryID;
		// The normals at the intersection points are simply the normal to the surface
		gaussBeam_geometricNormal normal;
		normal.normal_baseRay=this->reducedParamsPtr->normal;
		normal.normal_waistRayX=this->reducedParamsPtr->normal;
		normal.normal_waistRayY=this->reducedParamsPtr->normal;
		normal.normal_divRayX=this->reducedParamsPtr->normal;
		normal.normal_divRayY=this->reducedParamsPtr->normal;
		// call the hit function of the underlying material. 
		this->getMaterial(0)->hit(ray,normal,this->paramsPtr->geometryID);
		return GEOM_NO_ERR;
	}
	else
	{
		std::cout <<"error in VolumeScattererBox.hit(): rays of gaussian beamlet have inconsistent intersections on geometry:" << this->paramsPtr->geometryID << std::endl;
		return GEOM_GBINCONSISTENTINTERSECTIONS_ERR;// terminate the ray with an error
	}
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
geometryError VolumeScattererBox::createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	if (GEOM_NO_ERR != Geometry::createOptixInstance(context, geometrygroup, index, simParams, lambda) )
	{
		std::cout <<"error in VolumeScattererBox.createOptixInstance(): Geometry.creatOptiXInstacne() returned an error at geometry: " << this->paramsPtr->geometryID << std::endl;
		return GEOM_ERR;
	}

	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(VolumeScattererBox_ReducedParams), this->reducedParamsPtr), context) )
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
geometryError VolumeScattererBox::updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	if (this->update)
	{
		if (GEOM_NO_ERR != this->updateOptixInstance(context, geometrygroup, index, simParams, lambda) )
		{
			std::cout <<"error in VolumeScattererBox.updateOptixInstance(): Geometry.updateOptiXInstacne() returned an error at geometry: " << this->paramsPtr->geometryID << std::endl;
			return GEOM_ERR;
		}
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceQueryVariable( instance, "params", &params ), context) )
			return GEOM_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(VolumeScattererBox_ReducedParams), this->reducedParamsPtr), context) )
			return GEOM_ERR;

	}
	return GEOM_NO_ERR;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] GeometryParseParamStruct &parseResults_Geom
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError VolumeScattererBox::processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID)
{
	this->paramsPtr->normal=parseResults_Geom.normal;
	this->paramsPtr->root=parseResults_Geom.root;
	this->paramsPtr->tilt=parseResults_Geom.tilt;
	this->paramsPtr->apertureType=parseResults_Geom.aperture;
	this->paramsPtr->apertureRadius=parseResults_Geom.apertureHalfWidth1;
//	this->paramsPtr->rotNormal=parseResults_Geom.rotNormal1;
	this->paramsPtr->geometryID=geomID;
	return GEOM_NO_ERR;
}

geometryError VolumeScattererBox::parseXml(pugi::xml_node &geometry, TraceMode l_mode, vector<Geometry*> &geomVec)
{
	// parse base class
	if (GEOM_NO_ERR!=Geometry::parseXml(geometry,l_mode, geomVec))
	{
		std::cout << "error in VolumeScattererBox.parseXml(): Geometry.parseXml() returned an error." << std::endl;
		return GEOM_ERR;
	}
	double3 l_vec=make_double3(0,0,1);
	rotateRay(&l_vec,this->getParamsPtr()->tilt);
	this->paramsPtr->normal=l_vec;
	Parser_XML l_parser;
	if (!l_parser.attrByNameToDouble(geometry, "root.x", this->paramsPtr->root.x))
	{
		std::cout << "error in VolumeScattererBox.parseXml(): root.x is not defined" << std::endl;
		return GEOM_ERR;
	}
	if (!l_parser.attrByNameToDouble(geometry, "root.y", this->paramsPtr->root.y))
	{
		std::cout << "error in VolumeScattererBox.parseXml(): root.y is not defined" << std::endl;
		return GEOM_ERR;
	}
	if (!l_parser.attrByNameToDouble(geometry, "root.z", this->paramsPtr->root.z))
	{
		std::cout << "error in VolumeScattererBox.parseXml(): root.z is not defined" << std::endl;
		return GEOM_ERR;
	}
	if (!l_parser.attrByNameToDouble(geometry, "thickness", this->paramsPtr->thickness))
	{
		std::cout << "error in VolumeScattererBox.parseXml(): thickness is not defined" << std::endl;
		return GEOM_ERR;
	}
	geomVec.push_back(this);
	return GEOM_NO_ERR;

}