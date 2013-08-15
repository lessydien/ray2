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

/**\file Pupil_RayAim_FarField.cpp
* \brief plane surface
* 
*           
* \author Mauch
*/

#include "Pupil_RayAim_FarField.h"
#include <iostream>
#include "myUtil.h"

/**
 * \detail getParamsPtr of Surface
 *
 * \param[in] void
 * 
 * \return Pupil_RayAim_FarField_Params*
 * \sa 
 * \remarks 
 * \author Mauch
 */
pupilParams* Pupil_RayAim_FarField::getFullParamsPtr(void)
{
  return this->fullParamsPtr;
};

/**
 * \detail setParamsPtr of Surface
 *
 * \param[in] Pupil_RayAim_FarField_Params *paramsInPtr
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Pupil_RayAim_FarField::setFullParamsPtr(pupilParams *paramsIn)//Pupil_RayAim_FarField_Params *paramsIn)
{
	//Pupil_RayAim_FarField_Params *l_ptr=dynamic_cast<Pupil_RayAim_FarField_Params*>(paramsIn);
	//// if the incoming pointer has the correct type, copy the params
	//if (l_ptr != NULL)
	//	*(this->paramsPtr)=*l_ptr;
	//else
	//{
	//	std::cout << "error in Pupil_RayAim_FarField.setParams(): paramsIn seems to not be of type Pupil_RayAim_FarField_Params" << std::endl;
	//	return GEOM_ERR;
	//}
	//this->update=true;
};

/**
 * \detail aim function for geometric rays
 *
 * \param[in] rayStruct ray
 * 
 * \return double t. That is the factor t for which r=ray.position+t*ray.direction is the intersection point of the ray with the surface
 * \sa 
 * \remarks This is a wrapper that calls the inline function intersectRayAsphere that can be called from GPU as well
 * \author Mauch
 */
void Pupil_RayAim_FarField::aim(rayStruct *ray, unsigned long long iX, unsigned long long iY)
{
	aimRayAimFarField(ray->position,ray->direction,*(this->redParamsPtr), iX, iY);
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
//double Pupil_RayAim_FarField::intersect(diffRayStruct *ray)
//{
//	return intersectRayPupil_RayAim_FarField(ray->position,ray->direction,*(this->reducedParamsPtr));
//};

/**
 * \detail reduceParams
 *
 * \param[in] void
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
PupilError Pupil_RayAim_FarField::reduceParams(void)
{
	//if ( (this->paramsPtr!=NULL) && (this->reducedParamsPtr!=NULL) )
	//{
	//	this->reducedParamsPtr->geometryID=this->paramsPtr->geometryID;
	//	this->reducedParamsPtr->root=this->paramsPtr->root;
	//	this->reducedParamsPtr->apertureRadius=this->paramsPtr->apertureRadius;
	//	this->reducedParamsPtr->normal=this->paramsPtr->normal;
	//	this->reducedParamsPtr->apertureType=this->paramsPtr->apertureType;
	//	//this->reducedParamsPtr->rotNormal=this->paramsPtr->rotNormal;
	//	this->reducedParamsPtr->tilt=this->paramsPtr->tilt;
	//}
	return PUP_NO_ERR;
};


/**
 * \detail createCPUSimInstance 
 *
 * \param[in] double lambda, simMode mode
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
PupilError Pupil_RayAim_FarField::createCPUSimInstance(double lambda)
{
	//this->reduceParams();
	///* check wether any material is present */
	//if (this->materialListLength==0)
	//{
	//	std::cout << "error in Pupil_RayAim_FarField.createCPUInstance(): no material attached to surface at geometry:" << this->paramsPtr->geometryID << std::endl;
	//	return GEOM_NOMATERIAL_ERR;
	//}
	//this->mode=mode;
	///* create instances of material */
	//int i;
	//for (i=0; i<materialListLength; i++)
	//{
	//	if ( MAT_NO_ERR != this->materialList[i]->createCPUSimInstance(lambda) )
	//	{
	//		std::cout << "error in Pupil_RayAim_FarField.createCPUInstance(): material.createCPUSimInstance() returned an error at geometry:" << this->paramsPtr->geometryID << std::endl;
	//		return GEOM_ERR;
	//	}
	//}
	return PUP_NO_ERR;
};

/**
 * \detail updateCPUSimInstance 
 *
 * \param[in] double lambda, simMode mode
 * 
 * \return PupilError
 * \sa 
 * \remarks 
 * \author Mauch
 */
PupilError Pupil_RayAim_FarField::updateCPUSimInstance(double lambda)
{
	///* check wether any material is present */
	//if (this->materialListLength==0)
	//{
	//	std::cout << "error in Geometry.updateCPUSimInstance(): no material attached to surface at geometry:" << this->paramsPtr->geometryID << std::endl;
	//	return GEOM_NOMATERIAL_ERR;
	//}
	//this->mode=mode;
	///* create instances of material */
	//int i;
	//for (i=0; i<materialListLength; i++)
	//{
	//	if ( MAT_NO_ERR != this->materialList[i]->updateCPUSimInstance(lambda) )
	//	{
	//		std::cout << "error in Geometry.updateCPUSimInstance(): material.createCPUSimInstance() returned an error at geometry:" << this->paramsPtr->geometryID << std::endl;
	//		return GEOM_ERR;
	//	}
	//}
	//if (this->update)
	//	reduceParams();
	return PUP_NO_ERR;
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
PupilError Pupil_RayAim_FarField::processParseResults(PupilParseParamStruct &parseResults_Pupil)
{
//	this->paramsPtr->normal=parseResults_Geom.normal;
//	this->paramsPtr->root=parseResults_Geom.root;
//	this->paramsPtr->tilt=parseResults_Geom.tilt;
//	this->paramsPtr->apertureType=parseResults_Geom.aperture;
//	this->paramsPtr->apertureRadius=parseResults_Geom.apertureHalfWidth1;
////	this->paramsPtr->rotNormal=parseResults_Geom.rotNormal1;
//	this->paramsPtr->geometryID=geomID;
	return PUP_NO_ERR;
}