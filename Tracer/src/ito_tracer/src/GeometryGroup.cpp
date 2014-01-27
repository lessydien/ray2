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

/**\file GeometryGroup.cpp
* \brief container that can hold multiple geometries to form a hierarchical scene graph of the geometries
* 
*           
* \author Mauch
*/

#include "GeometryGroup.h"
#include "Geometry.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "Parser_XML.h"

/**
 * \detail trace
 *
 * traces the ray through the geometry group
 *
 * \param[in] rayStruct &ray
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryGroupError  GeometryGroup::trace(rayStruct &ray)
{
	int indexToTrace;
	double minDist=99999999999999999;//infinity
	double tempDist;
	if ( (this->mode.traceMode==TRACE_SEQ) )
	{
		Geometry* l_geometry=this->getGeometry(ray.currentGeometryID);
		if (l_geometry==NULL)
		{
			ray.running=false; // if there is no geometry left, we stop the ray
		}
		else
		{
			tempDist=l_geometry->intersect(&ray);
			if (tempDist>EPSILON)
				// if we intersect the geometry, hit it
				this->getGeometry(ray.currentGeometryID)->hit(ray,tempDist);
			else
				// if we miss the geometry, stop the ray
				ray.running=false;
		}
	}
	else
	{

		if (this->getGeometryListLength()==0)
		{
			ray.running=false; // stop ray
			return GEOMGROUP_NOGEOM_ERR;
		}
		//if there is only one geometry in the list -> trace this one.
		if (this->getGeometryListLength()<2)
		{
			indexToTrace=0;
			minDist=tempDist=this->getGeometry(0)->intersect(&ray);
			if (minDist<=EPSILON) 
			{
				ray.running=false;//stop ray
				return GEOMGROUP_NO_ERR; // we have no intersection here. So we return
			}
		}
		else //if there are 2 or more geometries
		{	
			//start with first geometry's data
			indexToTrace=-1;
			tempDist=this->getGeometry(0)->intersect(&ray);
			if (tempDist>EPSILON)
			{
				minDist=tempDist;
				indexToTrace=0;
			}
			//now check if there is a closer one
			for(unsigned int i=1;i<this->geometryListLength;i++)
			{
				tempDist=this->getGeometry(i)->intersect(&ray);
				
				if ((tempDist<minDist)&&(tempDist>0.000001))
				{
					minDist=tempDist;
					indexToTrace=i;
				}
				
			}
			if (minDist<EPSILON)
				std::cout << "distance to small" << std::endl;
			if ((minDist<0) || (indexToTrace==-1))
			{
				ray.running=false; //stop ray
				return GEOMGROUP_NO_ERR; 
			}
		}

		this->getGeometry(indexToTrace)->hit(ray,minDist);
	}
	
	return GEOMGROUP_NO_ERR;
};

/**
 * \detail trace
 *
 * traces the differential ray through the geometry group
 *
 * \param[in] diffRayStruct &ray
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryGroupError  GeometryGroup::trace(diffRayStruct &ray)
{
	int indexToTrace;
	double minDist=99999999999999999;//infinity
	double tempDist;
	if ( (this->mode.traceMode==TRACE_SEQ) )
	{
		tempDist=this->getGeometry(ray.currentGeometryID)->intersect(&ray);
		if (tempDist>0.000001)
			// if we intersect the geometry, hit it
			this->getGeometry(ray.currentGeometryID)->hit(ray,tempDist);
		else
			// if we miss the geometry, stop the ray
			ray.running=false;
	}
	else
	{

		if (this->getGeometryListLength()==0)
		{
			ray.running=false; // stop ray
			return GEOMGROUP_NOGEOM_ERR;
		}
		//if there is only one geometry in the list -> trace this one.
		if (this->getGeometryListLength()<2)
		{
			indexToTrace=0;
			minDist=tempDist=this->getGeometry(0)->intersect(&ray);
			if (minDist<=0) 
			{
				ray.running=false;//stop ray
				return GEOMGROUP_NO_ERR; // we have no intersection here. So we return
			}
		}
		else //if there are 2 or more geometries
		{	
			//start with first geometry's data
			indexToTrace=-1;
			tempDist=this->getGeometry(0)->intersect(&ray);
			if (tempDist>EPSILON)
			{
				minDist=tempDist;
				indexToTrace=0;
			}
			//now check if there is a closer one
			for(unsigned int i=1;i<this->geometryListLength;i++)
			{
				tempDist=this->getGeometry(i)->intersect(&ray);
				
				if ((tempDist<minDist)&&(tempDist>EPSILON))
				{
					minDist=tempDist;
					indexToTrace=i;
				}
				//if(minDist<0) minDist=tempDist; // do we need this !?!
				
			}
			if ((minDist<0) || (indexToTrace==-1))
			{
				ray.running=false; //stop ray
				return GEOMGROUP_NO_ERR; 
			}
		}

		this->getGeometry(indexToTrace)->hit(ray,minDist);
	}
	
	//return GEOMGROUP_NO_ERR;
};

/**
 * \detail trace
 *
 * traces the gaussian beam ray through the geometry group
 *
 * \param[in] gaussBeamRayStruct &ray
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryGroupError  GeometryGroup::trace(gaussBeamRayStruct &ray)
{
	int indexToTrace;
	double minDist;
	minDist=0;
	double tempDist;
	indexToTrace=0;
	gaussBeam_t t;
	if (this->mode.traceMode==TRACE_SEQ)
	{
		t=this->getGeometry(ray.baseRay.currentGeometryID+1)->intersect(&ray);
		if (t.t_baseRay>=0)
			this->getGeometry(ray.baseRay.currentGeometryID+1)->hit(ray,t);
	}
	else
	{
		// maybe we should check wether there is at least one geometry ???
		// now check for the closest geometry for the centre ray
		for(unsigned int i=0;i<this->geometryListLength;i++)
		{
			// hit the current geometry
			tempDist=this->getGeometry(i)->intersect(&(ray.baseRay));
			// the centre ray is decisive
			if ( ((tempDist<minDist)&&(tempDist>0.001)) || ((minDist==0)&&(tempDist>0.001)) )
			{
				minDist=tempDist;
				indexToTrace=i;
			}
		}
		// if minDist is still zero we hit no geometry and return without any further action
		if (minDist==0)
		{
			return GEOMGROUP_NO_ERR;
		}
		// now, intersect the closest geometry with all the rays
		t=this->getGeometry(indexToTrace)->intersect(&ray);
		// hit the closest geometry
		this->getGeometry(indexToTrace)->hit(ray,t);
	}
	
	return GEOMGROUP_NO_ERR;
};

/**
 * \detail getGeometryListLength
 *
 * \param[in] void
 * 
 * \return int
 * \sa 
 * \remarks 
 * \author Mauch
 */
int GeometryGroup::getGeometryListLength()
{
	return geometryListLength;
};

/**
 * \detail setGeometry
 *
 * \param[in] Geometry* oGeometryPtr, unsigned int index
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryGroupError GeometryGroup::setGeometry(Geometry* oGeometryPtr, unsigned int index)
{
	if (index >= geometryListLength)
	{
		std::cout << "error in GeometryGroup.setGeometry(): index exceeds size of geometryList" << std::endl;
		return GEOMGROUP_ERR;
	}
	geometryList[index]=oGeometryPtr;

	return GEOMGROUP_NO_ERR;
};

/**
 * \detail getGeometry
 *
 * \param[in] unsigned int index
 * 
 * \return Geometry*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Geometry* GeometryGroup::getGeometry(unsigned int index)
{
	if (index<this->geometryListLength)
		return geometryList[index];	
	else
		return NULL;
};

/**
 * \detail findClosestGeometry
 *
 * \param[in] void
 * 
 * \return GEOMGROUP_NO_ERR
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
geometryGroupError findClosestGeometry( void )
{
	return GEOMGROUP_NO_ERR;
};

/**
 * \detail createOptixInstance
 *
 * \param[in] RTcontext &context, RTgroup &OptiX_group, unsigned int index, TraceMode mode, double lambda
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryGroupError GeometryGroup::createOptixInstance(RTcontext &context, RTgroup &OptiX_group, unsigned int index, SimParams simParams, double lambda)
{

	/* check wether any geometry is present */
	if (this->geometryListLength==0)
	{
		std::cout << "error in GeometryGroup.createOtpixInstance(): no geometries attached to group" << std::endl;
		return GEOMGROUP_NOGEOM_ERR;
	}
	// if we plan to do nonsequential simulation we put all the geometries into one OptiX-OptiX_group
	// if we plan to do sequential raytracing we end up in another createOptixInstance...
	if (1)//mode.traceMode==SIM_GEOMRAYS_NONSEQ)
	{
		/* create optix geometry_group to hold instance transform */
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryGroupCreate( context, &OptiX_geometrygroup ), context ))
			return GEOMGROUP_ERR;
		/* set number of geometries inside group */
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetChildCount( OptiX_geometrygroup, geometryListLength ), context ))
			return GEOMGROUP_ERR;

		/* create acceleration object for OptiX_geometrygroup and specify some build hints*/
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationCreate(context,&acceleration), context ))
			return GEOMGROUP_ERR;
		const char* accel;
		Parser_XML l_parser;
		accel=this->accelTypeToAscii(this->paramsPtr->acceleration);
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationSetBuilder(acceleration,accel), context ))
			return GEOMGROUP_ERR;
		const char* traverser;
		traverser=this->accelTypeToTraverserAscii(this->paramsPtr->acceleration);
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationSetTraverser(acceleration,traverser), context ))
			return GEOMGROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetAcceleration( OptiX_geometrygroup, acceleration), context ))
			return GEOMGROUP_ERR;

		/* mark acceleration as dirty */
//		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationMarkDirty( acceleration ), context ))
//			return GEOMGROUP_ERR;

		/* add a transform node */
		//m[ 0] = 1.0f;  m[ 1] = 0.0f;  m[ 2] = 0.0f;  m[ 3] = 2.0f;
		//m[ 4] = 0.0f;  m[ 5] = 1.0f;  m[ 6] = 0.0f;  m[ 7] = 0.0f;
		//m[ 8] = 0.0f;  m[ 9] = 0.0f;  m[10] = 1.0f;  m[11] = 0.0f;
		//m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = 1.0f;

		//if (!RT_CHECK_ERROR_NOEXIT( rtTransformCreate( context, &transforms ), context ))
		//	return GEOMGROUP_ERR;
		//m[11] = 0.0f;//i*1.0f - (NUM_BOXES-1)*0.5f;
		//m[3] = 2.0f;//i*0.1f;
		//if (!RT_CHECK_ERROR_NOEXIT( rtTransformSetMatrix( transforms, 0, m, 0 ), context ))
		//	return GEOMGROUP_ERR;
		//if (!RT_CHECK_ERROR_NOEXIT( rtTransformSetChild( transforms, OptiX_geometrygroup ), context ))
		//	return GEOMGROUP_ERR;


		/* create Instances of geometries */
		unsigned int i;
		for (i=0; i<geometryListLength; i++)
		{
			if (GEOM_NO_ERR != geometryList[i]->createOptixInstance(context, OptiX_geometrygroup, i, simParams, lambda))
			{
				std::cout << "error in GeometryGroup.createOptixInstance: geometryList[i]->createOptiXInstance returned an error at index:" << i << std::endl;
				return GEOMGROUP_ERR;
			}
		}

		/* connect OptiX_geometrygroup to OptiX_group */
		if (!RT_CHECK_ERROR_NOEXIT( rtGroupSetChild( OptiX_group, 0, OptiX_geometrygroup ), context ))
			return GEOMGROUP_ERR;

		return GEOMGROUP_NO_ERR;
	}
};

/**
 * \detail createOptixInstance
 *
 * \param[in] RTcontext &context, RTselector &selector, unsigned int index, TraceMode mode, double lambda
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks for sequential scenes
 * \author Mauch
 */
geometryGroupError GeometryGroup::createOptixInstance(RTcontext &context, RTselector &selector, unsigned int index, SimParams simParams, double lambda)
{
	float				m[16]; // transformation matrix in homogenous coordinates

	/* check wether any geometry is present */
	if (this->geometryListLength==0)
	{
		std::cout << "error in GeometryGroup.createOtpixInstance(): no geometries attached to group" << std::endl;
		return GEOMGROUP_NOGEOM_ERR;
	}

	/* create Instances of geometryGroups */
	unsigned int i;
	for (i=0; i<geometryListLength; i++)
	{
		/* create geometry OptiX_group OptiX_group to hold instance transform */
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryGroupCreate( context, &OptiX_geometrygroup ), context ))
			return GEOMGROUP_ERR;
		/* set number of geometries inside goup */
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetChildCount( OptiX_geometrygroup, 1 ), context ))
			return GEOMGROUP_ERR;

		/* create acceleration object for OptiX_geometrygroup and specify some build hints*/
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationCreate(context,&acceleration), context ))
			return GEOMGROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationSetBuilder(acceleration,"NoAccel"), context ))
			return GEOMGROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationSetTraverser(acceleration,"NoAccel"), context ))
			return GEOMGROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetAcceleration( OptiX_geometrygroup, acceleration), context ))
			return GEOMGROUP_ERR;

		/* mark acceleration as dirty */
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationMarkDirty( acceleration ), context ))
			return GEOMGROUP_ERR;

		/* add a transform node */
		m[ 0] = 1.0f;  m[ 1] = 0.0f;  m[ 2] = 0.0f;  m[ 3] = 0.0f;
		m[ 4] = 0.0f;  m[ 5] = 1.0f;  m[ 6] = 0.0f;  m[ 7] = 0.0f;
		m[ 8] = 0.0f;  m[ 9] = 0.0f;  m[10] = 1.0f;  m[11] = 0.0f;
		m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = 1.0f;

		if (!RT_CHECK_ERROR_NOEXIT( rtTransformCreate( context, &transforms ), context ))
			return GEOMGROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtTransformSetChild( transforms, OptiX_geometrygroup ), context ))
			return GEOMGROUP_ERR;
		m[11] = 0.0f;//i*1.0f - (NUM_BOXES-1)*0.5f;
		m[7] = 0.0f;//i*0.1f;
		if (!RT_CHECK_ERROR_NOEXIT( rtTransformSetMatrix( transforms, 0, m, 0 ), context ))
			return GEOMGROUP_ERR;

		/* connect OptiX_geometrygroup to OptiX_group */
		if (!RT_CHECK_ERROR_NOEXIT( rtSelectorSetChild( selector, i, OptiX_geometrygroup ), context ))
			return GEOMGROUP_ERR;
		// create OptiX geometry inside OptiX_geometrygroup at index 0
		if ( GEOM_NO_ERR != geometryList[i]->createOptixInstance(context, OptiX_geometrygroup, 0, simParams, lambda) )
		{
			std::cout << "error in GeometryGroup.createOptixInstance: geometryList[i]->createOptixInstance returned an error at index:" << i << std::endl;
			return GEOMGROUP_ERR;
		}
	}
	return GEOMGROUP_NO_ERR;
};

/**
 * \detail updateOptixInstance
 *
 * \param[in] RTcontext &context, RTgroup &OptiX_group, unsigned int index, TraceMode mode, double lambda
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryGroupError GeometryGroup::updateOptixInstance(RTcontext &context, RTgroup &OptiX_group, unsigned int index, SimParams simParams, double lambda)
{
//	RTacceleration		acceleration;
//    RTtransform			transforms;
//	float				m[16]; // transformation matrix in homogenous coordinates

	/* check wether any geometry is present */
	if (this->geometryListLength==0)
	{
		std::cout << "error in GeometryGroup.updateOtpixInstance(): no geometries attached to group" << std::endl;
		return GEOMGROUP_NOGEOM_ERR;
	}
	// if we plan to do nonsequential simulation we put all the geometries into one OptiX-OptiX_group
	if (1)//mode.traceMode==SIM_GEOMRAYS_NONSEQ)
	{
		/* create geometry OptiX_group OptiX_group to hold instance transform */
//		RT_CHECK_ERROR_NOEXIT( rtGeometryGroupCreate( context, &OptiX_geometrygroup ) );
		/* set number of geometries inside goup */
//		RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetChildCount( OptiX_geometrygroup, geometryListLength ) );

		/* create acceleration object for OptiX_geometrygroup and specify some build hints*/
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationCreate(context,&acceleration) );
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationSetBuilder(acceleration,"NoAccel") );
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationSetTraverser(acceleration,"NoAccel") );
//		RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetAcceleration( OptiX_geometrygroup, acceleration) );

		/* mark acceleration as dirty */
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationMarkDirty( acceleration ) );

		/* add a transform node */
//		m[ 0] = 1.0f;  m[ 1] = 0.0f;  m[ 2] = 0.0f;  m[ 3] = 0.0f;
//		m[ 4] = 0.0f;  m[ 5] = 1.0f;  m[ 6] = 0.0f;  m[ 7] = 0.0f;
//		m[ 8] = 0.0f;  m[ 9] = 0.0f;  m[10] = 1.0f;  m[11] = 0.0f;
//		m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = 1.0f;

//		RT_CHECK_ERROR_NOEXIT( rtTransformCreate( context, &transforms ) );
//		RT_CHECK_ERROR_NOEXIT( rtTransformSetChild( transforms, OptiX_geometrygroup ) );
//		m[11] = 0.0f;//i*1.0f - (NUM_BOXES-1)*0.5f;
//		m[7] = 0.0f;//i*0.1f;
//		RT_CHECK_ERROR_NOEXIT( rtTransformSetMatrix( transforms, 0, m, 0 ) );

		/* update OptiX OptiX_group variables */

		/* create Instances of geometries */
		unsigned int i;
		for (i=0; i<geometryListLength; i++)
		{
			this->geometryList[i]->updateOptixInstance(context, OptiX_geometrygroup, i, simParams, lambda);
		}

		/* connect OptiX_geometrygroup to OptiX_group */
//		RT_CHECK_ERROR_NOEXIT( rtGroupSetChild( OptiX_group, index, OptiX_geometrygroup ) );
		return GEOMGROUP_NO_ERR;
	}
};

/**
 * \detail updateOptixInstance
 *
 * \param[in] RTcontext &context, RTselector &selector, unsigned int index, TraceMode mode, double lambda
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks for sequential scenes
 * \author Mauch
 */
geometryGroupError GeometryGroup::updateOptixInstance(RTcontext &context, RTselector &selector, unsigned int index, SimParams simParams, double lambda)
{
//	RTacceleration		acceleration;
//    RTtransform			transforms;
//	float				m[16]; // transformation matrix in homogenous coordinates

	/* check wether any geometry is present */
	if (this->geometryListLength==0)
	{
		std::cout << "error in GeometryGroup.createOtpixInstance(): no geometries attached to group" << std::endl;
		return GEOMGROUP_NOGEOM_ERR;
	}

	/* create Instances of geometryGroups */
	unsigned int i;
	for (i=0; i<geometryListLength; i++)
	{
		/* create geometry OptiX_group OptiX_group to hold instance transform */
//		RT_CHECK_ERROR_NOEXIT( rtGeometryGroupCreate( context, &OptiX_geometrygroup ) );
		/* set number of geometries inside goup */
//		RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetChildCount( OptiX_geometrygroup, 1 ) );

		/* create acceleration object for OptiX_geometrygroup and specify some build hints*/
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationCreate(context,&acceleration) );
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationSetBuilder(acceleration,"NoAccel") );
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationSetTraverser(acceleration,"NoAccel") );
//		RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetAcceleration( OptiX_geometrygroup, acceleration) );

		/* mark acceleration as dirty */
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationMarkDirty( acceleration ) );

		/* add a transform node */
//		m[ 0] = 1.0f;  m[ 1] = 0.0f;  m[ 2] = 0.0f;  m[ 3] = 0.0f;
//		m[ 4] = 0.0f;  m[ 5] = 1.0f;  m[ 6] = 0.0f;  m[ 7] = 0.0f;
//		m[ 8] = 0.0f;  m[ 9] = 0.0f;  m[10] = 1.0f;  m[11] = 0.0f;
//		m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = 1.0f;

//		RT_CHECK_ERROR_NOEXIT( rtTransformCreate( context, &transforms ) );
//		RT_CHECK_ERROR_NOEXIT( rtTransformSetChild( transforms, OptiX_geometrygroup ) );
//		m[11] = 0.0f;//i*1.0f - (NUM_BOXES-1)*0.5f;
//		m[7] = 0.0f;//i*0.1f;
//		RT_CHECK_ERROR_NOEXIT( rtTransformSetMatrix( transforms, 0, m, 0 ) );

		/* connect OptiX_geometrygroup to OptiX_group */
//		RT_CHECK_ERROR_NOEXIT( rtSelectorSetChild( selector, i, OptiX_geometrygroup ) );
		// create OptiX geometry inside OptiX_geometrygroup at index 0
		geometryList[i]->updateOptixInstance(context, OptiX_geometrygroup, 0, simParams, lambda);
	}
	return GEOMGROUP_NO_ERR;
};

/**
 * \detail createCPUSimInstance
 *
 * \param[in] double lambda, TraceMode mode 
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks
 * \author Mauch
 */
geometryGroupError GeometryGroup::createCPUSimInstance(double lambda, SimParams simParams )
{
	/* check wether any geometry is present */
	if (this->geometryListLength==0)
	{
		std::cout << "error in GeometryGroup.createCPUSimInstance(): no geometries attached to group" << std::endl;
		return GEOMGROUP_NOGEOM_ERR;
	}
	this->mode=simParams;
	/* create Instances of geometryGroups */
	unsigned int i;
	for (i=0; i<geometryListLength; i++)
	{
		if ( GEOM_NO_ERR != geometryList[i]->createCPUSimInstance(lambda, simParams) )
		{
			std::cout << "error in GeometryGroup.createCPUSimInstance(): geometry.createCPUSimInstance() returned an error at index:" << i << std::endl;
			return GEOMGROUP_ERR;
		}
	}
	return GEOMGROUP_NO_ERR;
};

/**
 * \detail updateCPUSimInstance
 *
 * \param[in] double lambda, TraceMode mode 
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks
 * \author Mauch
 */
geometryGroupError GeometryGroup::updateCPUSimInstance(double lambda, SimParams simParams )
{
	/* check wether any geometry is present */
	if (this->geometryListLength==0)
	{
		std::cout << "error in GeometryGroup.updateCPUSimInstance(): no geometries attached to group" << std::endl;
		return GEOMGROUP_NOGEOM_ERR;
	}
	this->mode=simParams;
	/* create Instances of geometryGroups */
	unsigned int i;
	for (i=0; i<geometryListLength; i++)
	{
		if ( GEOM_NO_ERR != geometryList[i]->updateCPUSimInstance(lambda, simParams) )
		{
			std::cout << "error in GeometryGroup.updateCPUSimInstance(): geometry.updateCPUSimInstance() returned an error at index:" << i << std::endl;
			return GEOMGROUP_ERR;
		}
	}
	return GEOMGROUP_NO_ERR;
};

/**
 * \detail setGeometryListLength
 *
 * \param[in] unsigned int length 
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks
 * \author Mauch
 */
geometryGroupError GeometryGroup::setGeometryListLength(unsigned int length)
{
	if (geometryList==NULL)
	{
		geometryList=new Geometry*[length];
		for (unsigned int i=0;i<length;i++)
			geometryList[i]=NULL;
		geometryListLength=length;
	}
	else
	{
		std::cout << "error in GeometryGroup.setGeometryListLength(): geometryList has beend initialized before." << std::endl;
		return GEOMGROUP_LISTCREATION_ERR;
	}
	return GEOMGROUP_NO_ERR;
};

/**
 * \detail createGeometry
 *
 * \param[in] unsigned int index
 * 
 * \return geometryGroupError
 * \sa 
 * \remarks
 * \author Mauch
 */
geometryGroupError GeometryGroup::createGeometry(unsigned int index)
{
	if ( (geometryList[index]!=NULL) && (index<geometryListLength) )
	{
		this->geometryList[index]=new Geometry();
		return GEOMGROUP_NO_ERR;
	}
	else
	{
		std::cout << "error in GeometryGroup.createGeometry(): invalid geometryIndex:" << index << std::endl;
		return GEOMGROUP_LISTCREATION_ERR;
	}
};

/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &field, vector<Field*> &fieldVec
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryGroupError  GeometryGroup::parseXml(pugi::xml_node &geomGroup)
{
	Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToAccelType(geomGroup, "accelType", this->getParamsPtr()->acceleration)))
		return GEOMGROUP_ERR;

	return GEOMGROUP_NO_ERR;
};

const char* GeometryGroup::accelTypeToAscii(accelType acceleration) const
{
	const char* ascii;
	switch (acceleration)
	{
	case ACCEL_NOACCEL:
		ascii="NoAccel";
		break;
	case ACCEL_BVH:
		ascii="Bvh";
		break;
	case ACCEL_SBVH:
		ascii="Sbvh";
		break;
	case ACCEL_MBVH:
		ascii="Mbvh";
		break;
	case ACCEL_LBVH:
		ascii="Lbvh";
		break;
	case ACCEL_TRIANGLEKDTREE:
		ascii="TriangleKdTree";
		break;
	default:
		ascii="NoAccel";
		break;
	}
	return ascii;
};

const char* GeometryGroup::accelTypeToTraverserAscii(accelType acceleration) const
{
	const char* ascii;
	switch (acceleration)
	{
	case ACCEL_NOACCEL:
		ascii="NoAccel";
		break;
	case ACCEL_BVH:
		ascii="Bvh";
		break;
	case ACCEL_SBVH:
		ascii="Bvh";
		break;
	case ACCEL_MBVH:
		ascii="Bvh";
		break;
	case ACCEL_LBVH:
		ascii="Bvh";
		break;
	case ACCEL_TRIANGLEKDTREE:
		ascii="KdTree";
		break;
	default:
		ascii="NoAccel";
		break;
	}
	return ascii;
};

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
bool GeometryGroup::checkParserError(char *msg)
{
	if (msg==NULL)
		return true;
	else
	{
		cout << "error in GeometryGroup.parseXML(): " << msg << endl;
		delete msg;
		msg=NULL;
		return false;
	}
};