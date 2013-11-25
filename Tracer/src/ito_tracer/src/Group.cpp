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

/**\file Group.cpp
* \brief container that can hold multiple geometrygroups thereby introducing another level for forming a hierarchical scene graph of the geometries
* 
*           
* \author Mauch
*/

#include "Group.h"
#include "GeometryGroup.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>

/**
 * \detail trace
 *
 * \param[in] rayStruct &ray
 * 
 * \return groupError
 * \sa 
 * \remarks
 * \author Mauch
 */
groupError  Group::trace(rayStruct &ray)
{
	for(int i=0;i<this->geometryGroupListLength;i++)
	{
		this->getGeometryGroup(i)->trace(ray);
	}
	return GROUP_NO_ERR;
};

/**
 * \detail trace
 *
 * \param[in] diffRayStruct &ray
 * 
 * \return groupError
 * \sa 
 * \remarks
 * \author Mauch
 */
groupError  Group::trace(diffRayStruct &ray)
{
	for(int i=0;i<this->geometryGroupListLength;i++)
	{
		this->getGeometryGroup(i)->trace(ray);
	}
	return GROUP_NO_ERR;
};

/**
 * \detail trace
 *
 * \param[in] gaussBeamRayStruct &ray
 * 
 * \return groupError
 * \sa 
 * \remarks
 * \author Mauch
 */
groupError  Group::trace(gaussBeamRayStruct &ray)
{
	for(int i=0;i<this->geometryGroupListLength;i++)
	{
		this->getGeometryGroup(i)->trace(ray);
	}
	return GROUP_NO_ERR;
};

/**
 * \detail getGeometryGroupListLength
 *
 * \param[in] void
 * 
 * \return int geometryGroupListLength
 * \sa 
 * \remarks
 * \author Mauch
 */
int Group::getGeometryGroupListLength(void)
{
	return geometryGroupListLength;
};

/**
 * \detail setGeometryGroup
 *
 * \param[in] GeometryGroup* oGeometryGroupPtr, int index
 * 
 * \return groupError
 * \sa 
 * \remarks this is dangerous as it only copies the pointer to the geometry OptiX_group !!
 * \author Mauch
 */
groupError Group::setGeometryGroup(GeometryGroup* oGeometryGroupPtr, int index)
{
	this->geometryGroupList[index]=oGeometryGroupPtr;
	return GROUP_NO_ERR;
};

/**
 * \detail getGeometryGroup
 *
 * \param[in] int index
 * 
 * \return GeometryGroup*
 * \sa 
 * \remarks 
 * \author Mauch
 */
GeometryGroup* Group::getGeometryGroup(int index)
{
	return this->geometryGroupList[index];	
};

/**
 * \detail createOptixInstance
 *
 * \param[in] RTcontext &context, TraceMode mode, double lambda
 * 
 * \return groupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
groupError Group::createOptixInstance(RTcontext &context, SimParams simParams, double lambda)
{

	//RTgroup OptiX_group;
	//RTselector selector;
	RTacceleration		top_level_acceleration;
	//RTvariable			top_object;
	RTprogram			l_visit_program;

	/* check wether a geometry OptiX_group is present */
	if (this->geometryGroupListLength==0)
	{
		std::cout << "error in Group.createOptixSimInstance(): no geometryGroups attached to group" << std::endl;
		return GROUP_NOGEOMGROUP_ERR;
	}

	if ( (simParams.traceMode==SIM_GEOMRAYS_SEQ) || (simParams.traceMode==SIM_DIFFRAYS_SEQ) )
	{
		/* create top level selector in context */
		if (!RT_CHECK_ERROR_NOEXIT( rtSelectorCreate( context, &selector ), context ))
			return GROUP_ERR;
		// if we are in sequential mode we put each geometry into its own geometrygroup. Therefore we create as many geometrygroups as we have geometries...
		if (!RT_CHECK_ERROR_NOEXIT( rtSelectorSetChildCount( selector, this->getGeometryGroup(0)->getGeometryListLength() ), context ))
			return GROUP_ERR;

		if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "top_object", &top_object ), context ))
			return GROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( top_object, selector ), context ))
			return GROUP_ERR;

		/* create Instances of geometryGroups */
		int i;
		for (i=0; i<geometryGroupListLength; i++)
		{
			if ( GEOMGROUP_NO_ERR != geometryGroupList[i]->createOptixInstance(context, selector, i, simParams, lambda) )
			{
				std::cout << "error in Group.createOptixInstance: geometryGroupList[i]->createOptixInstance returned an error at index:" << i << std::endl;
				return GROUP_ERR;
			}
		}

		/* set visit program */
		if (!RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_visit, "visit_program", &l_visit_program ), context ))
			return GROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtSelectorSetVisitProgram( selector, l_visit_program ), context ))
			return GROUP_ERR;

		return GROUP_NO_ERR;
	}
	else
	{
		/* create top level OptiX_group in context */
		if (!RT_CHECK_ERROR_NOEXIT( rtGroupCreate( context, &OptiX_group ), context ))
			return GROUP_ERR;
		// if we are in nonsequential mode we create only on geometryGroup to place all geometries in
		if (!RT_CHECK_ERROR_NOEXIT( rtGroupSetChildCount( OptiX_group, this->getGeometryGroupListLength() ), context ))
			return GROUP_ERR;

		if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "top_object", &top_object ), context ))
			return GROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( top_object, OptiX_group ), context ))
			return GROUP_ERR;

		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationCreate( context, &top_level_acceleration ), context ))
			return GROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationSetBuilder(top_level_acceleration,"NoAccel"), context ))
			return GROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationSetTraverser(top_level_acceleration,"NoAccel"), context ))
			return GROUP_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtGroupSetAcceleration( OptiX_group, top_level_acceleration), context ))
			return GROUP_ERR;

		if (!RT_CHECK_ERROR_NOEXIT( rtAccelerationMarkDirty( top_level_acceleration ), context ))
			return GROUP_ERR;


		/* create Instances of geometryGroups */
		int i;
		for (i=0; i<geometryGroupListLength; i++)
		{
			if ( GEOMGROUP_NO_ERR != geometryGroupList[i]->createOptixInstance(context, OptiX_group, i, simParams, lambda) )
			{
				std::cout << "error in Group.createOptixInstance: geometryGroupList[i]->createOptixInstance returned an error at index:" << i << std::endl;
				return GROUP_ERR;
			}
		}

		return GROUP_NO_ERR;
	}
};

/**
 * \detail updateOptixInstance
 *
 * \param[in] RTcontext &context, TraceMode mode, double lambda
 * 
 * \return groupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
groupError Group::updateOptixInstance(RTcontext &context, SimParams simParams, double lambda)
{
	
	/* check wether a geometry OptiX_group is present */
	if (this->geometryGroupListLength==0)
	{
		std::cout << "error in Group.updateOptixSimInstance(): no geometryGroups attached to group" << std::endl;
		return GROUP_NOGEOMGROUP_ERR;
	}

	if ( (this->mode==SIM_GEOMRAYS_SEQ) || (this->mode==SIM_DIFFRAYS_SEQ) )
	{
		/* create top level selector in context */
//		RT_CHECK_ERROR_NOEXIT( rtSelectorCreate( context, &selector ) );
		// if we are in sequential mode we put each geometry into its own geometrygroup. Therefore we create as many geometrygroups as we have geometries...
		if (!RT_CHECK_ERROR_NOEXIT( rtSelectorSetChildCount( selector, this->getGeometryGroup(0)->getGeometryListLength() ), context ))
			return GROUP_ERR;

//		RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "top_object", &top_object ) );
//		RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( top_object, selector ) );

		/* create Instances of geometryGroups */
		int i;
		for (i=0; i<geometryGroupListLength; i++)
		{
			geometryGroupList[i]->updateOptixInstance(context, selector, i, simParams, lambda);
		}

		/* set visit program */
//		RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_visit, "visit_program", &l_visit_program ) );
//		RT_CHECK_ERROR_NOEXIT( rtSelectorSetVisitProgram( selector, l_visit_program ) );

		return GROUP_NO_ERR;
	}
	else
	{
		/* create top level OptiX_group in context */
//		RT_CHECK_ERROR_NOEXIT( rtGroupCreate( context, &OptiX_group ) );
		// if we are in nonsequential mode we create only on geometryGroup to place all geometries in
		if (!RT_CHECK_ERROR_NOEXIT( rtGroupSetChildCount( OptiX_group, this->getGeometryGroupListLength() ), context ))
			return GROUP_ERR;

//		RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "top_object", &top_object ) );
//		RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( top_object, OptiX_group ) );

//		RT_CHECK_ERROR_NOEXIT( rtAccelerationCreate( context, &top_level_acceleration ) );
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationSetBuilder(top_level_acceleration,"NoAccel") );
//		RT_CHECK_ERROR_NOEXIT( rtAccelerationSetTraverser(top_level_acceleration,"NoAccel") );
//		RT_CHECK_ERROR_NOEXIT( rtGroupSetAcceleration( OptiX_group, top_level_acceleration) );

//		RT_CHECK_ERROR_NOEXIT( rtAccelerationMarkDirty( top_level_acceleration ) );


		/* create Instances of geometryGroups */
		int i;
		for (i=0; i<geometryGroupListLength; i++)
		{
			geometryGroupList[i]->updateOptixInstance(context, OptiX_group, i, simParams, lambda);
		}

		return GROUP_NO_ERR;
	}
};

/**
 * \detail createCPUSimInstance
 *
 * \param[in] double lambda, TraceMode mode
 * 
 * \return groupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
groupError Group::createCPUSimInstance(double lambda, SimParams simParams )
{
	/* check wether a geometry OptiX_group is present */
	if (this->geometryGroupListLength==0)
	{
		std::cout << "error in Group.createCPUSimInstance(): no geometryGroups attached to group" << std::endl;
		return GROUP_NOGEOMGROUP_ERR;
	}
		
	/* create Instances of geometryGroups */
	int i;
	for (i=0; i<geometryGroupListLength; i++)
	{
		if ( GEOMGROUP_NO_ERR != geometryGroupList[i]->createCPUSimInstance(lambda, simParams) )
		{
			std::cout << "error in Group.createCPUSimInstance(): geometryGroup[i].createCPUSimInstance() returned an error at index" << i << std::endl;
			return GROUP_ERR;
		}
	}
	return GROUP_NO_ERR;
};

/**
 * \detail updateCPUSimInstance
 *
 * \param[in] double lambda, TraceMode mode
 * 
 * \return groupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
groupError Group::updateCPUSimInstance(double lambda, SimParams simParams )
{
	/* check wether a geometry OptiX_group is present */
	if (this->geometryGroupListLength==0)
	{
		std::cout << "error in Group.updateCPUSimInstance(): no geometryGroups attached to group" << std::endl;
		return GROUP_NOGEOMGROUP_ERR;
	}
		
	/* create Instances of geometryGroups */
	int i;
	for (i=0; i<geometryGroupListLength; i++)
	{
		if ( GEOMGROUP_NO_ERR != geometryGroupList[i]->updateCPUSimInstance(lambda, simParams) )
		{
			std::cout << "error in Group.updateCPUSimInstance(): geometryGroup[i].updateCPUSimInstance() returned an error at index" << i << std::endl;
			return GROUP_ERR;
		}
	}
	return GROUP_NO_ERR;
};

/**
 * \detail setGeometryGroupListLength
 *
 * \param[in] int length
 * 
 * \return groupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
groupError Group::setGeometryGroupListLength(int length)
{
	if (geometryGroupList==NULL)
	{
		geometryGroupList=new GeometryGroup*[length];
		for (int i=0;i<length;i++)
			geometryGroupList[i]=NULL;
		geometryGroupListLength=length;
	}
	else
	{
		std::cout << "error in Group.setGeometryGroupListLength(): geometryGroupList has been initialized before." << std::endl;
		return GROUP_LISTCREATION_ERR;
	}
	return GROUP_NO_ERR;
}

/**
 * \detail createGeometryGroup
 *
 * \param[in] int index
 * 
 * \return groupError
 * \sa 
 * \remarks 
 * \author Mauch
 */
groupError Group::createGeometryGroup(int index)
{
	if ( (geometryGroupList[index]==NULL) && (index<geometryGroupListLength) )
	{
		this->geometryGroupList[index]=new GeometryGroup();
		return GROUP_NO_ERR;
	}
	else
	{
		std::cout << "error in Group.createGeometryGroup(): invalid index" << index << std::endl;
		return GROUP_LISTCREATION_ERR;
	}
};



