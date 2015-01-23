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

/**\file Geometry.cpp
* \brief base class for all geometries
* 
*           
* \author Mauch
*/

#include "Geometry.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "Material.h"
#include "MaterialLib.h"
#include "differentialRayTracing\MaterialLib_DiffRays.h"
#include "geometricRender\MaterialLib_GeomRender.h"

#include "Parser_XML.h"

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] GeometryParseParamStruct &parseResults_Geom
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID)
{
	std::cout << "error in Geometry.processParseResults(): not defined for the given Field representation" << "...\n";
	return GEOM_ERR;
};

/**
 * \detail parseXml 
 *
 * sets the parameters of the base class geometry according to the given xml node
 *
 * \param[in] xml_node &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::parseXml(pugi::xml_node &geometry, SimParams simParams, vector<Geometry*> &geomVec)
{
	Parser_XML l_parser;

	const char* l_pName=l_parser.attrValByName(geometry, "name");
	if (l_pName)
	{
		sprintf(this->name, "%s", l_pName);
		this->name[GEOM_CMT_LENGTH-1]='\0';
	}

	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "tilt.x", this->getParamsPtr()->tilt.x)))
		return GEOM_ERR;
	this->getParamsPtr()->tilt.x=this->getParamsPtr()->tilt.x/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "tilt.y", this->getParamsPtr()->tilt.y)))
		return GEOM_ERR;
	this->getParamsPtr()->tilt.y=this->getParamsPtr()->tilt.y/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "tilt.z", this->getParamsPtr()->tilt.z)))
		return GEOM_ERR;
	this->getParamsPtr()->tilt.z=this->getParamsPtr()->tilt.z/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToApertureType(geometry, "apertureType", this->getParamsPtr()->apertureType)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "apertureRadius.x", this->getParamsPtr()->apertureRadius.x)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "apertureRadius.y", this->getParamsPtr()->apertureRadius.y)))
		return GEOM_ERR;
	//if (!l_parser.attrByNameToInt(geometry, "geometryID", this->getParamsPtr()->geometryID))
	//{
	//	std::cout << "error in Geometry.parseXml(): geometryID is not defined" << "...\n";
	//	return GEOM_ERR;
	//}
	// look for material material
	vector<xml_node>* l_pMatNodes;
	l_pMatNodes=l_parser.childsByTagName(geometry,"material");
	if (l_pMatNodes->size() != 1)
	{
		std::cout << "error in Geometry.parseXml() of Geometry " << this->name << ": there must be exactly 1 material attached to each geometry." << "...\n";
		return GEOM_ERR;
	}
	// create material
	MaterialFab* l_pMatFab;
    switch (simParams.simMode)
    {
        case SIM_GEOM_RT:
            l_pMatFab=new MaterialFab();
            break;
        case SIM_DIFF_RT:
            l_pMatFab=new MaterialFab_DiffRays();
            break;
        case SIM_GEOM_RENDER:
            //l_pMatFab=new MaterialFab(); // we can use the same materials in geometric render mode
            l_pMatFab=new MaterialFab_GeomRender();
            break;
        default:
            std::cout << "error in Geometry.parseXml(): unknown simulation mode." << "...\n";
            return GEOM_ERR;
            break;
    }
	Material* l_pMaterial;
	if (!l_pMatFab->createMatInstFromXML(l_pMatNodes->at(0),l_pMaterial, simParams))
	{
		std::cout << "error in Geometry.parseXml() of Geometry " << this->name << ": matFab.createInstFromXML() returned an error." << "...\n";
		return GEOM_ERR;
	}

	this->setMaterial(l_pMaterial,0);

	delete l_pMatNodes;
    delete l_pMatFab;
	return GEOM_NO_ERR;
}

/**
 * \detail setBoundingBox_min of Surface
 *
 * sets the coordinates of the minimum corner of the bounding box of the surface
 *
 * \param[in] float *box_min
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Geometry::setBoundingBox_min(float *box_min) 
{
	memcpy(&boundingBox_min, box_min, sizeof(boundingBox_min));
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
bool Geometry::checkParserError(char *msg)
{
	if (msg==NULL)
		return true;
	else
	{
		cout << "error in Geometry.parseXML() of geometry " << this->name << ": " << msg << endl;
		delete msg;
		msg=NULL;
		return false;
	}
};

/**
 * \detail getBoundingBox_min of Surface
 *
 * returns the coordinates of the minimum corner of the bounding box of the surface
 *
 * \param[in] void
 * 
 * \return float*
 * \sa 
 * \remarks 
 * \author Mauch
 */
float* Geometry::getBoundingBox_min(void) 
{
	return boundingBox_min;
};

/**
 * \detail setBoundingBox_max of Surface
 *
 * sets the coordinates of the maximum corner of the bounding box of the surface
 *
 * \param[in] float* box_max
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Geometry::setBoundingBox_max(float* box_max) 
{
	memcpy(&boundingBox_max, box_max, sizeof(boundingBox_max));
	//*boundingBox_max=*box_max;
};

/**
 * \detail getBoundingBox_max of Surface
 *
 * returns the coordinates of the maximum corner of the bounding box of the surface
 *
 * \param[in] void
 * 
 * \return float*
 * \sa 
 * \remarks 
 * \author Mauch
 */
float* Geometry::getBoundingBox_max(void) 
{
	return boundingBox_max;
};

/**
 * \detail getBoundingBox_min of Surface
 *
 * returns the coordinates of the minimum corner of the bounding box of the surface
 *
 * \param[in] void
 * 
 * \return float*
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometry_type Geometry::getType(void)
{
	return type;
};

/**
 * \detail setType of Surface
 *
 * \param[in] geometry_type typeIn
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Geometry::setType(geometry_type typeIn)
{
	type=typeIn;
};

/**
 * \detail setPathToPtxIntersect of Surface
 *
 * sets the path to the ptx file that the .cu file defining the intersection on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Geometry::setPathToPtxIntersect(char* path)
{
	memcpy(this->path_to_ptx_intersect, path, sizeof(this->path_to_ptx_intersect));
};
/**
 * \detail getPathToPtxIntersect of Surface
 *
 * returns the path to the ptx file that the .cu file defining the intersection on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return char* path
 * \sa 
 * \remarks 
 * \author Mauch
 */
const char* Geometry::getPathToPtxIntersect(void)
{
	return this->path_to_ptx_intersect;
};

/**
 * \detail setPathToPtxBoundingBox of Surface
 *
 * sets the path to the ptx file that the .cu file defining the boundingBox function on the GPU will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void Geometry::setPathToPtxBoundingBox(char* path)
{
	memcpy(this->path_to_ptx_boundingBox, path, sizeof(this->path_to_ptx_boundingBox));
};

/**
 * \detail getPathToPtxIntersect of Surface
 *
 * returns the path to the ptx file that the .cu file defining boundingBox function on the GPU of the surface will be compiled to
 *
 * \param[in] void
 * 
 * \return char* path
 * \sa 
 * \remarks 
 * \author Mauch
 */
char* Geometry::getPathToPtxBoundingBox(void)
{
	return this->path_to_ptx_boundingBox;
};

/**
 * \detail createOptixBoundingBox of Surface
 *
 * sets the parameters to the boundingBox function on the GPU and creates an OptiX instance of it
 *
 * \param[in] RTcontext &context, RTgeometry &geometry
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::createOptixBoundingBox(RTcontext &context, RTgeometry &geometry )
{
	RTprogram  geometry_bounding_box_program;
    RTvariable box_min_var;
    RTvariable box_max_var;
	//char* path_to_ptx;
	//sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "macrosim_tracer_generated_boundingBox.cu.ptx" );
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_boundingBox, "bounds", &geometry_bounding_box_program ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometrySetBoundingBoxProgram( geometry, geometry_bounding_box_program ), context) )
		return GEOM_ERR;
//	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryDeclareVariable( geometry, "boxmin", &box_min_var ), context) )
//		return GEOM_ERR;
//	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryDeclareVariable( geometry, "boxmax", &box_max_var ), context) )
//		return GEOM_ERR;
//	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSet3fv( box_min_var, this->boundingBox_min ), context) )
//		return GEOM_ERR;
//	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSet3fv( box_max_var, this->boundingBox_max ), context) )
//		return GEOM_ERR;

	return GEOM_NO_ERR;
};

/**
 * \detail createCPUSimInstance of Surface
 *
 * calculates the reduced params of the surface and the materials attached to it from the full parameter set for the given simulation parameters
 *
 * \param[in] double lambda, TraceMode mode
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::createCPUSimInstance(double lambda, SimParams simParams )
{
	this->reduceParams();
	/* check wether any material is present */
	if (this->materialListLength==0)
	{
		std::cout << "error in Geometry.createCPUInstance(): no material attached to surface at geometry:" << this->getParamsPtr()->geometryID << "...\n";
		return GEOM_NOMATERIAL_ERR;
	}
	this->mode=simParams.traceMode;
	/* create instances of material */
	int i;
	for (i=0; i<materialListLength; i++)
	{
		if ( MAT_NO_ERR != this->materialList[i]->createCPUSimInstance(lambda) )
		{
			std::cout << "error in Geometry.createCPUInstance(): material.createCPUSimInstance() returned an error at geometry:" << this->getParamsPtr()->geometryID << "...\n";
			return GEOM_ERR;
		}
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
geometryError Geometry::reduceParams(void)
{
	std::cout << "error in Geometry.reduceParams(): method seems to not be overwritten in child class." << "...\n";
	return GEOM_ERR;
};

/**
 * \detail createCPUSimInstance of Surface
 *
 * updates the reduced params of the surface and the materials attached to it from the full parameter set for the given simulation parameters
 *
 * \param[in] double lambda, TraceMode mode
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::updateCPUSimInstance(double lambda, SimParams simParams )
{
	/* check wether any material is present */
	if (this->materialListLength==0)
	{
		std::cout << "error in Geometry.updateCPUSimInstance(): no material attached to surface at geometry:" << this->getParamsPtr()->geometryID << "...\n";
		return GEOM_NOMATERIAL_ERR;
	}
	this->mode=simParams.traceMode;
	/* create instances of material */
	int i;
	for (i=0; i<materialListLength; i++)
	{
		if ( MAT_NO_ERR != this->materialList[i]->updateCPUSimInstance(lambda) )
		{
			std::cout << "error in Geometry.updateCPUSimInstance(): material.createCPUSimInstance() returned an error at geometry:" << this->getParamsPtr()->geometryID << "...\n";
			return GEOM_ERR;
		}
	}
	if (this->update)
		this->reduceParams();
	return GEOM_NO_ERR;
};

/**
 * \detail setMaterial of Surface
 *
 * \param[in] Material *oMaterialPtr, int index
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::setMaterial(Material *oMaterialPtr, int index)
{
	/* check wether the place in the list is valid */
	if ( (index<materialListLength) )
	{
		materialList[index]=oMaterialPtr;
		return GEOM_NO_ERR;
	}
	/* return error if we end up here */
	std::cout << "error in Geometry.setMaterial: invalid material index" << "...\n";
	return GEOM_LISTCREATION_ERR;
};

/**
 * \detail createMaterial 
 *
 * creates a Material instance at index i
 *
 * \param[in] int index
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::createMaterial(int index)
{
	if ( (materialList[index]!=NULL) && (index<materialListLength) )
	{
		materialList[index]=new Material();
		return GEOM_NO_ERR;
	}
	else
	{
		std::cout << "error in Geometry.createMaterial(): invalid material index at geometry:" << this->getParamsPtr()->geometryID << "...\n";
		return GEOM_LISTCREATION_ERR;
	}
}

/**
 * \detail getMaterial
 *
 * \param[in] int index
 * 
 * \return Material*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Material* Geometry::getMaterial(int index)
{
	return materialList[index];	
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
geometryError Geometry::createOptixInstance(RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda)
{
	this->reduceParams();
	/* check wether any material is present */
	if (this->materialListLength==0)
	{
		std::cout <<"error in Geometry.createOptixInstance(): no material attached to surface at geometry:" << this->getParamsPtr()->geometryID << "...\n";
		return GEOM_NOMATERIAL_ERR;
	}
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryCreate( context, &geometry ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometrySetPrimitiveCount( geometry, 1u ), context) )
		return GEOM_ERR;
	/* Create this geometry instance */
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceCreate( context, &instance ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceSetGeometry( instance, geometry ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceSetMaterialCount( instance, materialListLength ), context) )
		return GEOM_ERR;

	/* create instances of material */
	int i;
	for (i=0; i<materialListLength; i++)
	{
		if (MAT_NO_ERR != this->materialList[i]->createOptiXInstance(context, instance, i, simParams, lambda) )
		{
			std::cout <<"error in Geometry.createOptixInstance(): materialList[i]->createOptiXInstance() returned an error at index:" << i << " at geometry: " << this->getParamsPtr()->geometryID << "...\n";
			return GEOM_ERR;
		}
	}

	/* set the variables of the geometry */
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceDeclareVariable( instance, "params", &params ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceDeclareVariable( instance, "geometryID", &geometryID ), context) )
		return GEOM_ERR;
	//if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceDeclareVariable( instance, "mode", &l_mode ), context) )
	//	return GEOM_ERR;
	//if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceDeclareVariable( instance, "materialListLength", &l_materialListLength ), context) )
	//	return GEOM_ERR;

	//if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_mode, sizeof(traceMode), &mode), context) )
	//	return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSet1i(geometryID, this->getParamsPtr()->geometryID), context) )
		return GEOM_ERR;
	//if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSet1i(l_materialListLength, this->materialListLength), context) )
	//	return GEOM_ERR;

	/* set intersection program */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_intersect, "intersect", &geometry_intersection_program ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometrySetIntersectionProgram( geometry, geometry_intersection_program ), context) )
		return GEOM_ERR;

	/* set bounding box program */
	if (GEOM_NO_ERR != this->createOptixBoundingBox( context, geometry ) )
	{
		std::cout <<"error in Geometry.createOptixInstance(): createOptixBoundingBox() returned an error at geometry:" << this->getParamsPtr()->geometryID << "...\n";
		return GEOM_ERR;
	}

	/* add this geometry instance to geometry group */
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetChild( geometrygroup, index, instance ), context) )
		return GEOM_ERR;

	this->update=false;
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
geometryError Geometry::updateOptixInstance(RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda)
{
	if (this->update)
	{
		this->reduceParams();
		/* check wether any geometry is present */
		if (this->materialListLength==0)
		{
			std::cout <<"error in PlaneSurface.updateOptixInstance(): no material attached to surface at geometry:" << this->getParamsPtr()->geometryID << "...\n";
			return GEOM_NOMATERIAL_ERR;
		}

		/* create instances of material */
		int i;
		for (i=0; i<materialListLength; i++)
		{
			if (MAT_NO_ERR != this->materialList[i]->updateOptiXInstance(context, instance, i, simParams, lambda) )
			{
				std::cout <<"error in PlaneSurface.updateOptixInstance(): materialList[i] returned an error at index:" << i << " at geometry: " << this->getParamsPtr()->geometryID << "...\n";
				return GEOM_ERR;
			}
		}

		/* set the variables of the geometry */
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceQueryVariable( instance, "geometryID", &geometryID ), context ) )
			return GEOM_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceQueryVariable( instance, "mode", &l_mode ), context ) )
			return GEOM_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceQueryVariable( instance, "materialListLength", &l_materialListLength ), context ) )
			return GEOM_ERR;

		if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(l_mode, sizeof(TraceMode), &mode), context ) )
			return GEOM_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtVariableSet1i(geometryID, this->getParamsPtr()->geometryID), context ) )
			return GEOM_ERR;
		if (!RT_CHECK_ERROR_NOEXIT( rtVariableSet1i(l_materialListLength, this->materialListLength), context ) )
			return GEOM_ERR;

	}
	this->update=false;

	/* dummy function to be overwritten by child class */
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
double  Geometry::intersect(rayStruct *ray)
{
	/* dummy function to be overwritten by child class */
	return -1;
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
double  Geometry::intersect(diffRayStruct *ray)
{
	/* dummy function to be overwritten by child class */
	std::cout << "error in Geometry.intersect(): intersect is not yet implemented for differential rays for the given geometry. Geometry is ignored..." << "...\n";
	return -1;
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
geometryError  Geometry::hit(rayStruct &ray, double t)
{
	std::cout << "error in Geometry.hit(): hit is not yet implemented for geometric rays for the given geometry" << "...\n";
	/* dummy function to be overwritten by child class */
	return GEOM_ERR;
};

/**
 * \detail hit function function for differential rays
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
geometryError  Geometry::hit(diffRayStruct &ray, double t)
{
	std::cout << "error in Geometry.hit(): hit is not yet implemented for differential rays for the given geometry" << "...\n";
	/* dummy function to be overwritten by child class */
	return GEOM_ERR;
};

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
gaussBeam_t  Geometry::intersect(gaussBeamRayStruct *ray)
{
	/* dummy function to be overwritten by child class */
	gaussBeam_t t;
	t.t_baseRay=0;
	return t;
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
geometryError  Geometry::hit(gaussBeamRayStruct &ray, gaussBeam_t t)
{
	/* dummy function to be overwritten by child class */
	return GEOM_NO_ERR;
};

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
Geometry_Params* Geometry::getParamsPtr(void)
{
	// dummy function to be overwritten by child classes
	Geometry_Params* help;
//	help->dummy=true;
	return help;
	//return this->paramsPtr;
}

/**
 * \detail setParamsPtr of Surface
 *
 * \param[in] Geometry_Params *paramsInPtr
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::setParams(Geometry_Params *params)
{
	// dummy function to be overwritten by child classes
	return GEOM_NO_ERR;
}

/**
 * \detail setMaterialListLength
 *
 * \param[in] int length
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::setMaterialListLength(int length)
{
	if (materialList==NULL)
	{
		materialList=new Material*[length];
		materialListLength=length;
	}
	else
	{
		std::cout << "error in Geometry.setMaterialListLength(): materialList already defined at geometry:" << this->getParamsPtr()->geometryID << "...\n";
		return GEOM_LISTCREATION_ERR;
	}
	return GEOM_NO_ERR;
}

/**
 * \detail setComment
 *
 * \param[in] char *ptrIn
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError Geometry::setComment(char *ptrIn)
{
	memcpy ( this->comment, ptrIn, (GEOM_CMT_LENGTH)*sizeof(char) );
	return GEOM_NO_ERR;
}

/**
 * \detail getComment
 *
 * \param[in] void
 * 
 * \return char *ptrIn
 * \sa 
 * \remarks 
 * \author Mauch
 */
char* Geometry::getComment(void)
{
	return this->comment;
}




