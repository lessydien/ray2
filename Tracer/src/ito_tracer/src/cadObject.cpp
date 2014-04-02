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

/**\file CadObject.cpp
* \brief plane surface
* 
*           
* \author Mauch
*/

#include "CadObject.h"
#include "CadObject_intersect.h"
#include <iostream>
#include "myUtil.h"
#include "Parser_XML.h"

/**
 * \detail getParamsPtr of Surface
 *
 * \param[in] void
 * 
 * \return CadObject_Params*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Geometry_Params* CadObject::getParamsPtr(void)
{
  return this->paramsPtr;
};

/**
 * \detail setParamsPtr of Surface
 *
 * \param[in] CadObject_Params *paramsInPtr
 * 
 * \return geometryError
 * \sa 
 * \remarks 
 * \author Mauch
 */
geometryError CadObject::setParams(Geometry_Params *paramsIn)//CadObject_Params *paramsIn)
{
	CadObject_Params *l_ptr=dynamic_cast<CadObject_Params*>(paramsIn);
	// if the incoming pointer has the correct type, copy the params
	if (l_ptr != NULL)
		*(this->paramsPtr)=*l_ptr;
	else
	{
		std::cout << "error in CadObject.setParams(): paramsIn seems to not be of type CadObject_Params" << "...\n";
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
double CadObject::intersect(rayStruct *ray)
{
	return 0;//intersectRayCadObject(ray->position,ray->direction,*(this->reducedParamsPtr));
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
//double CadObject::intersect(diffRayStruct *ray)
//{
//	return intersectRayCadObject(ray->position,ray->direction,*(this->reducedParamsPtr));
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
geometryError CadObject::reduceParams(void)
{
	if ( (this->paramsPtr!=NULL) && (this->reducedParamsPtr!=NULL) )
	{
		this->reducedParamsPtr->geometryID=this->paramsPtr->geometryID;
		this->reducedParamsPtr->root=this->paramsPtr->root;
//		this->reducedParamsPtr->apertureRadius=this->paramsPtr->apertureRadius;
		this->reducedParamsPtr->normal=this->paramsPtr->normal;
//		this->reducedParamsPtr->apertureType=this->paramsPtr->apertureType;
		//this->reducedParamsPtr->rotNormal=this->paramsPtr->rotNormal;
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
geometryError CadObject::hit(rayStruct &ray, double t)
{
	Mat_hitParams hitParams;
	hitParams.normal=this->paramsPtr->normal;
	int i;
	for (i=0;i<this->materialListLength;i++)
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
//geometryError CadObject::hit(diffRayStruct &ray, double t)
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
geometryError CadObject::createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	// CAD objects are placed in their own geometryGroup
	this->reduceParams();

	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryCreate( context, &geometry ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometrySetPrimitiveCount( geometry, model->getCompiledIndexCount()/3 ), context) )
		return GEOM_ERR;

	/* set intersection program */
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_intersect, "intersect", &geometry_intersection_program ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometrySetIntersectionProgram( geometry, geometry_intersection_program ), context) )
		return GEOM_ERR;
	//RTprogram  l_bounding_box_program;
	if ( !RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_intersect, "bounds", &geometry_boundingBox_program ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometrySetBoundingBoxProgram( geometry, geometry_boundingBox_program ), context) )
		return GEOM_ERR;

	int vertSize=model->getCompiledVertexSize();
	int vertCount=model->getCompiledVertexCount();
	int normSize=model->getNormalSize();
	int normCount=model->getNormalCount();
	int indCount=model->getIndexCount();
	int test=model->getPositionCount();


	/* declare vertex buffer */
	RTvariable vertex_buffer;
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "vertex_buffer", &vertex_buffer ), context ))
		return GEOM_ERR;
    /* Render vertex buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &vertex_buffer_obj ), context ))
		return GEOM_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( vertex_buffer_obj, RT_FORMAT_USER ), context ))
		return GEOM_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( vertex_buffer_obj, vertSize*sizeof(float) ), context ))
		return GEOM_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( vertex_buffer_obj, vertCount ), context ))
		return GEOM_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( vertex_buffer, vertex_buffer_obj ), context ))
		return GEOM_ERR;

	// fill vertex buffer
	RTsize buffer_width;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferGetSize1D(vertex_buffer_obj, &buffer_width), context ))
		return GEOM_ERR;

	void *dataVert;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(vertex_buffer_obj, &dataVert) , context))
		return GEOM_ERR;
	memcpy(dataVert, model->getCompiledVertices(), vertSize*vertCount*sizeof(float));
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( vertex_buffer_obj ) , context))
		return GEOM_ERR;

	/* declare index buffer */
	RTvariable index_buffer;
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "index_buffer", &index_buffer ), context ))
		return GEOM_ERR;
	/* Render index buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &index_buffer_obj ), context ))
		return GEOM_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( index_buffer_obj, RT_FORMAT_INT3 ), context ))
		return GEOM_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( index_buffer_obj, indCount/3 ), context ))
		return GEOM_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( index_buffer, index_buffer_obj ), context ))
		return GEOM_ERR;
	// fill index buffer
	void *dataIdx;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(index_buffer_obj, &dataIdx) , context))
		return GEOM_ERR;
	memcpy(dataIdx, model->getCompiledIndices(), indCount/3*sizeof(int3));
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( index_buffer_obj ) , context))
		return GEOM_ERR;

	/* Create this geometry instance */
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceCreate( context, &instance ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceSetGeometry( instance, geometry ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceSetMaterialCount( instance, materialListLength ), context) )
		return GEOM_ERR;

	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceDeclareVariable( instance, "geometryID", &geometryID ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSet1i(geometryID, this->getParamsPtr()->geometryID), context) )
		return GEOM_ERR;

	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceDeclareVariable( instance, "params", &params ), context) )
		return GEOM_ERR;
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(CadObject_ReducedParams), this->reducedParamsPtr), context) )
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


	/* set bounding box program */
	//if (GEOM_NO_ERR != this->createOptixBoundingBox( context, geometry ) )
	//{
	//	std::cout <<"error in Geometry.createOptixInstance(): createOptixBoundingBox() returned an error at geometry:" << this->getParamsPtr()->geometryID << "...\n";
	//	return GEOM_ERR;
	//}

	/* add this geometry instance to geometry group */
	if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryGroupSetChild( geometrygroup, index, instance ), context) )
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
geometryError CadObject::updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda )
{
	if (this->update)
	{
		if (GEOM_NO_ERR != this->updateOptixInstance(context, geometrygroup, index, simParams, lambda) )
		{
			std::cout <<"error in CadObject.updateOptixInstance(): Geometry.updateOptiXInstacne() returned an error at geometry: " << this->paramsPtr->geometryID << "...\n";
			return GEOM_ERR;
		}
		/* set the variables of the geometry */
		if ( !RT_CHECK_ERROR_NOEXIT( rtGeometryInstanceQueryVariable( instance, "params", &params ), context) )
			return GEOM_ERR;
		if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(CadObject_ReducedParams), this->reducedParamsPtr), context) )
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
geometryError CadObject::processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID)
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

geometryError CadObject::parseXml(pugi::xml_node &geometry, SimParams simParams, vector<Geometry*> &geomVec)
{
	// parse base class
	if (GEOM_NO_ERR!=Geometry::parseXml(geometry,simParams, geomVec))
	{
		std::cout << "error in CadObject.parseXml(): Geometry.parseXml() returned an error." << "...\n";
		return GEOM_ERR;
	}
	double3 l_vec=make_double3(0,0,1);
	rotateRay(&l_vec,this->getParamsPtr()->tilt);
	this->paramsPtr->normal=l_vec;
	Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "root.x", this->paramsPtr->root.x)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "root.y", this->paramsPtr->root.y)))
		return GEOM_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(geometry, "root.z", this->paramsPtr->root.z)))
		return GEOM_ERR;
	const char* l_objFileName=l_parser.attrValByName(geometry, "objFilename");
	if (l_objFileName==NULL)
	{
		std::cout << "error in CadObject.parseXml(): glassName is not defined" << "...\n";
		return GEOM_ERR;
	}
	// load object file
	model = new nv::Model();
	if(!model->loadModelFromFile(l_objFileName)) {
		std::cout << "error in CadObject.parseXml(): Unable to load model '" << l_objFileName << "'" << "...\n";
		return GEOM_ERR;
	}
	model->compileModel();

	geomVec.push_back(this);
	return GEOM_NO_ERR;

}