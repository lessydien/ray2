#include "GeometryPrimitive.h"
#include <sutil.h>
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int  GeometryPrimitive::getPrimitiveID(void) 
{
	return primitiveID;
};

void GeometryPrimitive::setPrimitiveID(int ID) 
{
	primitiveID = ID;
};

void GeometryPrimitive::setBoundingBox_min(float *box_min) 
{
	memcpy(&boundingBox_min, box_min, sizeof(boundingBox_min));
};

float* GeometryPrimitive::getBoundingBox_min(void) 
{
	return boundingBox_min;
};

void GeometryPrimitive::setBoundingBox_max(float* box_max) 
{
	memcpy(&boundingBox_max, box_max, sizeof(boundingBox_max));
	//*boundingBox_max=*box_max;
};

float* GeometryPrimitive::getBoundingBox_max(void) 
{
	return boundingBox_max;
};

geometryPrimitive_type GeometryPrimitive::getPrimitiveType(void)
{
	return type;
};

void GeometryPrimitive::setPathToPtxIntersect(char* path)
{
	memcpy(this->path_to_ptx_intersect, path, sizeof(this->path_to_ptx_intersect));
};

const char* GeometryPrimitive::getPathToPtxIntersect(void)
{
	return this->path_to_ptx_intersect;
};

void GeometryPrimitive::setPathToPtxBoundingBox(char* path)
{
	memcpy(this->path_to_ptx_boundingBox, path, sizeof(this->path_to_ptx_boundingBox));
};

const char* GeometryPrimitive::getPathToPtxBoundingBox(void)
{
	return this->path_to_ptx_boundingBox;
};

void GeometryPrimitive::createOptiXBoundingBox( RTcontext &context, RTgeometry &geometry )
{
	RTprogram  geometry_bounding_box_program;
    RTvariable box_min_var;
    RTvariable box_max_var;
	//char path_to_ptx[512];

    //sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_planeSurface.cu.ptx" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, this->path_to_ptx_boundingBox, "bounds", &geometry_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( geometry, geometry_bounding_box_program ) );

    RT_CHECK_ERROR( rtGeometryDeclareVariable( geometry, "boxmin", &box_min_var ) );
    RT_CHECK_ERROR( rtGeometryDeclareVariable( geometry, "boxmax", &box_max_var ) );
    RT_CHECK_ERROR( rtVariableSet3fv( box_min_var, this->boundingBox_min ) );
	RT_CHECK_ERROR( rtVariableSet3fv( box_max_var, this->boundingBox_max ) );
};

void GeometryPrimitive::setType(geometryPrimitive_type type)
{
	this->type=type;
};

geometryPrimitive_type GeometryPrimitive::getType(void)
{
	return this->type;
};

void GeometryPrimitive::setMaterial(RTmaterial &materialIn)
{
  material=materialIn;
};

RTmaterial GeometryPrimitive::getMaterial(void)
{
  return material;
};


