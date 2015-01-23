#include "Material.h"
#include <sutil.h>
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void Material::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	
};

char* Material::getPathToPtx(void)
{
	return this->path_to_ptx;
};



MaterialError Material::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index)
{
    RTprogram closest_hit_program;
    RTprogram any_hit_program;
	RTmaterial OptiXMaterial;

    /* Create our hit programs to be shared among all materials */
    //sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionAbsorbing.cu.ptx" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, this->path_to_ptx, "closestHit", &closest_hit_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, this->path_to_ptx, "anyHit", &any_hit_program ) );

    RT_CHECK_ERROR( rtMaterialCreate( context, &OptiXMaterial ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( OptiXMaterial, 0, closest_hit_program ) );
    RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( OptiXMaterial, 0, any_hit_program ) );

	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, index, OptiXMaterial ) );
	return MAT_NO_ERROR;	
};

void Material::hit(rayStruct &ray, double3 normal)
{
	ray.direction= make_double3(0.0,0.0,0.0);
	ray.depth++;
}

