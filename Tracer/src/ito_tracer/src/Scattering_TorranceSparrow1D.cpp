#include "Scattering_TorranceSparrow1D.h"
#include "GlobalConstants.h"
#include <sutil.h>
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "rayTracingMath.h"


void MaterialScattering_TorranceSparrow1D::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	
};

char* MaterialScattering_TorranceSparrow1D::getPathToPtx(void)
{
	return this->path_to_ptx;
};

void MaterialScattering_TorranceSparrow1D::hit(rayStruct &ray, double3 normal, int geometryID)
{
	extern Group oGroup;
	if (hitTorranceSparrow1D(ray, normal, this->params) )
	{
		ray.depth++;
		ray.currentGeometryID=geometryID;
		if (ray.depth<MAX_DEPTH_CPU )//&& ray.flux>MIN_FLUX_CPU)
		{			
			oGroup.trace(ray);
		}
	}
	else
	{
		// some error mechanism !!
	}

}

void MaterialScattering_TorranceSparrow1D::hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID)
{
	extern Group oGroup;
	// refract all the rays making up the gaussian beam
	//ray.baseRay.direction=calcSnellsLaw(ray.baseRay.direction, normal.normal_baseRay,ray.nImmersed, n);
	//ray.waistRayX.direction=calcSnellsLaw(ray.waistRayX.direction, normal.normal_waistRayX,ray.nImmersed, n);
	//ray.waistRayY.direction=calcSnellsLaw(ray.waistRayY.direction, normal.normal_waistRayY,ray.nImmersed, n);
	//ray.divRayX.direction=calcSnellsLaw(ray.divRayX.direction, normal.normal_divRayX,ray.nImmersed, n);
	//ray.divRayY.direction=calcSnellsLaw(ray.divRayY.direction, normal.normal_divRayY,ray.nImmersed, n);
	//ray.baseRay.depth++;
	//if (ray.baseRay.depth<MAX_DEPTH_CPU && ray.baseRay.flux>MIN_FLUX_CPU)
	//{			
	//	oGroup.trace(ray);
	//}
}



MaterialError MaterialScattering_TorranceSparrow1D::createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda)
{
    RTprogram closest_hit_program;
    RTprogram any_hit_program;
	RTmaterial OptiXMaterial;
	RTvariable		l_params;

	if (mode==SIM_GAUSSBEAMRAYS_NONSEQ)
	{
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionRefracting_GaussBeam.cu.ptx" );
		this->setPathToPtx(path_to_ptx);
	}
    /* Create our hit programs to be shared among all materials */
    //sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionAbsorbing.cu.ptx" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, this->path_to_ptx, "closestHit", &closest_hit_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, this->path_to_ptx, "anyHit", &any_hit_program ) );

	/* set the variables of the geometry */
	RT_CHECK_ERROR( rtProgramDeclareVariable( closest_hit_program, "params", &l_params ) );
	RT_CHECK_ERROR( rtVariableSetUserData(l_params, sizeof(MatTorranceSparrow1D_params), &(this->params)) );

    RT_CHECK_ERROR( rtMaterialCreate( context, &OptiXMaterial ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( OptiXMaterial, 0, closest_hit_program ) );
    RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( OptiXMaterial, 0, any_hit_program ) );

	RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, index, OptiXMaterial ) );
	return MAT_NO_ERROR;	
};

MaterialError MaterialScattering_TorranceSparrow1D::createCPUSimInstance(double lambda, simMode mode)
{
	// calc the refractive indices at current wavelength
	//if (!calcRefrIndices(lambda))
	//	return MAT_ERROR;

	return MAT_NO_ERROR;
};

void MaterialScattering_TorranceSparrow1D::setParams(MatTorranceSparrow1D_params paramsIn)
{
	this->params=paramsIn;
};

MatTorranceSparrow1D_params MaterialScattering_TorranceSparrow1D::getParams(void)
{
	return this->params;
};


