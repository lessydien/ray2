#ifndef MATERIALSCATTERING_TORRANCESPARROW_H
#define MATERIALSCATTERING_TORRANCESPARROW_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "Group.h"
#include "MaterialScattering_TorranceSparrow1D_hit.h"

class MaterialScattering_TorranceSparrow1D: public Material
{
	protected:
		MatTorranceSparrow1D_params params;


  public:
    /* standard constructor */
    MaterialScattering_TorranceSparrow1D()
	{
		params.Kdl=0;
		params.Kdl=0;
		params.Ksp=0;
		params.sigmaXsl=0;
		params.sigmaXsp=0;
		params.scatAxis=make_double3(0,0,0);
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionScattering_TorranceSparrow1D.cu.ptx" );
		this->setPathToPtx(path_to_ptx);
	}

	void setParams(MatTorranceSparrow1D_params paramsIn);
	MatTorranceSparrow1D_params getParams(void);
	//void set_nRefr2(double nIn);
	//double get_nRefr2(void);
    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	void setPathToPtx(char* path);
	char* getPathToPtx(void);
	void hit(rayStruct &ray,double3 normal, int geometryID);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID);
	MaterialError createCPUSimInstance(double lambda, simMode mode);
};

#endif


