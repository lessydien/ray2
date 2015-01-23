#ifndef MATERIALREFLECTING_H
#define MATERIALREFLECTING_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>

class MaterialReflecting: public Material
{
	protected:

  public:
    /* standard constructor */
    MaterialReflecting()
	{
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionReflecting.cu.ptx" );
		this->setPathToPtx(path_to_ptx);
	}

    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index);
	void setPathToPtx(char* path);
	char* getPathToPtx(void);
	void hit(rayStruct &ray,double3 normal);

};

#endif


