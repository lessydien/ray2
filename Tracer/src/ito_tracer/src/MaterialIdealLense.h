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

/**\file MaterialIdealLense.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALIDEALLENSE_H
#define MATERIALIDEALLENSE_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "Group.h"
#include "MaterialIdealLense_hit.h"

#define PATH_TO_HIT_IDEALLENSE "ITO-MacroSim_generated_hitFunctionIdealLense"

/* declare class */
/**
  *\class   MatIdealLense_DispersionParams 
  *\ingroup Material
  *\brief   full set of params describing the chromatic behaviour of the material properties
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class MatIdealLense_DispersionParams
{
public:
	double f0; //!> focal length of ideal lense at centre wwavelength
	double lambda0; //!> centre wavelength 
	double A; //!> dispersion constant
	double3 root; //!> root of the ideal lense
	double3 orientation; //!> orientation of the ideal lense
	double2 apertureHalfWidth; //!> aperture radius of ideal lense. This is needed to calc the thickness of the ideal lense...
};

/* declare class */
/**
  *\class   MaterialIdealLense 
  *\ingroup Material
  *\brief   
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class MaterialIdealLense: public Material
{
	protected:
		MatIdealLense_params params; // reduced parameter set for the ray trace ( on GPU )
		MatIdealLense_DispersionParams *dispersionParamsPtr; // complete parameter set for glass material
		MaterialError calcFocalLength(double lambda);
		double lambda_old; // lambda of the last OptiX update. If a new call to updateOptiXInstance comes with another lambda we need to update the refractive index even if the material did not change

  public:
    /* standard constructor */
    MaterialIdealLense()
	{
		params.f=0;
		dispersionParamsPtr=new MatIdealLense_DispersionParams();
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_IDEALLENSE );
//		this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialIdealLense()
	{
		if (this->dispersionParamsPtr!=NULL)
		{
			delete this->dispersionParamsPtr;
			this->dispersionParamsPtr=NULL;
		}
//		delete path_to_ptx;
	}


	void setParams(MatIdealLense_params params);
	MatIdealLense_params getParams(void);
    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	MaterialError createCPUSimInstance(double lambda);
//	void setPathToPtx(char* path);
//	char* getPathToPtx(void);
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID);
	void setDispersionParams(MatIdealLense_DispersionParams *params);
	MatIdealLense_DispersionParams* getDispersionParams(void);
	MaterialError updateCPUSimInstance(double lambda);
	MaterialError processParseResults(MaterialParseParamStruct &parseResults_Mat);
};

#endif


