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

/**\file MaterialFilter.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALFILTER_H
#define MATERIALFILTER_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "MaterialFilter_hit.h"

#define PATH_TO_HIT_FILTER "ITO-MacroSim_generated_hitFunctionFilter"

/* declare class */
/**
  *\class   MatFilter_DispersionParams 
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
class MatFilter_DispersionParams
{
public:
	double lambdaMax; //!> upper edge wavelength of pass band
	double lambdaMin; //!> lower edge wavelength of pass band
};

/* declare class */
/**
  *\class   MaterialFilter
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
class MaterialFilter: public Material
{
	protected:
		MatFilter_params params; // reduced parameter set for the ray trace ( on GPU )
		MatFilter_DispersionParams *dispersionParamsPtr; // complete parameter set 

		MaterialError calcFilter(double lambda);
		double lambda_old; // lambda of the last OptiX update. If a new call to updateOptiXInstance comes with another lambda we need to update the refractive index even if the material did not change


  public:
    /* standard constructor */
    MaterialFilter()
	{
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_FILTER );
		this->dispersionParamsPtr=NULL;
//		this->setPathToPtx(path_to_ptx);
	}

	/* Destruktor */
    ~MaterialFilter()
	{
		if (this->dispersionParamsPtr!=NULL)
			delete this->dispersionParamsPtr;
//		delete path_to_ptx;
	}

	void setParams(MatFilter_params params);
	MatFilter_params getParams(void);

    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError createCPUSimInstance(double lambda);
	MaterialError updateCPUSimInstance(double lambda);
//	void setPathToPtx(char* path);
//	char* getPathToPtx(void);
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID);
	void setDispersionParams(MatFilter_DispersionParams *params);
	MatFilter_DispersionParams* getDispersionParams(void);
	MaterialError processParseResults(MaterialParseParamStruct &parseResults_Mat);
};
#endif