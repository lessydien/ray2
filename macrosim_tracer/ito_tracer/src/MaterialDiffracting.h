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

/**\file MaterialDiffracting.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALDIFFRACTING_H
#define MATERIALDIFFRACTING_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "Group.h"
#include "MaterialDiffracting_hit.h"


#define PATH_TO_HIT_DIFFRACTING "macrosim_tracer_generated_hitFunctionDiffracting"

/* declare class */
/**
  *\class   MatDiffracting_DispersionParams 
  *\ingroup Material
  *\brief   full set of params that is describing the chromatic behaviour of the material properties
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
class MatDiffracting_DispersionParams : public MatDispersionParams
{
public:
	MaterialDispersionFormula dispersionFormula;
	double lambdaMin;
	double lambdaMax;
	double paramsNom[5];
	double paramsDenom[5];

	double2 importanceAreaHalfWidth;
	double3 importanceAreaRoot;
	double3 importanceAreaTilt;
	ApertureType importanceAreaApertureType;
};

/* declare class */
/**
  *\class   MaterialDiffracting 
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
class MaterialDiffracting: public Material
{
	protected:
		MatDiffracting_params params; // reduced parameter set for the ray trace ( on GPU )
		MatDiffracting_DispersionParams *fullParamsPtr; // complete parameter set for material
		MatDiffracting_DispersionParams *immersionDispersionParamsPtr; // complete parameter set for immersion medium
		double lambda_old; // lambda of the last OptiX update. If a new call to updateOptiXInstance comes with another lambda we need to update the refractive index even if the material did not change

  public:
    /* standard constructor */
    MaterialDiffracting()
	{
		params.n1=1;
		params.n2=1;
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_DIFFRACTING );
//		this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialDiffracting()
	{
		if (this->fullParamsPtr!=NULL)
			delete this->fullParamsPtr;
		if (this->immersionDispersionParamsPtr==NULL)
			delete this->immersionDispersionParamsPtr;
//		delete path_to_ptx;
	}

	void setParams(MatDiffracting_params params);
	MatDiffracting_params getParams(void);
    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError createCPUSimInstance(double lambda);
//	void setPathToPtx(char* path);
//	char* getPathToPtx(void);
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
//	void hit(diffRayStruct &ray, double3 normal, double3 mainDirX, double3 mainDirY, double2 mainRad, double t_hit, int geometryID);
	void setDispersionParams(MatDiffracting_DispersionParams *params);
	MaterialError calcRefrIndices(double lambda);
	MatDiffracting_DispersionParams* getFullParams(void);
	void setImmersionDispersionParams(MatDiffracting_DispersionParams *params);
	MatDiffracting_DispersionParams* getImmersionDispersionParams(void);
	MaterialError updateOptiXInstance(double lambda);
	virtual MaterialError processParseResults(MaterialParseParamStruct &parseResults_Mat, parseGlassResultStruct* parseResults_GlassPtr);
	
};

#endif


