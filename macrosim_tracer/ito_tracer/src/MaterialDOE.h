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

/**\file MaterialDOE.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALDOE_H
#define MATERIALDOE_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "Group.h"
#include "MaterialDOE_hit.h"

#define PATH_TO_HIT_DOE "macrosim_tracer_generated_hitFunctionDOE"

/* declare class */
/**
  *\class   MatDOE_DispersionParams 
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
class MatDOE_DispersionParams : public MatDispersionParams
{
public:
	MaterialDispersionFormula dispersionFormula;
	double lambdaMin;
	double lambdaMax;
	double paramsNom[6];
	double paramsDenom[6];

};

/* declare class */
/**
  *\class   MaterialDOE 
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
class MaterialDOE: public Material
{
	protected:
		//double *coeffVec;
		MatDOE_coeffVec coeffVec;
		//double *effLookUpTable;
		MatDOE_lookUp effLookUpTable;
		MatDOE_params params; // reduced parameter set for the ray trac ( on GPU )
		MatDOE_DispersionParams *glassDispersionParamsPtr; // complete parameter set for glass material
		MatDOE_DispersionParams *immersionDispersionParamsPtr; // complete parameter set for immersion medium
		MaterialError calcRefrIndices(double lambda);

		RTvariable		l_coeffVec;
		RTvariable		l_effLookUpTable;

  public:
    /* standard constructor */
    MaterialDOE()
	{
		params.n1=1;
		params.n2=1;
		this->glassDispersionParamsPtr=NULL;
		this->immersionDispersionParamsPtr=NULL;
		/* set ptx path for OptiX calculations */
		//path_to_ptx=(char*)malloc(512*sizeof(char));
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_DOE );
		params.effLookUpTableDims=make_short3(0,0,0);
		//this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialDOE()
	{
		if (this->glassDispersionParamsPtr != NULL)
		{
			delete this->glassDispersionParamsPtr;
			this->glassDispersionParamsPtr=NULL;
		}
		if (this->immersionDispersionParamsPtr != NULL)
		{
			delete this->immersionDispersionParamsPtr;
			this->immersionDispersionParamsPtr=NULL;
		}
//		delete path_to_ptx;
	}

	void setParams(MatDOE_params params);
	MatDOE_params getParams(void);
    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError createCPUSimInstance(double lambda);
	double calcSourceImmersion(double lambda);
//	void setPathToPtx(char* path);
//	char* getPathToPtx(void);
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
//	void hit(diffRayStruct &ray,double3 normal, double3 mainDirX, double3 mainDirY, double2 mainRad, double t_hit, int geometryID);
	void setGlassDispersionParams(MatDOE_DispersionParams *params);
	MatDOE_DispersionParams* getGlassDispersionParams(void);
	void setImmersionDispersionParams(MatDOE_DispersionParams *params);
	MatDOE_DispersionParams* getImmersionDispersionParams(void);
	virtual MaterialError processParseResults(MaterialParseParamStruct &parseResults_Mat, parseGlassResultStruct* parseResults_GlassPtr, parseGlassResultStruct* parseResults_ImmPtr);
	MaterialError parseXml(pugi::xml_node &geometry, SimParams simParams);	
};

#endif


