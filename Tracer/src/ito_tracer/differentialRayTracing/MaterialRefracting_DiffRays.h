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

/**\file MaterialRefracting_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALREFRACTING_DIFFRAYS_H
#define MATERIALREFRACTING_DIFFRAYS_H

//#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "../Group.h"
#include "Material_DiffRays.h"
#include "../MaterialRefracting.h"
#include "MaterialRefracting_DiffRays_hit.h"

#define PATH_TO_HIT_REFRACTING_DIFFRAYS "ITO-MacroSim_generated_hitFunctionRefracting"


///* declare class */
///**
//  *\class   MatRefracting_DispersionParams 
//  *\brief   full set of params that is describing the chromatic behaviour of the material properties
//  *
//  *         
//  *
//  *         \todo
//  *         \remarks           
//  *         \sa       NA
//  *         \date     04.01.2011
//  *         \author  Mauch
//  *
//  */
//class MatRefracting_DiffRays_DispersionParams : public MatDispersionParams
//{
//public:
//	MaterialDispersionFormula dispersionFormula;
//	double lambdaMin;
//	double lambdaMax;
//	double paramsNom[5];
//	double paramsDenom[5];
//};

/* declare class */
/**
  *\class   MaterialRefracting_DiffRays 
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
class MaterialRefracting_DiffRays: public MaterialRefracting
{
	protected:
		//MatRefracting_DiffRays_params params; // reduced parameter set for the ray trac ( on GPU )
		//MatRefracting_DiffRays_DispersionParams *glassDispersionParamsPtr; // complete parameter set for glass material
		//MatRefracting_DiffRays_DispersionParams *immersionDispersionParamsPtr; // complete parameter set for immersion medium
		//MatRefracting_params params; // reduced parameter set for the ray trac ( on GPU )
		//MatRefracting_DispersionParams *glassDispersionParamsPtr; // complete parameter set for glass material
		//MatRefracting_DispersionParams *immersionDispersionParamsPtr; // complete parameter set for immersion medium
		//MaterialError calcRefrIndices(double lambda);

  public:
    /* standard constructor */
    MaterialRefracting_DiffRays()
	{
		params.n1=1;
		params.n2=1;
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, PATH_TO_HIT_REFRACTING_DIFFRAYS );
//		this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialRefracting_DiffRays()
	{
		// This will be taken care of in the destructor of the base class !!!

		//if (this->glassDispersionParamsPtr != NULL)
		//	delete this->glassDispersionParamsPtr;
		//if (this->immersionDispersionParamsPtr != NULL)
		//	delete this->immersionDispersionParamsPtr;
	}

//	void setParams(MatRefracting_params params);
//	MatRefracting_params getParams(void);
 //   MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	//MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	//MaterialError createCPUSimInstance(double lambda);
//	double calcSourceImmersion(double lambda);
//	void setPathToPtx(char* path);
//	char* getPathToPtx(void);
	void hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID);
//	void setGlassDispersionParams(MatRefracting_DispersionParams *params);
//	MatRefracting_DispersionParams* getGlassDispersionParams(void);
//	void setImmersionDispersionParams(MatRefracting_DispersionParams *params);
//	MatRefracting_DispersionParams* getImmersionDispersionParams(void);
	//MaterialError updateOptiXInstance(double lambda);
	
};

#endif


