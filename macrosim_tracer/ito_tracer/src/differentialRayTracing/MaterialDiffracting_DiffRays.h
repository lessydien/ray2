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

/**\file MaterialDiffracting_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALDIFFRACTING_DIFFRAYS_H
#define MATERIALDIFFRACTING_DIFFRAYS_H

#include "../Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "../Group.h"
#include "../MaterialDiffracting.h"
#include "MaterialDiffracting_DiffRays_hit.h"

#define PATH_TO_HIT_DIFFRACTING_DIFFRAYS "macrosim_tracer_generated_hitFunctionDiffracting"

///* declare class */
///**
//  *\class   MatDiffracting_DispersionParams 
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
//class MatDiffracting_DispersionParams : public MatDispersionParams
//{
//public:
//	MaterialDispersionFormula dispersionFormula;
//	double lambdaMin;
//	double lambdaMax;
//	double paramsNom[5];
//	double paramsDenom[5];
//	double2 importanceConeAngleMax;
//	double2 importanceConeAngleMin;
//	int importanceObjNr;
//	double phiRotZ;
//};


/* declare class */
/**
  *\class   MaterialDiffracting_DiffRays 
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
class MaterialDiffracting_DiffRays: public MaterialDiffracting
{
	protected:

  public:
    /* standard constructor */
    MaterialDiffracting_DiffRays()
	{
		params.n1=1;
		params.n2=1;
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_DIFFRACTING_DIFFRAYS );
//		this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialDiffracting_DiffRays()
	{
//		if (this->fullParamsPtr!=NULL)
//			delete this->fullParamsPtr;
//		if (this->immersionDispersionParamsPtr==NULL)
//			delete this->immersionDispersionParamsPtr;
//		delete path_to_ptx;
	}

 //   MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	//MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	//MaterialError createCPUSimInstance(double lambda);
	void hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID);
	//MaterialError updateOptiXInstance(double lambda);
	
};

#endif


