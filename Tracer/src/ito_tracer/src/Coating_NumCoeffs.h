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

/**\file Coating_NumCoeffs.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef COATING_NUMCOEFFS_H
  #define COATING_NUMCOEFFS_H

#include <optix.h>
#include "Coating.h"
#include <optix_math.h>
#include "rayData.h"
#include "GlobalConstants.h"
#include "Coating_NumCoeffs_hit.h"
#include <stdio.h>
//#include "CoatingLib.h"

#define PATH_TO_HIT_COAT_NUMCOEFFS "_CoatNumCoeffs"

/* declare class */
/**
  *\class   Coating_NumCoeffs_FullParams
  *\ingroup Coating
  *\brief   base class of full set of params of Coating_NumCoeffs
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
class Coating_NumCoeffs_FullParams: public Coating_FullParams
{
	public:
		Coating_NumCoeffs_FullParams()
		{
			type=CT_NUMCOEFFS;
			t=0;
			r=0;
		}
		~Coating_NumCoeffs_FullParams()
		{
		}
		//CoatingType type;
		double t; // amplitude transmission coefficient
		double r; // amplitude reflection coefficient
};

/* declare class */
/**
  *\class   Coating_NumCoeffs
  *\ingroup Coating
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
class Coating_NumCoeffs: public Coating
{
  protected:
//    char* path_to_ptx;
	Coating_NumCoeffs_FullParams *fullParamsPtr;
	Coating_NumCoeffs_ReducedParams *reducedParamsPtr;
	//CoatingError calcCoatingCoeffs(double lambda, double3 normal, double3 direction);
		
  public:
	 Coating_NumCoeffs() // standard constructor
	 {
		 this->fullParamsPtr=new Coating_NumCoeffs_FullParams(); 
		 this->fullParamsPtr->type = CT_NUMCOEFFS;
		 this->fullParamsPtr->t=1;
		 this->fullParamsPtr->r=0;
		 this->reducedParamsPtr=new Coating_NumCoeffs_ReducedParams();
		 path_to_ptx=(char*)malloc(512*sizeof(char));
		 sprintf( path_to_ptx, "%s", PATH_TO_HIT_COAT_NUMCOEFFS );

	 }
	 ~Coating_NumCoeffs() // destructor
	 {
		 delete this->fullParamsPtr;
		 delete this->reducedParamsPtr;
		 delete path_to_ptx;
	 }

//	void setPathToPtx(char* path) {	memcpy(this->path_to_ptx, path, sizeof(this->path_to_ptx));	};
//	char* getPathToPtx(void){return this->path_to_ptx;};
	CoatingError setFullParams(Coating_NumCoeffs_FullParams* ptrIn);
	Coating_NumCoeffs_FullParams* getFullParams(void);
	Coating_NumCoeffs_ReducedParams* getReducedParams(void);
//    virtual CoatingError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	CoatingError createCPUSimInstance(double lambda);
	CoatingError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	bool hit(rayStruct &ray, Mat_hitParams hitParams);
	bool hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	CoatingError processParseResults(MaterialParseParamStruct &parseResults_Mat);
	CoatingError parseXml(pugi::xml_node &geometry, SimParams simParams);
	bool checkParserError(char *msg);
};

#endif

