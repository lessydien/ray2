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

/**\file Material.h
* \brief 
* 
*           
* \author Mauch
*/

/**
 *\defgroup Material
 */

#ifndef MATERIAL_H
  #define MATERIAL_H

#include <optix.h>
#include <optix_math.h>
#include "rayData.h"
#include "GlobalConstants.h"
#include "Coating.h"
#include "Scatter.h"
//#include "CoatingLib.h"
//#include "ScatterLib.h"
#include "Material_hit.h"
#include "MaterialParams.h"
#include "differentialRayTracing/Material_DiffRays_hit.h"
#include "pugixml.hpp"


typedef enum 
{
  MAT_ERR,
  MAT_NO_ERR
} MaterialError;

/* declare class */
/**
  *\class   Material
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
class Material
{
  protected:
    char* path_to_ptx;
	Coating *coatingPtr;
	Scatter *scatterPtr;
	/* OptiX variables */
	RTprogram closest_hit_program;
    RTprogram any_hit_program;
	RTmaterial OptiXMaterial;
	RTvariable		l_coatingParams;
	RTvariable		l_scatterParams;
	RTvariable		l_params;	
	double lambda_old; // lambda of the last OptiX update. If a new call to updateOptiXInstance comes with another lambda we need to update the refractive index even if the material did not change
		
  public:
	bool update; // if true, the variables of the OptiX-instance have to be update before next call to rtTrace

    Material()
	{
		update=false;
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	virtual ~Material()
	{
		delete path_to_ptx;
	}
	void setPathToPtx(char* path);
	MaterialError setCoating(Coating* ptrIn);
	MaterialError createMaterialHitProgramPtx(RTcontext context, TraceMode mode);
	Coating* getCoating(void);
	MaterialError setScatter(Scatter* ptrIn);
	Scatter* getScatter(void);
	char* getPathToPtx(void);

    virtual MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	virtual MaterialError createCPUSimInstance(double lambda);
	virtual MaterialError updateCPUSimInstance(double lambda);
    virtual MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	virtual double calcSourceImmersion(double lambda);
	virtual void hit(rayStruct &ray,Mat_hitParams hitParams, double t_hit, int geometryID);
	virtual void hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID);
	virtual void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID);
	virtual void setGlassDispersionParams(MatDispersionParams *params);
	virtual MatDispersionParams* getGlassDispersionParams(void);
	virtual void setImmersionDispersionParams(MatDispersionParams *params);
	virtual MatDispersionParams* getImmersionDispersionParams(void);

	virtual MaterialError processParseResults(MaterialParseParamStruct &parseResults_Mat);
	virtual MaterialError parseXml(pugi::xml_node &node);
	virtual bool checkParserError(char *msg);
};

#endif

