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

/**\file MaterialAbsorbing.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALABSORBING_H
#define MATERIALABSORBING_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "MaterialAbsorbing_hit.h"

#define PATH_TO_HIT_ABSORBING "ITO-MacroSim_generated_hitFunctionAbsorbing"

/* declare class */
/**
  *\class   MaterialAbsorbing
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
class MaterialAbsorbing: public Material
{
	protected:

  public:
    /* standard constructor */
    MaterialAbsorbing()
	{
		/* set ptx path for OptiX calculations */
		//path_to_ptx[512];
		sprintf( this->path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_ABSORBING );
		//this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialAbsorbing()
	{
		//delete path_to_ptx;
	}
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
	void hit(diffRayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID);
	MaterialError processParseResults(MaterialParseParamStruct &parseResults_Mat);
	MaterialError parseXml(pugi::xml_node &geometry);
};
#endif