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

/**\file MaterialVolumeScatterBox.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALVOLUMESCATTERBOX_H
#define MATERIALVOLUMESCATTERBOX_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "Group.h"
#include "MaterialVolumeScatterBox_hit.h"

#define PATH_TO_HIT_VOLUMESCATTERBOX "ITO-MacroSim_generated_hitFunctionVolumeScatterBox"

/* declare class */
/**
  *\class   MaterialVolumeScatterBox 
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
class MaterialVolumeScatterBox: public Material
{
	protected:
		MatVolumeScatterBox_params params; // reduced parameter set for the ray trac ( on GPU )

  public:
    /* standard constructor */
    MaterialVolumeScatterBox()
	{
		params.n1=1;
		params.n2=1;
		/* set ptx path for OptiX calculations */
		//path_to_ptx=(char*)malloc(512*sizeof(char));
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_VOLUMESCATTERBOX );
		//this->setPathToPtx(path_to_ptx);
	}
	/* Destruktor */
    ~MaterialVolumeScatterBox()
	{
//		delete path_to_ptx;
	}

	void setParams(MatVolumeScatterBox_params params);
	MatVolumeScatterBox_params getParams(void);
    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	MaterialError createCPUSimInstance(double lambda);
	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
	MaterialError parseXml(pugi::xml_node &geometry);	
};

#endif


