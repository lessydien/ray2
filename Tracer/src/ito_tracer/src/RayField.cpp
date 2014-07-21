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

/**\file RayField.cpp
* \brief base class of all ray based representations of the light field
* 
*           
* \author Mauch
*/

#include "RayField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include <ctime>
#include "Parser_XML.h"
#include "MaterialLib.h"
#include "SimAssistant.h"

/**
 * \detail calcSubsetDim 
 *
 * \param[in] void
 * 
 * \return long2
 * \sa 
 * \remarks 
 * \author Mauch
 */
long2 RayField::calcSubsetDim(void)
{
	std::cout << "error in RayField.calcSubsetDim(): not implemented yet for this RayField type" << "...\n";
	return make_long2(0,0);
}

/**
 * \detail setLambda 
 *
 * \param[in] double
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::setLambda(double lambda)
{
	std::cout << "error in RayField.setLambda(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
}

/**
 * \detail getRayListLength 
 *
 * \param[in] void
 * 
 * \return unsigend long long
 * \sa 
 * \remarks 
 * \author Mauch
 */
unsigned long long RayField::getRayListLength(void)
{
	return this->rayListLength;
};

/**
 * \detail setRay 
 *
 * \param[in] rayStructBase ray, unsigned long long index
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::setRay(rayStructBase ray, unsigned long long index)
{
	//if (index <= this->rayListLength)
	//{
	//	rayList[index]=ray;
	//	return FIELD_NO_ERR;
	//}
	//else
	//{
	//	return FIELD_INDEXOUTOFRANGE_ERR;
	//}
	std::cout << "error in RayField.setRay(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail getRay 
 *
 * \param[in] unsigned long long index
 * 
 * \return rayStructBase*
 * \sa 
 * \remarks 
 * \author Mauch
 */
rayStructBase* RayField::getRay(unsigned long long index)
{
	//if (index <= this->rayListLength)
	//{
	//	return &rayList[index];	
	//}
	//else
	//{
	//	return 0;
	//}
	std::cout << "error in RayField.getRay(): not implemented yet for this RayField type" << "...\n";
	return NULL;
};

/**
 * \detail getRayList 
 *
 * \param[in] void
 * 
 * \return rayStructBase*
 * \sa 
 * \remarks 
 * \author Mauch
 */
rayStructBase* RayField::getRayList(void)
{
//	return &rayList[0];
	std::cout << "error in RayField.getRayList(): not implemented yet for this RayField type" << "...\n";
	return NULL;
};

/* functions for GPU usage */

void RayField::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx_rayGeneration, path, sizeof(this->path_to_ptx_rayGeneration));
};

/**
 * \detail getPathToPtx 
 *
 * returns the path to the ptx file that the .cu file defining the behaviour of the RayField on the GPU of the surface will be compiled to
 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
const char* RayField::getPathToPtx(void)
{
	return this->path_to_ptx_rayGeneration;
};

/**
 * \detail copyRayList 
 *
 * \param[in] rayStruct *data, long long length
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::copyRayList(rayStruct *data, long long length)
{
	std::cout << "error in RayField.copyRayList(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail copyRayListSubset 

 *
 * \param[in] rayStruct *data, long2 launchOffset, long2 subsetDim
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::copyRayListSubset(rayStruct *data, long2 launchOffset, long2 subsetDim)
{
	std::cout << "error in RayField.copyRayListSubset(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;

};
/**
 * \detail createCPUSimInstance 
 *
 * \param[in] unsigned long long launch_width, unsigned long long launch_height, double3 start, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double lambda
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void RayField::createCPUSimInstance(unsigned long long launch_width, unsigned long long launch_height, double3 start, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double lambda)
{
	std::cout << "error in RayField.createCPUSimInstance(): not implemented yet for this RayField type" << "...\n";
};

/**
 * \detail createCPUSimInstance 
 *
 * \param[in] 
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::createCPUSimInstance()
{
	std::cout << "error in RayField.createCPUSimInstance(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail createLayoutInstance 
 *
 * \param[in] 
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::createLayoutInstance()
{
	std::cout << "error in RayField.createLayoutInstance(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail createOptixInstance 
 *
 * \param[in] RTcontext &context
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	RTvariable   seed_buffer;

    RTvariable epsilon;
	RTvariable max_depth;
	RTvariable min_flux;

	RTvariable offsetX;
	RTvariable offsetY;

	/* Ray generation program */
	char rayGenName[128];
	sprintf(rayGenName, "rayGeneration");
	switch (this->getParamsPtr()->dirDistrType)
	{
	case RAYDIR_RAND_RECT:
		strcat(rayGenName, "_DirRand");
		break;
	case RAYDIR_RANDIMPAREA:
		strcat(rayGenName, "_DirRandImpArea");
		break;
	case RAYDIR_UNIFORM:
		strcat(rayGenName, "_DirUniform");
		break;
	case RAYDIR_GRID_RECT:
		strcat(rayGenName, "_DirGridRect");
		break;
	case RAYDIR_GRID_RAD:
		strcat(rayGenName, "_DirGridRad");
		break;
	default:
		std::cout <<"error in RayField.createOptixInstance(): unknown distribution of ray directions." << "...\n";
		return FIELD_ERR;
		break;
	}
	switch (this->getParamsPtr()->posDistrType)
	{
	case RAYPOS_RAND_RECT:
		strcat(rayGenName, "_PosRandRect");
		break;
	case RAYPOS_GRID_RECT:
		strcat(rayGenName, "_PosGridRect");
		break;
	case RAYPOS_RAND_RAD:
		strcat(rayGenName, "_PosRandRad");
		break;
	case RAYPOS_GRID_RAD:
		strcat(rayGenName, "_PosGridRad");
		break;
	default:
		std::cout <<"error in RayField.createOptixInstance(): unknown distribution of ray positions." << "...\n";
		return FIELD_ERR;
		break;
	}
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_rayGeneration, rayGenName, &this->ray_gen_program ), context ))
		return FIELD_ERR;

	/* declare seed buffer */
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "seed_buffer", &seed_buffer ), context ))
		return FIELD_ERR;
    /* Render seed buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &seed_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( seed_buffer_obj, RT_FORMAT_UNSIGNED_INT ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( seed_buffer_obj, GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( seed_buffer, seed_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare variables */
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "launch_offsetX", &offsetX ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "launch_offsetY", &offsetY ), context ))
		return FIELD_ERR;

	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "scene_epsilon", &epsilon ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "max_depth", &max_depth ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "min_flux", &min_flux ), context ))
		return FIELD_ERR;

    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSet1f( epsilon, EPSILON ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSet1i( max_depth, MAX_DEPTH_CPU ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSet1f( min_flux, MIN_FLUX_CPU ), context ))
		return FIELD_ERR;

	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetX, sizeof(long long), &(this->getParamsPtr()->launchOffsetX)), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetY, sizeof(long long), &(this->getParamsPtr()->launchOffsetY)), context ))
		return FIELD_ERR;

    if (!RT_CHECK_ERROR_NOEXIT( rtContextSetRayGenerationProgram( context,0, this->ray_gen_program ), context ))
		return FIELD_ERR;

	return FIELD_NO_ERR;
};

/**
 * \detail createOptixInstance 
 *
 * \param[in] RTcontext* context, unsigned long long width, unsigned long long height, double3 start, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::createOptixInstance(RTcontext* context, unsigned long long width, unsigned long long height, double3 start, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference)
{
	std::cout << "error in RayField.createOptixInstance(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail initGPUSubset 
 *
 * \param[in] unsigned int jx, unsigned int jy
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::initGPUSubset(RTcontext &context)
{
	std::cout << "error in RayField.initGPUSubset(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
}
/**
 * \detail traceScene 
 *
 * \param[in] Group &oGroup
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::traceScene(Group &oGroup)
{
	std::cout << "error in RayField.traceScene(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

/**
* \detail initCPUSubset 
*
* \param[in] void
* 
* \return fieldError
* \sa 
* \remarks 
* \author Mauch
*/
fieldError RayField::initCPUSubset()
{
	std::cout << "error in RayField.initCPUSubset(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail initGPUSubset 

 *
 * \param[in] RTcontext &context, RTbuffer &seed_buffer_obj
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
 fieldError RayField::initGPUSubset(RTcontext &context, RTbuffer &seed_buffer_obj)
{
	RTvariable offsetX;
	RTvariable offsetY;
	RTvariable seed_buffer;

	long long l_offsetX=this->getParamsPtr()->launchOffsetX;
	long long l_offsetY=this->getParamsPtr()->launchOffsetY;

	if (!RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( this->ray_gen_program, "launch_offsetX", &offsetX ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( this->ray_gen_program, "launch_offsetY", &offsetY ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( this->ray_gen_program, "seed_buffer", &seed_buffer ), context))
		return FIELD_ERR;

	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetX, sizeof(long long), &l_offsetX), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetY, sizeof(long long), &l_offsetY), context ))
		return FIELD_ERR;

	// before we refill the seed buffer for the nex launch we need to check which seeds we need to keept for the upcoming launch
	// we need to keep those seeds that were used to determine those starting points that not all rays where launched from yet...

	// calc iGesMax of last launch
	unsigned long long iGesMaxOld;
	if ( (l_offsetX==0) && (l_offsetY==0) )
		iGesMaxOld=0;
	else
	{
		if ( (l_offsetX==0) && (l_offsetY>0) )
			iGesMaxOld=l_offsetY*this->getParamsPtr()->width*this->getParamsPtr()->nrRayDirections.x*this->getParamsPtr()->nrRayDirections.y-1;
		else
			iGesMaxOld=l_offsetY*this->getParamsPtr()->width*this->getParamsPtr()->nrRayDirections.x*this->getParamsPtr()->nrRayDirections.y+l_offsetX-1;
	}
	// calc the index of the seed buffer where the seeds that we need to keep start
	unsigned long long iPosX=floorf(iGesMaxOld/(this->getParamsPtr()->nrRayDirections.x*this->getParamsPtr()->nrRayDirections.y));
	unsigned long long iPosY=floorf(iPosX/this->getParamsPtr()->width);
	iPosX=iPosX % this->getParamsPtr()->width;
	unsigned long long startIndex=(iPosX+iPosY*this->getParamsPtr()->width) % this->getParamsPtr()->GPUSubset_width;


	/* refill seed buffer */
	void *data;
	// read the seed buffer from the GPU
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(seed_buffer_obj, &data), context ))
		return FIELD_ERR;
	uint* seeds = reinterpret_cast<uint*>( data );


	RTsize buffer_width, buffer_height;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferGetSize1D(seed_buffer_obj, &buffer_width), context ))
		return FIELD_ERR;
	//for ( unsigned int i = 0; i < (unsigned int)buffer_width; ++i )
	//	seeds[i] = (uint)BRandom(x);
	
	for ( unsigned int i = 0; i < (unsigned int)startIndex; ++i )
		seeds[i] = (uint)BRandom(x);
	for ( unsigned int i = startIndex+1; i < (unsigned int)buffer_width; ++i )
		seeds[i] = (uint)BRandom(x);

	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( seed_buffer_obj ), context ))
		return FIELD_ERR;
	
	return FIELD_NO_ERR;
 };

/**
 * \detail traceScene 
 *
 * \param[in] Group &oGroup, bool RunOnCPU
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::traceScene(Group &oGroup, bool RunOnCPU)
{
	std::cout << "error in RayField.traceScene(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail traceStep 
 *
 * \param[in] Group &oGroup, bool RunOnCPU
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::traceStep(Group &oGroup, bool RunOnCPU)
{
	std::cout << "error in RayField.traceStep(): not implemented yet for this RayField type" << "...\n";
	return FIELD_ERR;
};

fieldError RayField::createOptiXContext()
{
	RTprogram  miss_program;
    //RTvariable output_buffer;

    /* variables for the miss program */

    /* Setup context */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextCreate( &context ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtContextSetRayTypeCount( context, 1 ), context )) 
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtContextSetEntryPointCount( context, 1 ), context ))
		return FIELD_ERR;

	//rtContextSetExceptionEnabled(context, RT_EXCEPTION_ALL, 1);
	//rtContextSetPrintEnabled(context, 1);
	//rtContextSetPrintBufferSize(context, 14096 );
	//rtContextSetPrintLaunchIndex(context, -1, 0, 0);

    /* variables for the miss program */

	char* path_to_ptx;
	path_to_ptx=(char*)malloc(512*sizeof(char));
    /* Miss program */
	sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_missFunction.cu.ptx" );
    if (!RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, path_to_ptx, "miss", &miss_program ), context ))
	{
		cout << "error in RayField.createOptixContext(): creating miss program from ptx at " << path_to_ptx << " failed." << endl;
		return FIELD_ERR;
	}
    if (!RT_CHECK_ERROR_NOEXIT( rtContextSetMissProgram( context, 0, miss_program ), context ))
		return FIELD_ERR;

	rtContextSetStackSize(context, 1536);
	//rtContextGetStackSize(context, &stack_size_bytes);

	delete path_to_ptx;
	return FIELD_NO_ERR;
};

/**
 * \detail initSimulation 
 *
 * \param[in] Group &oGroup, simAssParams &params
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::initSimulation(Group &oGroup, simAssParams &params)
{
	tracedRayNr=0;
	if (params.RunOnCPU)
	{
		if (FIELD_NO_ERR!=this->createCPUSimInstance())
		{
			std::cout <<"error in RayField.createOptixInstance(): create CPUSimInstance() returned an error." << "...\n";
			return FIELD_ERR;
		}
		if (GROUP_NO_ERR != oGroup.createCPUSimInstance(this->getParamsPtr()->lambda, params.simParams) )
		{
			std::cout << "error in RayField.initSimulation(): group.createCPUSimInstance() returned an error" << "...\n";
			return FIELD_ERR;
		}
	}
	else
	{
		if (FIELD_NO_ERR != this->createOptiXContext())
		{
			std::cout << "error in RayField.initSimulation(): createOptiXInstance() returned an error" << "...\n";
			return FIELD_ERR;
		}
		// convert geometry to GPU code
		if ( GROUP_NO_ERR != oGroup.createOptixInstance(context, params.simParams, this->getParamsPtr()->lambda) )
		{
			std::cout << "error in RayField.initSimulation(): group.createOptixInstance returned an error" << "...\n";
			return ( FIELD_ERR );
		}
			// convert rayfield to GPU code
			if ( FIELD_NO_ERR != this->createOptixInstance(context, output_buffer_obj, seed_buffer_obj) )
			{
				std::cout << "error in RayField.initSimulation(): SourceList[i]->createOptixInstance returned an error at index:" << 0 << "...\n";
				return ( FIELD_ERR );
			}
			if (!RT_CHECK_ERROR_NOEXIT( rtContextValidate( context ), context ))
				return FIELD_ERR;
			if (!RT_CHECK_ERROR_NOEXIT( rtContextCompile( context ), context ))
				return FIELD_ERR;
	}
	return FIELD_NO_ERR;
}

/**
 * \detail initLayout 
 *
 * \param[in] Group &oGroup, simAssParams &params
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::initLayout(Group &oGroup, simAssParams &params)
{
	tracedRayNr=0;
	this->createLayoutInstance();
	if (GROUP_NO_ERR != oGroup.createCPUSimInstance(this->getParamsPtr()->lambda, params.simParams) )
	{
		std::cout << "error in RayField.initSimulation(): group.createCPUSimInstance() returned an error" << "...\n";
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
}

/**
 * \detail writeData2File 
 *
 * \param[in] FILE *hFile_pos, rayDataOutParams outParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
//fieldError RayField::writeData2File(FILE *hFile_pos, rayDataOutParams outParams)
//{
//	return FIELD_NO_ERR;
//};

/**
 * \detail setParamsPtr 
 *
 * \param[in] rayFieldParams *paramsPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void RayField::setParamsPtr(rayFieldParams *paramsPtr)
{
	std::cout << "error in RayField.setParamsPtr(): not implemented for given RayField" << "...\n";
};

/**
 * \detail getParamsPtr 
 *
 * \param[in] void
 * 
 * \return rayFieldParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
rayFieldParams* RayField::getParamsPtr(void)
{
	std::cout << "error in RayField.getParamsPtr(): not implemented for given RayField" << "...\n";
	return NULL;
};


/**
 * \detail setMaterial 
 *
 * \param[in] Material *oMaterialPtr, int index
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::setMaterial(Material *oMaterialPtr, int index)
{
	/* check wether the place in the list is valid */
	if ( (index<materialListLength) )
	{
		materialList[index]=oMaterialPtr;
		return FIELD_NO_ERR;
	}
	/* return error if we end up here */
	std::cout <<"error in RayField.setMaterial(): invalid material index" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail getMaterial 
 *
 * \param[in] int index
 * 
 * \return Material*
 * \sa 
 * \remarks 
 * \author Mauch
 */
Material* RayField::getMaterial(int index)
{
	return materialList[index];	
};

/**
 * \detail setMaterialListLength 
 *
 * \param[in] int length
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::setMaterialListLength(int length)
{
	if (materialList==NULL)
	{
		materialList=new Material*[length];
		materialListLength=length;
	}
	else
	{
		std::cout <<"error in RayField.setMaterialListLength(): materialList has been initialized before" << "...\n";
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
};

/**
 * \detail setMaterialListLength 
 *
 * \param[in] int length
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
int RayField::getMaterialListLength(void)
{
	return this->materialListLength;
};

/**
 * \detail doSim
 *
 * \param[in] 
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  RayField::doSim(Group &oGroup, simAssParams &params, bool &simDone)
{
	std::cout <<"error in RayField.doSim(): this has not yet been implemented for the given Field representation" << "...\n";
	return FIELD_ERR;
};

/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &det
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError RayField::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams)
{

	Parser_XML l_parser;

	// call base class function
	if (FIELD_NO_ERR != Field::parseXml(field, fieldVec, simParams))
	{
		std::cout << "error in RayField.parseXml(): Field.parseXml()  returned an error." << "...\n";
		return FIELD_ERR;
	}

	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "root.x", this->getParamsPtr()->translation.x)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "root.y", this->getParamsPtr()->translation.y)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "root.z", this->getParamsPtr()->translation.z)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "tilt.x", this->getParamsPtr()->tilt.x)))
		return FIELD_ERR;
	this->getParamsPtr()->tilt.x=this->getParamsPtr()->tilt.x/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "tilt.y", this->getParamsPtr()->tilt.y)))
		return FIELD_ERR;
	this->getParamsPtr()->tilt.y=this->getParamsPtr()->tilt.y/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "tilt.z", this->getParamsPtr()->tilt.z)))
		return FIELD_ERR;
	this->getParamsPtr()->tilt.z=this->getParamsPtr()->tilt.z/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "rayDirection.x", this->getParamsPtr()->rayDirection.x)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "rayDirection.y", this->getParamsPtr()->rayDirection.y)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "rayDirection.z", this->getParamsPtr()->rayDirection.z)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "power", this->getParamsPtr()->flux)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "coherence", this->getParamsPtr()->coherence)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "alphaMax.x", this->getParamsPtr()->alphaMax.x)))
		return FIELD_ERR;
	this->getParamsPtr()->alphaMax.x=this->getParamsPtr()->alphaMax.x/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "alphaMax.y", this->getParamsPtr()->alphaMax.y)))
		return FIELD_ERR;
	this->getParamsPtr()->alphaMax.y=this->getParamsPtr()->alphaMax.y/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "alphaMin.x", this->getParamsPtr()->alphaMin.x)))
		return FIELD_ERR;
	this->getParamsPtr()->alphaMin.x=this->getParamsPtr()->alphaMin.x/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "alphaMin.y", this->getParamsPtr()->alphaMin.y)))
		return FIELD_ERR;
	this->getParamsPtr()->alphaMin.y=this->getParamsPtr()->alphaMin.y/360*2*PI;
	unsigned long l_val;
	if (!this->checkParserError(l_parser.attrByNameToLong(field, "width", l_val)))
		return FIELD_ERR;
	this->getParamsPtr()->width=l_val;
	if (!this->checkParserError(l_parser.attrByNameToLong(field, "height", l_val)))
		return FIELD_ERR;
	this->getParamsPtr()->height=l_val;
	if (!this->checkParserError(l_parser.attrByNameToLong(field, "widthLayout", l_val)))
		return FIELD_ERR;
	this->getParamsPtr()->widthLayout=l_val;
	if (!this->checkParserError(l_parser.attrByNameToLong(field, "heightLayout", l_val)))
		return FIELD_ERR;
	this->getParamsPtr()->heightLayout=l_val;
	if (!this->checkParserError(l_parser.attrByNameToRayDirDistrType(field, "rayDirDistrType", this->getParamsPtr()->dirDistrType)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToRayPosDistrType(field, "rayPosDistrType", this->getParamsPtr()->posDistrType)))
		return FIELD_ERR;
	double rotX=this->getParamsPtr()->tilt.x;
	double rotY=this->getParamsPtr()->tilt.y;
	double rotZ=this->getParamsPtr()->tilt.z;
	double3x3 MrotX, MrotY, MrotZ, Mrot;
	MrotX=make_double3x3(1,0,0, 0,cos(rotX),-sin(rotX), 0,sin(rotX),cos(rotX));
	MrotY=make_double3x3(cos(rotY),0,sin(rotY), 0,1,0, -sin(rotY),0,cos(rotY));
	MrotZ=make_double3x3(cos(rotZ),-sin(rotZ),0, sin(rotZ),cos(rotZ),0, 0,0,1);
	Mrot=MrotX*MrotY;
	this->getParamsPtr()->Mrot=Mrot*MrotZ;

	double2 l_aprtHalfWidth;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "apertureHalfWidth.x", l_aprtHalfWidth.x)))
		return FIELD_ERR;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "apertureHalfWidth.y", l_aprtHalfWidth.y)))
		return FIELD_ERR;

	this->getParamsPtr()->rayPosStart=make_double3(-l_aprtHalfWidth.x,-l_aprtHalfWidth.y,0);
	this->getParamsPtr()->rayPosEnd=make_double3(l_aprtHalfWidth.x,l_aprtHalfWidth.y,0);

	// look for material material
	vector<xml_node>* l_pMatNodes;
	l_pMatNodes=l_parser.childsByTagName(field,"material");
	if (l_pMatNodes->size() != 1)
	{
		std::cout << "error in RayField.parseXml(): there must be exactly 1 material attached to each Rayfield." << "...\n";
		return FIELD_ERR;
	}
	// create material
	MaterialFab l_matFab;
	Material* l_pMaterial;
	if (!l_matFab.createMatInstFromXML(l_pMatNodes->at(0),l_pMaterial, simParams))
	{
		std::cout << "error in Geometry.parseXml(): matFab.createInstFromXML() returned an error." << "...\n";
		return FIELD_ERR;
	}

    this->setMaterialListLength(1);
	this->setMaterial(l_pMaterial,0);

	delete l_pMatNodes;


	return FIELD_NO_ERR;
};

/**
 * \detail checks wether parseing was succesfull and assembles the error message if it was not
 *
 * returns the coordinates of the minimum corner of the bounding box of the surface
 *
 * \param[in] char *msg
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool RayField::checkParserError(char *msg)
{
	if (msg==NULL)
		return true;
	else
	{
		cout << "error in RayField.parseXML(): " << msg << endl;
		delete msg;
		msg=NULL;
		return false;
	}
};