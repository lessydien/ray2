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

/**\file pathIntTissueRayField.cpp
* \brief Rayfield for geometric raytracing
* 
*           
* \author Mauch
*/
#include <omp.h>
#include "pathIntTissueRayField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include "Converter.h"
#include "MatlabInterface.h"
#include "DetectorLib.h"
#include "MaterialLib.h"
#include <ctime>

using namespace optix;

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
fieldError PathIntTissueRayField::setLambda(double lambda)
{
	this->rayParamsPtr->lambda=lambda;
	return FIELD_NO_ERR;
}

/**
 * \detail getRayListLength 

 *
 * \param[in] void
 * 
 * \return unsigned long long
 * \sa 
 * \remarks 
 * \author Mauch
 */
unsigned long long PathIntTissueRayField::getRayListLength(void)
{
	return this->rayListLength;
};

/**
 * \detail setRay 

 *
 * \param[in] rayStruct ray, unsigned long long index
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathIntTissueRayField::setRay(rayStruct ray, unsigned long long index)
{
	if (index <= this->rayListLength)
	{
		this->rayList[index]=ray;
		return FIELD_NO_ERR;
	}
	else
	{
		return FIELD_INDEXOUTOFRANGE_ERR;
	}
};

/**
 * \detail getRay 

 *
 * \param[in] unsigned long long index
 * 
 * \return rayStruct*
 * \sa 
 * \remarks 
 * \author Mauch
 */
rayStruct* PathIntTissueRayField::getRay(unsigned long long index)
{
	if (index <= this->rayListLength)
	{
		return &(this->rayList[index]);	
	}
	else
	{
		return 0;
	}
};

/**
 * \detail getRayList 

 *
 * \param[in] void
 * 
 * \return rayStruct*
 * \sa 
 * \remarks 
 * \author Mauch
 */
rayStruct* PathIntTissueRayField::getRayList(void)
{
	return &(this->rayList[0]);	
};

/**
 * \detail setRayList 

 *
 * \param[in] rayStruct* rayStructPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void PathIntTissueRayField::setRayList(rayStruct* rayStructPtr)
{
	if (this->rayList!=NULL)
		free (rayList);
	this->rayList=rayStructPtr;
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
fieldError PathIntTissueRayField::copyRayList(rayStruct *data, long long length)
{
	if (length > this->rayListLength)
	{
		std::cout << "error in PathIntTissueRayField.copyRayList(): subset dimensions exceed rayLIst dimension" << "...\n";
		return FIELD_ERR;
	}

	// copy the ray list line per line
	for (unsigned long long jy=0;jy<this->rayParamsPtr->GPUSubset_height;jy++)
	{
		unsigned long long testIndex=jy*Field::GPU_SUBSET_WIDTH_MAX;
		//                     memory range of completed lines + offsetX + number of line in current block*width of complete rayblock // we always allocate the max buffer on GPU, therefore we always need to adress the start of the line in this maximum buffer...
		memcpy(&(this->rayList[jy*Field::GPU_SUBSET_WIDTH_MAX]), &data[jy*Field::GPU_SUBSET_WIDTH_MAX], this->rayParamsPtr->GPUSubset_width*sizeof(rayStruct));
	}
	return FIELD_NO_ERR;
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
fieldError PathIntTissueRayField::copyRayListSubset(rayStruct *data, long2 launchOffset, long2 subsetDim)
{
//	long2 testOffset=launchOffset;
//	long2 testDim=subsetDim;
	//  ----memory range of completed lines---- + ---memory range blocks in given line---
	if (launchOffset.y*this->rayParamsPtr->width+(subsetDim.x+launchOffset.x)*subsetDim.y > this->rayListLength)
	{
		std::cout << "error in PathIntTissueRayField.copyRayListSubset(): subset dimensions exceed rayLIst dimension" << "...\n";
		return FIELD_ERR;
	}
	// copy the ray list line per line
	for (long long jy=0;jy<subsetDim.y;jy++)
	{
		unsigned long long testIndex=launchOffset.y*this->rayParamsPtr->width+launchOffset.x+jy*this->rayParamsPtr->width;
		//                     memory range of completed lines + offsetX + number of line in current block*width of complete rayblock // we always allocate the max buffer on GPU, therefore we always need to adress the start of the line in this maximum buffer...
		memcpy(&(this->rayList[launchOffset.y*this->rayParamsPtr->width+launchOffset.x+jy*this->rayParamsPtr->width]), &data[jy*Field::GPU_SUBSET_WIDTH_MAX], subsetDim.x*sizeof(rayStruct));
	}
	//memcpy(this->rayList, data, length*sizeof(rayStruct));
	return FIELD_NO_ERR;
};
/* functions for GPU usage */

/**
 * \detail setPathToPtx 

 *
 * \param[in] char* path
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void PathIntTissueRayField::setPathToPtx(char* path)
{
	memcpy(this->path_to_ptx_rayGeneration, path, sizeof(this->path_to_ptx_rayGeneration));
};

/**
 * \detail getPathToPtx 

 *
 * \param[in] void
 * 
 * \return char* path
 * \sa 
 * \remarks 
 * \author Mauch
 */
const char* PathIntTissueRayField::getPathToPtx(void)
{
	return this->path_to_ptx_rayGeneration;
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
fieldError PathIntTissueRayField::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	RTvariable   seed_buffer;

    RTvariable epsilon;
	RTvariable max_depth;
	RTvariable min_flux;

	RTvariable output_buffer;
    /* variables for ray gen program */
	RTvariable offsetX;
	RTvariable offsetY;

	RTvariable params;

	//if (FIELD_NO_ERR != RayField::createOptixInstance(context, output_buffer_obj, seed_buffer_obj) )
	//{
	//	std::cout <<"error in PathIntTissueRayField.createOptixInstance(): RayField.createOptixInstance() returned an error." << "...\n";
	//	return FIELD_ERR;
	//}

	/* Ray generation program */
	char rayGenName[128];
	sprintf(rayGenName, "rayGenerationPathIntTissueRayField");

	if (!RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_rayGeneration, rayGenName, &this->ray_gen_program ), context ))
		return FIELD_ERR;

	/* declare result buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "output_buffer", &output_buffer ), context ))
		return FIELD_ERR;
    /* Render result buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_OUTPUT, &output_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( output_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( output_buffer_obj, sizeof(rayStruct) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize2D( output_buffer_obj, Field::GPU_SUBSET_WIDTH_MAX, Field::GPU_SUBSET_WIDTH_MAX ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( output_buffer, output_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare seed buffer */
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "seed_buffer", &seed_buffer ) , context))
		return FIELD_ERR;
    /* Render seed buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &seed_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( seed_buffer_obj, RT_FORMAT_UNSIGNED_INT ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize2D( seed_buffer_obj, Field::GPU_SUBSET_WIDTH_MAX, Field::GPU_SUBSET_HEIGHT_MAX ) , context))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( seed_buffer, seed_buffer_obj ) , context))
		return FIELD_ERR;

	if (FIELD_NO_ERR!=this->createCPUSimInstance())
	{
		std::cout <<"error in PathTracingRayField.createOptixInstance(): create CPUSimInstance() returned an error." << "...\n";
		return FIELD_ERR;
	}

	/* declare variables */
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "params", &params ), context ))
		return FIELD_ERR;

	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "launch_offsetX", &offsetX ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "launch_offsetY", &offsetY ), context ))
		return FIELD_ERR;

	this->rayParamsPtr->nImmersed=this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);

//	// calc the dimensions of the simulation subset
//	if ( this->rayParamsPtr->width < GPU_SUBSET_WIDTH_MAX ) 
//	{
//		this->rayParamsPtr->GPUSubset_width=this->rayParamsPtr->width;
//	}
//	else
//	{
//		this->rayParamsPtr->GPUSubset_width=GPU_SUBSET_WIDTH_MAX;
//	}
//	if ( this->rayParamsPtr->height < GPU_SUBSET_HEIGHT_MAX ) 
//	{
//		this->rayParamsPtr->GPUSubset_height=this->rayParamsPtr->height;
//	}
//	else
//	{
//		this->rayParamsPtr->GPUSubset_height=GPU_SUBSET_HEIGHT_MAX;
//	}
//
//	//l_offsetX=0;
//	//l_offsetY=0;
////	this->rayParamsPtr->launchOffsetX=0;//l_offsetX;
////	this->rayParamsPtr->launchOffsetY=0;//l_offsetY;

	
	// transfer the dimension of the whole simulation to GPU
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(pathIntTissueRayFieldParams), this->rayParamsPtr), context) )
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
 * \detail initGPUSubset 

 *
 * \param[in] RTcontext &context, RTbuffer &seed_buffer_obj
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
 fieldError PathIntTissueRayField::initGPUSubset(RTcontext &context, RTbuffer &seed_buffer_obj)
{
	RTvariable offsetX;
	RTvariable offsetY;
	RTvariable seed_buffer;

	long long l_offsetX=this->rayParamsPtr->launchOffsetX;
	long long l_offsetY=this->rayParamsPtr->launchOffsetY;

	if (!RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( this->ray_gen_program, "launch_offsetX", &offsetX ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( this->ray_gen_program, "launch_offsetY", &offsetY ) , context))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramQueryVariable( this->ray_gen_program, "seed_buffer", &seed_buffer ), context))
		return FIELD_ERR;

	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetX, sizeof(long long), &l_offsetX) , context))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetY, sizeof(long long), &l_offsetY) , context))
		return FIELD_ERR;

	/* refill seed buffer */
	void *data;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(seed_buffer_obj, &data) , context))
		return FIELD_ERR;
	uint* seeds = reinterpret_cast<uint*>( data );
	RTsize buffer_width, buffer_height;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferGetSize2D(seed_buffer_obj, &buffer_width, &buffer_height) , context))
		return FIELD_ERR;
	uint32_t x[5];
	int seed = (int)time(0);            // random seed
	RandomInit(seed, x);
	for ( unsigned int i = 0; i < (unsigned int)buffer_width*(unsigned int)buffer_height; ++i )
		seeds[i] = (uint)BRandom(x);
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( seed_buffer_obj ) , context))
		return FIELD_ERR;
	
	return FIELD_NO_ERR;
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
fieldError PathIntTissueRayField::initCPUSubset()
{
	clock_t start, end;
	double msecs=0;

	// check wether we will be able to fit all the rays into our raylist. If not some eror happened earlier and we can not proceed...
	if ((this->rayParamsPtr->GPUSubset_width*this->rayParamsPtr->GPUSubset_height)<=this->rayListLength)
	{

		// see if there are any rays to create	
		if (this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width >= 1)
		{
			// start timing
			start=clock();

			std::cout << "initalizing random seed" << "...\n";

			int seed = (int)time(0);            // random seed
			RandomInit(seed, x);

			// create random seeds for all the rays
			for(signed long long j=0;j<this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width;j++)
			{
				this->rayList[j].currentSeed=(uint)BRandom(x);
			}

			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize random seeds of " << this->rayParamsPtr->GPUSubset_width*this->rayParamsPtr->GPUSubset_height << " rays." << "...\n";

			// start timing
			start=clock();

			// create all the rays
		std::cout << "initializing rays on " << numCPU << " cores of CPU." << "...\n";

		omp_set_num_threads(numCPU);

#pragma omp parallel default(shared)
{
			#pragma omp for schedule(dynamic, 50)//schedule(static)//

			for (signed long long j=0; j<this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width; j++)
			{
				unsigned long long jx = j % this->rayParamsPtr->GPUSubset_width;
				unsigned long long iy = (j-jx)/this->rayParamsPtr->GPUSubset_width;
				unsigned long long rayListIndex=jx+iy*GPU_SUBSET_WIDTH_MAX;

				uint32_t x_l[5];
				RandomInit(this->rayList[j].currentSeed, x_l); // seed random generator

				rayStruct rayData;
				rayData.flux=this->rayParamsPtr->flux;
				rayData.depth=0;
				rayData.running=true;
				rayData.currentGeometryID=0;
				rayData.lambda=this->rayParamsPtr->lambda;
				rayData.nImmersed=this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);
				rayData.opl=0;

				// consider offsets of current subset
				unsigned long long jGes=jx+this->rayParamsPtr->launchOffsetX;
				unsigned long long iGes=(iy+this->rayParamsPtr->launchOffsetY);

				rayData.position=this->rayParamsPtr->sourcePos;
				rayData.direction=make_double3(0,0,1);

				rayData.currentSeed=(uint)BRandom(x);
				this->setRay(rayData,(unsigned long long)(jx+iy*GPU_SUBSET_WIDTH_MAX));
			}// end for
} // end omp
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize " << this->rayParamsPtr->GPUSubset_width*this->rayParamsPtr->GPUSubset_height << " rays." << "...\n";

		}
		else if(this->rayParamsPtr->width*this->rayParamsPtr->height<1)
		{
			//not Possible. Report error or set n=-n
			std::cout << "error in PathIntTissueRayField.initCPUInstance: negative raynumber" << "...\n";
			return FIELD_ERR;
		}
		this->update=false;
	}	// end if GPUsubsetwidth*height<rayListLength
	else
	{
		std::cout << "error in PathIntTissueRayField.initCPUInstance: rayList is smaller than simulation subset" << "...\n";
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
};

/**
 * \detail createOptixInstance 

 *
 * \param[in] RTcontext* context, unsigned long long width, unsigned long long height, double3 rayPosStart, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathIntTissueRayField::createOptixInstance(RTcontext* context, unsigned long long width, unsigned long long height, double3 rayPosStart, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference)
{
	RTbuffer   xGrad_buffer_obj;
	RTbuffer   yGrad_buffer_obj;
	RTvariable xGrad_buffer;
	RTvariable yGrad_buffer;
//    RTvariable radiance_ray_type;
    RTvariable epsilon;
	RTvariable max_depth;
	RTvariable min_flux;
	RTprogram  ray_gen_program;
    /* variables for ray gen program */
    RTvariable origin_max;
	RTvariable origin_min;
	//RTvariable number;
	RTvariable launch_width;
	RTvariable launch_height;
	RTvariable l_size_xGrad;
	RTvariable l_size_yGrad;
	RTvariable l_lambda;

	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "xGrad_buffer", &xGrad_buffer ) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "yGrad_buffer", &yGrad_buffer ) );
	/* render buffers */
	RT_CHECK_ERROR2( rtBufferCreate( *context, RT_BUFFER_INPUT, &xGrad_buffer_obj ) );
    RT_CHECK_ERROR2( rtBufferSetFormat( xGrad_buffer_obj, RT_FORMAT_USER ) );
	RT_CHECK_ERROR2( rtBufferSetElementSize( xGrad_buffer_obj, sizeof(double) ) );
	RT_CHECK_ERROR2( rtBufferSetSize1D( xGrad_buffer_obj, size_xGrad ) );
    RT_CHECK_ERROR2( rtVariableSetObject( xGrad_buffer, xGrad_buffer_obj ) );

	RT_CHECK_ERROR2( rtBufferCreate( *context, RT_BUFFER_INPUT, &yGrad_buffer_obj ) );
    RT_CHECK_ERROR2( rtBufferSetFormat( yGrad_buffer_obj, RT_FORMAT_USER ) );
	RT_CHECK_ERROR2( rtBufferSetElementSize( yGrad_buffer_obj, sizeof(double) ) );
	RT_CHECK_ERROR2( rtBufferSetSize1D( yGrad_buffer_obj, size_yGrad ) );
    RT_CHECK_ERROR2( rtVariableSetObject( yGrad_buffer, yGrad_buffer_obj ) );

	/* fill buffers */
	void *data;
	RT_CHECK_ERROR2( rtBufferMap(xGrad_buffer_obj, &data) );
	double* l_xGrad = reinterpret_cast<double*>( data );
	memcpy(l_xGrad, xGrad, size_xGrad*sizeof(double));
	RT_CHECK_ERROR2( rtBufferUnmap( xGrad_buffer_obj ) );

	void *data2;
	RT_CHECK_ERROR2( rtBufferMap(yGrad_buffer_obj, &data2) );
	double* l_yGrad = reinterpret_cast<double*>( data2 );
	memcpy(l_yGrad, yGrad, size_yGrad*sizeof(double));
	RT_CHECK_ERROR2( rtBufferUnmap( yGrad_buffer_obj ) );


	/* Ray generation program */
    RT_CHECK_ERROR2( rtProgramCreateFromPTXFile( *context, this->path_to_ptx_rayGeneration, "rayGeneration_WaveIn", &this->ray_gen_program ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "launch_width", &launch_width ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "launch_height", &launch_height ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "lambda", &l_lambda ) );

	unsigned int l_launch_height, l_launch_width;//numberVar;
	
	//numberVar = n;//(unsigned int)rayListLength/100;
	l_launch_height=this->rayParamsPtr->height;
	l_launch_width=this->rayParamsPtr->width;

//	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "radiance_ray_type", &radiance_ray_type ) );
    RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "scene_epsilon", &epsilon ) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "max_depth", &max_depth ) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "min_flux", &min_flux ) );

//    RT_CHECK_ERROR2( rtVariableSet1ui( radiance_ray_type, 0u ) );
    RT_CHECK_ERROR2( rtVariableSet1f( epsilon, SCENE_EPSILON ) );
	//RT_CHECK_ERROR2( rtVariableSet1i( max_depth, MAX_DEPTH_GPU ) );
	RT_CHECK_ERROR2( rtVariableSet1i( max_depth, MAX_DEPTH_CPU ) );
	RT_CHECK_ERROR2( rtVariableSet1f( min_flux, MIN_FLUX_CPU ) );


	RT_CHECK_ERROR2( rtVariableSetUserData(l_lambda, sizeof(double), &this->rayParamsPtr->lambda) );
	RT_CHECK_ERROR2( rtVariableSet1ui(launch_width, l_launch_width ) );
	RT_CHECK_ERROR2( rtVariableSet1ui(launch_height, l_launch_height ) );

    RT_CHECK_ERROR2( rtContextSetRayGenerationProgram( *context,0, this->ray_gen_program ) );

	return FIELD_NO_ERR;
};

//void PathIntTissueRayField::createCPUSimInstance(unsigned long long nWidth, unsigned long long nHeight,double distance, double3 this->rayParamsPtr->rayDirection, double3 firstRayPosition, double flux, double lambda)
//{
//	//creates a ray list parallel to the x,y plane
//	if ((nWidth*nHeight) <= this->rayListLength)
//	{
//		uint32_t x[5];
//		int seed = (int)time(0);            // random seed
//		RandomInit(seed, x); // seed random generator
//
//		rayStruct rayData;
//		rayData.flux=flux;
//		rayData.depth=0;
//		rayData.direction=this->rayParamsPtr->rayDirection;
//		rayData.position.z=firstRayPosition.z;
//		rayData.running=true;
//		rayData.currentGeometryID=0;
//		rayData.lambda=lambda;
//		rayData.nImmersed=this->rayParamsPtr->nImmersed
//		rayData.opl=0;
//
//		for(unsigned long long i=0;i<nHeight;i++)
//		{
//			for(unsigned long long j=0;j<nWidth;j++)
//			{
//				rayData.position.x=firstRayPosition.x+i*distance;
//				rayData.position.y=firstRayPosition.y+j*distance;
//				this->setRay(rayData,(unsigned long long)(i*nWidth+j));
//			}
//		}
//	}
//};


/**
 * \detail setParamsPtr 
 *
 * \param[in] diffRayFieldParams *paramsPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void PathIntTissueRayField::setParamsPtr(pathIntTissueRayFieldParams *paramsPtr)
{
	this->rayParamsPtr=paramsPtr;
	this->update=true;
};

/**
 * \detail getParamsPtr 
 *
 * \param[in] void
 * 
 * \return diffRayFieldParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
pathIntTissueRayFieldParams* PathIntTissueRayField::getParamsPtr(void)
{
	return this->rayParamsPtr;
};

/**
 * \detail createCPUSimInstance 
 *
 *
 * \param[in] void
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathIntTissueRayField::createCPUSimInstance()
{
	if (this->rayList != NULL)
	{
		delete this->rayList;
		this->rayListLength=0;
		rayList=NULL;
	}
	rayList=(rayStruct*) malloc(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX*sizeof(rayStruct));
	if (!rayList)
	{
		std::cout << "error in PathIntTissueRayField.createLayoutInstance(): memory for rayList could not be allocated" << "...\n";
		return FIELD_ERR;
	}
	this->rayListLength=GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX;

	unsigned int l_launch_width, l_launch_height, l_offsetX, l_offsetY;
	if ( this->rayParamsPtr->width < GPU_SUBSET_WIDTH_MAX ) 
	{
		l_launch_width=this->rayParamsPtr->width;
	}
	else
	{
		l_launch_width=GPU_SUBSET_WIDTH_MAX;
	}
	if ( this->rayParamsPtr->height < GPU_SUBSET_HEIGHT_MAX ) 
	{
		l_launch_height=this->rayParamsPtr->height;
	}
	else
	{
		l_launch_height=GPU_SUBSET_HEIGHT_MAX;
	}

	l_offsetX=0;
	l_offsetY=0;
	this->rayParamsPtr->launchOffsetX=l_offsetX;
	this->rayParamsPtr->launchOffsetY=l_offsetY;

	this->rayParamsPtr->GPUSubset_height=l_launch_height;
	this->rayParamsPtr->GPUSubset_width=l_launch_width;

	return FIELD_NO_ERR;
};

/**
 * \detail createLayoutInstance 
 *
 *
 * \param[in] void
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathIntTissueRayField::createLayoutInstance()
{
	if (this->rayList != NULL)
	{
		delete this->rayList;
		this->rayListLength=0;
		rayList=NULL;
	}
	rayList=(rayStruct*) malloc(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX*sizeof(rayStruct));
	if (!rayList)
	{
		std::cout << "error in PathIntTissueRayField.createLayoutInstance(): memory for rayList could not be allocated. try to reduce ray tiling size" << "...\n";
		return FIELD_ERR;
	}
	this->rayListLength=GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX;

	unsigned int l_launch_width, l_launch_height, l_offsetX, l_offsetY;
	if ( this->rayParamsPtr->widthLayout < GPU_SUBSET_WIDTH_MAX ) 
	{
		l_launch_width=this->rayParamsPtr->widthLayout;
	}
	else
	{
		l_launch_width=GPU_SUBSET_WIDTH_MAX;
	}
	if ( this->rayParamsPtr->heightLayout < GPU_SUBSET_HEIGHT_MAX ) 
	{
		l_launch_height=this->rayParamsPtr->heightLayout;
	}
	else
	{
		l_launch_height=GPU_SUBSET_HEIGHT_MAX;
	}

	l_offsetX=0;
	l_offsetY=0;
	this->rayParamsPtr->launchOffsetX=l_offsetX;
	this->rayParamsPtr->launchOffsetY=l_offsetY;

	this->rayParamsPtr->GPUSubset_height=l_launch_height;
	this->rayParamsPtr->GPUSubset_width=l_launch_width;

	this->rayParamsPtr->totalLaunch_height=this->rayParamsPtr->heightLayout;
	this->rayParamsPtr->totalLaunch_width=this->rayParamsPtr->widthLayout;
	this->rayParamsPtr->width=this->rayParamsPtr->widthLayout;
	this->rayParamsPtr->height=this->rayParamsPtr->heightLayout;

	return FIELD_NO_ERR;

	return FIELD_NO_ERR;
};

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
fieldError PathIntTissueRayField::traceScene(Group &oGroup)
{
	long long index;
	for (index=0; index < this->rayListLength; index++)
	{
	    for(;;) // iterative tracing
		{
			if(!this->rayList[index].running) 
			    break;
			oGroup.trace(rayList[index]);
		}
	}

	return FIELD_NO_ERR;
};

long2 PathIntTissueRayField::calcSubsetDim(void)
{
	unsigned long long width=this->rayParamsPtr->width;
	unsigned long long height=this->rayParamsPtr->height;

	long2 l_GPUSubsetDim;

	// calc launch_width of current launch
	long long restWidth=width-this->rayParamsPtr->launchOffsetX;
	long long restHeight=height-this->rayParamsPtr->launchOffsetY;
	// if the restWidth is smaller than the maximum subset-width. Take restWidth
	if (restWidth < GPU_SUBSET_WIDTH_MAX)
	{
		l_GPUSubsetDim.x=restWidth;
	}
	else
	{
		l_GPUSubsetDim.x=GPU_SUBSET_WIDTH_MAX;
	}
	// if the restHeight is smaller than the maximum subset-width. Take restHeight
	if (restHeight < GPU_SUBSET_HEIGHT_MAX)
	{
		l_GPUSubsetDim.y=restHeight;
	}
	else
	{
		l_GPUSubsetDim.y=GPU_SUBSET_HEIGHT_MAX;
	}
	this->rayParamsPtr->GPUSubset_width=l_GPUSubsetDim.x;
	this->rayParamsPtr->GPUSubset_height=l_GPUSubsetDim.y;
	return l_GPUSubsetDim;
};

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
fieldError PathIntTissueRayField::traceScene(Group &oGroup, bool RunOnCPU)
{
	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;
	// start timing
	start=clock();

//	long2 l_GPUSubsetDim=calcSubsetDim();
//	this->rayParamsPtr->GPUSubset_width=l_GPUSubsetDim.x;
//	this->rayParamsPtr->GPUSubset_height=l_GPUSubsetDim.y;

	if (RunOnCPU)
	{
		std::cout << "tracing on " << numCPU << " cores of CPU." << "...\n";

		if (FIELD_NO_ERR!= initCPUSubset())
		{
			std::cout << "error in GeometrciRayField.traceScene(): initCPUSubset() returned an error." << "...\n";
			return FIELD_ERR;
		}

		std::cout << "starting the actual trace..." << "...\n";		

		omp_set_num_threads(numCPU);

		//int threadCounter[20];
		//for (unsigned int i=0;i<20;i++)
		//	threadCounter[i]=0;


#pragma omp parallel default(shared) //shared(threadCounter)
{
		#pragma omp for schedule(dynamic, 50)
		//for (signed long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		//{
		for (signed long long j=0; j<this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width; j++)
		{
//			threadCounter[omp_get_thread_num()]=threadCounter[omp_get_thread_num()]+1;
			unsigned long long jx = j % this->rayParamsPtr->GPUSubset_width;
			unsigned long long jy = (j-jx)/this->rayParamsPtr->GPUSubset_width;
			unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
			rayStruct test=rayList[rayListIndex];
			for(;;) // iterative tracing
			{
				if(!this->rayList[rayListIndex].running) 
					break;
				oGroup.trace(rayList[rayListIndex]);
			}
			//std::cout << "Iteration " << jy << " running Cur Thread " << omp_get_thread_num() << "Num Threads " << omp_get_num_threads() << "Max Threads " << omp_get_max_threads() << " running" << "...\n";
		}
}
		//for (int i=0;i<20;i++)
		//{
		//	std::cout << "Thread number " << i << " has run " << threadCounter[i] << " times" << "...\n";
		//}
	}
	else
	{
		//RTsize				buffer_width, buffer_height; // get size of output buffer
		void				*data; // pointer to cast output buffer into
 		//rayStruct			*bufferData;

		std::cout << "tracing on GPU." << "...\n";

		initGPUSubset(context, seed_buffer_obj);
		// start current launch
		if (!RT_CHECK_ERROR_NOEXIT( rtContextLaunch2D( (context), 0, this->rayParamsPtr->GPUSubset_width, this->rayParamsPtr->GPUSubset_height), context))//this->rayParamsPtr->launchOffsetX, this->rayParamsPtr->launchOffsetY ) );
			return FIELD_ERR;

		// update scene
//		oGroup.updateOptixInstance(context, mode, lambda);
				
//		RT_CHECK_ERROR_NOEXIT( rtContextLaunch2D( context, 0, width, height ) );
		/* unmap output-buffer */
		//RT_CHECK_ERROR_NOEXIT( rtBufferGetSize2D(output_buffer_obj, &buffer_width, &buffer_height) );
		// recast from Optix RTsize to standard int
		//unsigned long long l_bufferWidth = (unsigned long long)(buffer_width);
		//unsigned long long l_bufferHeight = (unsigned long long)(buffer_height);//static_cast<int>(buffer_height);

		if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(output_buffer_obj, &data) , context))
			return FIELD_ERR;
//			end=clock();
		// cast data pointer to format of the output buffer
		//bufferData=(rayStruct*)data;
		//rayStruct test=bufferData[250];
		//SourceList->setRayList((rayStruct*)data);
		//std::cout << "DEBUG: jx=" << jx << " jy=" << jy << "...\n";
		//copyRayListSubset((rayStruct*)data, l_launchOffset, l_GPUSubsetDim);
		if (FIELD_NO_ERR != copyRayList((rayStruct*)data,this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width) )
		{
			std::cout << "error in PathIntTissueRayField.traceScene(): copyRayList() returned an error" << "...\n";
			return FIELD_NO_ERR;
		}
		
		if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( output_buffer_obj ) , context))
			return FIELD_ERR;
	}
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<" ms to trace " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";

	return FIELD_NO_ERR;
};

/**
 * \detail traceStep 
 *
 * \param[in] Group &oGroup
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathIntTissueRayField::traceStep(Group &oGroup, bool RunOnCPU)
{
	if (!RunOnCPU)
		std::cout << "warning in PathIntTissueRayField.traceStep(): GPU acceleration is not implemented, continuing on CPU anyways..." << "...\n";

	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;
	// start timing
	start=clock();

//	long2 l_GPUSubsetDim=calcSubsetDim();
//	this->rayParamsPtr->GPUSubset_width=l_GPUSubsetDim.x;
//	this->rayParamsPtr->GPUSubset_height=l_GPUSubsetDim.y;
	std::cout << "tracing on " << numCPU << " cores of CPU." << "...\n";

//#pragma omp parallel default(shared)
//{
//		#pragma omp for schedule(dynamic, 50)
		for (signed long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		{
			//int id;
			//id = omp_get_thread_num();

			//printf("Hello World from thread %d\n", id);

			for (signed long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
				rayStruct test=rayList[rayListIndex];
				if (this->rayList[rayListIndex].running) 
				{
					this->traceRay(rayList[rayListIndex]);
				}
			}
		}
//}

	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<" ms to trace " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";

	return FIELD_NO_ERR;
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
fieldError PathIntTissueRayField::doSim(Group &oGroup, simAssParams &params, bool &simDone)
{
	unsigned long long width=this->getParamsPtr()->totalLaunch_width;//SourceListPtrPtr->getParamsPtr()->width*SourceListPtrPtr->getParamsPtr()->nrRayDirections.x*SourceListPtrPtr->getParamsPtr()->nrRayDirections.y;
	unsigned long long height=this->getParamsPtr()->totalLaunch_height;//SourceListPtrPtr->getParamsPtr()->height;

	unsigned long long roughNrOfSubsets=std::floorf(width*height/(this->getSubsetWidthMax()*this->getSubsetHeightMax()))+1;


	std::cout << "****************************************************** " << "...\n";
	std::cout << "starting subset.......... " << "...\n";
	std::cout << "...\n";
	/***********************************************
	/	trace rays
	/***********************************************/

	long2 l_GPUSubsetDim=this->calcSubsetDim();

	if (FIELD_NO_ERR != this->traceScene(oGroup, params.RunOnCPU) )//, context, output_buffer_obj, seed_buffer_obj) )
	{
		std::cout << "error in PathIntTissueRayField.doSim(): PathIntTissueRayField.traceScene() returned an error" << "...\n";
		return FIELD_ERR;
	}
//	this->subsetCounter++;
//	Field::callbackProgress(Field::p2ProgCallbackObject, floorf(this->subsetCounter*100/roughNrOfSubsets));

	// increment x-offset
	this->getParamsPtr()->launchOffsetX=this->getParamsPtr()->launchOffsetX+this->getParamsPtr()->GPUSubset_width;				
	if (this->getParamsPtr()->launchOffsetX>width-1)
	{
		// increment y-offset
		this->getParamsPtr()->launchOffsetY=this->getParamsPtr()->launchOffsetY+this->getParamsPtr()->GPUSubset_height;
		// reset x-offset
		this->getParamsPtr()->launchOffsetX=0;
		if (this->getParamsPtr()->launchOffsetY>height-1)
			simDone=true;
	}

	tracedRayNr=tracedRayNr+l_GPUSubsetDim.x*l_GPUSubsetDim.y;
	std::cout << " " << tracedRayNr <<" out of " << width*height << " rays traced in total" << "...\n";

	if (simDone)
	{
		if (!params.RunOnCPU)
		{
			// clean up
			if (!RT_CHECK_ERROR_NOEXIT( rtContextDestroy( context ), context ))
				return FIELD_ERR;
		}
	}
	return FIELD_NO_ERR;
};

/**
 * \detail writeData2File 

 *
 * \param[in] FILE *hFile, rayDataOutParams outParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
//fieldError PathIntTissueRayField::writeData2File(FILE *hFile, rayDataOutParams outParams)
//{
//	writeGeomRayData2File(hFile, this->rayList, this->rayListLength, outParams);
//	return FIELD_NO_ERR;
//};

/**
 * \detail write2TextFile 

 *
 * \param[in] char* filename
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathIntTissueRayField::write2TextFile(char* filename, detParams &oDetParams)
{
//	char t_filename[512];
//	sprintf(t_filename, "%s%sPathIntTissueRayField_%i.txt", filename, PATH_SEPARATOR, oDetParams.subSetNr);

	FILE* hFileOut;
	char t_filename[512];
	sprintf(t_filename, "%s%s%i%s", OUTPUT_FILEPATH, PATH_SEPARATOR, oDetParams.subSetNr, oDetParams.filenamePtr);
	hFileOut = fopen( t_filename, "w" ) ;
	if (!hFileOut)
	{
		std::cout << "error in PathIntTissueRayField.write2TextFile(): could not open output file: " << filename << "...\n";
		return FIELD_ERR;
	}
	if (1) //(oDetParams.reduceData==1)
	{
		for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		{
			for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
				if ((rayList[rayListIndex].currentGeometryID==oDetParams.geomID) || (oDetParams.geomID=-1))
				{
					// write the data in row major format, where width is the size of one row and height is the size of one coloumn
					// if the end of a row is reached append a line feed 
					fprintf(hFileOut, "%.20lf ;%.20lf ;%.20lf; %.20lf ;%.20lf ;%.20lf ;%.20lf ;%.20lf; %i \n", rayList[rayListIndex].position.x, rayList[rayListIndex].position.y, rayList[rayListIndex].position.z, rayList[rayListIndex].direction.x, rayList[rayListIndex].direction.y, rayList[rayListIndex].direction.z, rayList[rayListIndex].flux, rayList[rayListIndex].opl, rayList[rayListIndex].currentGeometryID);
				}
			}
		}
	}
	else
	{
		for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		{
			for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;

				if ((rayList[rayListIndex].currentGeometryID==oDetParams.geomID) || (oDetParams.geomID=-1))
				{
					// write the data in row major format, where width is the size of one row and height is the size of one coloumn
					// if the end of a row is reached append a line feed 
					fprintf(hFileOut, "%.20lf ;%.20lf ;%.20lf; %.20lf ;%.20lf ;%.20lf; %.20lf; %.20lf; %i; %i; %.20lf; \n", rayList[rayListIndex].position.x, rayList[rayListIndex].position.y, rayList[rayListIndex].position.z, rayList[rayListIndex].direction.x, rayList[rayListIndex].direction.y, rayList[rayListIndex].direction.z, rayList[rayListIndex].flux, rayList[rayListIndex].opl, rayList[rayListIndex].currentGeometryID, rayList[rayListIndex].depth, rayList[rayListIndex].nImmersed);
				}
			}
		}
	} // end if reducedData
	fclose(hFileOut);
	return FIELD_NO_ERR;
};

/**
 * \detail write2MatFile
 *
 * saves the field to a mat file
 *
 * \param[in] char* filename
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
//fieldError PathIntTissueRayField::write2MatFile(char* filename, detParams &oDetParams)
//{
//	MatlabInterface oMatInterface;
//	mxArray *mat_RayField = NULL, *mat_posX=NULL, *mat_posY=NULL, *mat_posZ=NULL, *mat_dirX=NULL, *mat_dirY=NULL, *mat_dirZ=NULL, *mat_flux=NULL, *mat_j=NULL;
//
//	mat_RayField = mxCreateDoubleMatrix(this->rayListLength, 7, mxREAL);
//	engPutVariable(oMatInterface.getEnginePtr(), "RayField", mat_RayField);
//	mat_posX = mxCreateDoubleScalar(rayList[0].position.x);
//	mat_posY = mxCreateDoubleScalar(rayList[0].position.y);
//	mat_posZ = mxCreateDoubleScalar(rayList[0].position.z);
//	mat_dirX = mxCreateDoubleScalar(rayList[0].direction.x);
//	mat_dirY = mxCreateDoubleScalar(rayList[0].direction.y);
//	mat_dirZ = mxCreateDoubleScalar(rayList[0].direction.z);
//	mat_flux = mxCreateDoubleScalar(rayList[0].flux);
//	mat_j = mxCreateNumericMatrix(1, 1,  mxUINT64_CLASS, mxREAL);
//	int result;
//	// define loop counter in matlab
//	result=engEvalString(oMatInterface.getEnginePtr(), "j=0;");
//	for (unsigned long long j=0; j<this->rayListLength; j++)
//	{
//		if ((rayList[j].currentGeometryID==oDetParams.geomID) || (oDetParams.geomID=-1))
//		{
//			// put the loop index to matlab
//
//			// write data to matlab
//			memcpy((char *) mxGetPr(mat_posX), (char *) (&(rayList[j].position.x)), sizeof(double));
//			//mxSetPr(mat_posX, &(rayList[j].position.x));
//			engPutVariable(oMatInterface.getEnginePtr(), "posX", mat_posX);
//			//mxSetPr(mat_posY, &(rayList[j].position.y));
//			memcpy((char *) mxGetPr(mat_posY), (char *) (&(rayList[j].position.y)), sizeof(double));
//			engPutVariable(oMatInterface.getEnginePtr(), "posY", mat_posY);
//			//mxSetPr(mat_posZ, &(rayList[j].position.z));
//			memcpy((char *) mxGetPr(mat_posZ), (char *) (&(rayList[j].position.z)), sizeof(double));
//			engPutVariable(oMatInterface.getEnginePtr(), "posZ", mat_posZ);
//			//mxSetPr(mat_dirX, &(rayList[j].direction.x));
//			memcpy((char *) mxGetPr(mat_dirX), (char *) (&(rayList[j].direction.x)), sizeof(double));
//			engPutVariable(oMatInterface.getEnginePtr(), "dirX", mat_dirX);
//			//mxSetPr(mat_dirY, &(rayList[j].direction.y));
//			memcpy((char *) mxGetPr(mat_dirY), (char *) (&(rayList[j].direction.y)), sizeof(double));
//			engPutVariable(oMatInterface.getEnginePtr(), "dirY", mat_dirY);
//			//mxSetPr(mat_dirZ, &(rayList[j].direction.z));
//			memcpy((char *) mxGetPr(mat_dirZ), (char *) (&(rayList[j].direction.z)), sizeof(double));
//			engPutVariable(oMatInterface.getEnginePtr(), "dirZ", mat_dirZ);
//			//mxSetPr(mat_flux, &(rayList[j].flux));
//			memcpy((char *) mxGetPr(mat_flux), (char *) (&(rayList[j].flux)), sizeof(double));
//			engPutVariable(oMatInterface.getEnginePtr(), "flux", mat_flux);
//			// increment loop counter in matlab
//			result=engEvalString(oMatInterface.getEnginePtr(), "j=j+1;");
//			result=engEvalString(oMatInterface.getEnginePtr(), "RayField(j,1)=posX;");
//			result=engEvalString(oMatInterface.getEnginePtr(), "RayField(j,2)=posY;");
//			result=engEvalString(oMatInterface.getEnginePtr(), "RayField(j,3)=posZ;");
//			result=engEvalString(oMatInterface.getEnginePtr(), "RayField(j,4)=dirX;");
//			result=engEvalString(oMatInterface.getEnginePtr(), "RayField(j,5)=dirY;");
//			result=engEvalString(oMatInterface.getEnginePtr(), "RayField(j,6)=dirZ;");
//			result=engEvalString(oMatInterface.getEnginePtr(), "RayField(j,7)=flux;");
//			
//		}
//	}
//	/* save the struct into a .mat file */
//	char t_filename[512];
//	sprintf(t_filename, "%s%sPathIntTissueRayField", filename, PATH_SEPARATOR);
//
//	char saveCommand[564];
//	sprintf(saveCommand, "save %s RayField;", t_filename);
//	result=engEvalString(oMatInterface.getEnginePtr(), saveCommand);
//	/*
//	 * We're done! Free memory, close MATLAB engine and exit.
//	 */
//	mxDestroyArray(mat_RayField);
//	mxDestroyArray(mat_posX);
//	mxDestroyArray(mat_posY);
//	mxDestroyArray(mat_posZ);
//	mxDestroyArray(mat_dirX);
//	mxDestroyArray(mat_dirY);
//	mxDestroyArray(mat_dirZ);
//	mxDestroyArray(mat_flux);
//
//	return FIELD_NO_ERR;
//};


/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &field, vector<Field*> &fieldVec
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  PathIntTissueRayField::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams)
{
	Parser_XML l_parser;

	if (!l_parser.attrByNameToDouble(field, "root.x", this->rayParamsPtr->translation.x))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): root.x is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "root.y", this->rayParamsPtr->translation.y))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): root.y is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "root.z", this->rayParamsPtr->translation.z))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): root.z is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "tilt.x", this->getParamsPtr()->tilt.x))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): tilt.x is not defined" << "...\n";
		return FIELD_ERR;
	}
	this->getParamsPtr()->tilt.x=this->getParamsPtr()->tilt.x/360*2*PI;
	if (!l_parser.attrByNameToDouble(field, "tilt.y", this->getParamsPtr()->tilt.y))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): tilt.y is not defined" << "...\n";
		return FIELD_ERR;
	}
	this->getParamsPtr()->tilt.y=this->getParamsPtr()->tilt.y/360*2*PI;
	if (!l_parser.attrByNameToDouble(field, "tilt.z", this->getParamsPtr()->tilt.z))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): tilt.z is not defined" << "...\n";
		return FIELD_ERR;
	}
	this->getParamsPtr()->tilt.z=this->getParamsPtr()->tilt.z/360*2*PI;
	if (!l_parser.attrByNameToDouble(field, "power", this->getParamsPtr()->flux))
	{
		std::cout << "error in RayField.parseXml(): power is not defined" << "...\n";
		return FIELD_ERR;
	}
	unsigned long l_val;
	if (!l_parser.attrByNameToLong(field, "width", l_val))
	{
		std::cout << "error in RayField.parseXml(): width is not defined" << "...\n";
		return FIELD_ERR;
	}
	this->getParamsPtr()->width=l_val;
	if (!l_parser.attrByNameToLong(field, "height", l_val))
	{
		std::cout << "error in RayField.parseXml(): height is not defined" << "...\n";
		return FIELD_ERR;
	}
	this->getParamsPtr()->height=l_val;
	if (!l_parser.attrByNameToLong(field, "widthLayout", l_val))
	{
		std::cout << "error in RayField.parseXml(): widthLayout is not defined" << "...\n";
		return FIELD_ERR;
	}
	this->getParamsPtr()->widthLayout=l_val;
	if (!l_parser.attrByNameToLong(field, "heightLayout", l_val))
	{
		std::cout << "error in RayField.parseXml(): heightLayout is not defined" << "...\n";
		return FIELD_ERR;
	}
	this->getParamsPtr()->heightLayout=l_val;

	if (!l_parser.attrByNameToDouble(field, "meanFreePath", this->rayParamsPtr->meanFreePath))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): meanFreePath is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "anisotropy", this->rayParamsPtr->anisotropy))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): anisotropy is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "sourcePos.x", this->rayParamsPtr->sourcePos.x))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): sourcePos.x is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "sourcePos.y", this->rayParamsPtr->sourcePos.y))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): sourcePos.y is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "sourcePos.z", this->rayParamsPtr->sourcePos.z))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): sourcePos.z is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "volumeWidth.x", this->rayParamsPtr->volumeWidth.x))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): volumeWidth.x is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "volumeWidth.y", this->rayParamsPtr->volumeWidth.y))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): volumeWidth.y is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "volumeWidth.z", this->rayParamsPtr->volumeWidth.z))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): volumeWidth.z is not defined" << "...\n";
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "lambda", this->getParamsPtr()->lambda))
	{
		std::cout << "error in PathIntTissueRayField.parseXml(): lambda is not defined" << "...\n";
		return FIELD_ERR;
	}

	this->rayParamsPtr->totalLaunch_height=this->rayParamsPtr->height;
	this->rayParamsPtr->totalLaunch_width=this->rayParamsPtr->width;
	this->rayParamsPtr->nrRayDirections=make_ulong2(1,1);

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

bool PathIntTissueRayField::traceRay(rayStruct &ray)
{
	ray.depth++;
	if (ray.depth > 10)
		ray.running=false;
	double l_dist=this->rayParamsPtr->meanFreePath;
	ray.position=ray.position+l_dist*ray.direction;
	ray.opl=ray.nImmersed*l_dist;

	rotateRay(&ray.direction, make_double3(0.2, 0.0, 0.0));
	return true;
}