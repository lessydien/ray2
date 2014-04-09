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

/**\file GeometricRenderField.cpp
* \brief Rayfield for geometric raytracing
* 
*           
* \author Mauch
*/
#include <omp.h>
#include "GeometricRenderField.h"
#include "..\myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "..\Geometry.h"
#include "math.h"
#include "..\randomGenerator.h"
#include "..\Converter.h"
#include "..\DetectorLib.h"
#include "..\MaterialLib.h"
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
fieldError GeometricRenderField::setLambda(double lambda)
{
	this->renderFieldParamsPtr->lambda=lambda;
	return FIELD_NO_ERR;
}

void GeometricRenderField::setSimMode(SimMode &simMode)
{
	simMode=SIM_GEOM_RENDER;
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
fieldError GeometricRenderField::setMaterial(Material *oMaterialPtr, int index)
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
Material* GeometricRenderField::getMaterial(int index)
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
fieldError GeometricRenderField::setMaterialListLength(int length)
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
int GeometricRenderField::getMaterialListLength(void)
{
	return this->materialListLength;
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
void GeometricRenderField::setPathToPtx(char* path)
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
const char* GeometricRenderField::getPathToPtx(void)
{
	return this->path_to_ptx_rayGeneration;
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
fieldError GeometricRenderField::createLayoutInstance()
{
	if (this->layoutRayList != NULL)
	{
		delete this->layoutRayList;
		this->rayListLength=0;
		layoutRayList=NULL;
	}
	layoutRayList=(geomRenderRayStruct*) malloc(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX*sizeof(geomRenderRayStruct));
	if (!layoutRayList)
	{
		std::cout << "error in GeometricRenderField.createLayoutInstance(): memory for rayList could not be allocated. try to reduce ray tiling size" << "...\n";
		return FIELD_ERR;
	}
	this->rayListLength=GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX;

	unsigned int l_launch_width, l_launch_height, l_offsetX, l_offsetY;
    if ( this->renderFieldParamsPtr->widthLayout*this->renderFieldParamsPtr->heightLayout*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y< GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX ) 
	{
		l_launch_width=this->renderFieldParamsPtr->widthLayout*this->renderFieldParamsPtr->heightLayout*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y;
	}
	else
	{
		l_launch_width=GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX;
	}
	l_launch_height=1;

	l_offsetX=0;
	l_offsetY=0;
	this->renderFieldParamsPtr->launchOffsetX=l_offsetX;
	this->renderFieldParamsPtr->launchOffsetY=l_offsetY;

	this->renderFieldParamsPtr->GPUSubset_height=1;
	this->renderFieldParamsPtr->GPUSubset_width=l_launch_width;

	this->renderFieldParamsPtr->totalLaunch_height=1;
	this->renderFieldParamsPtr->totalLaunch_width=this->renderFieldParamsPtr->widthLayout*this->renderFieldParamsPtr->heightLayout*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y;
    this->renderFieldParamsPtr->layout_height=this->renderFieldParamsPtr->heightLayout;
    this->renderFieldParamsPtr->layout_width=this->renderFieldParamsPtr->widthLayout;
    this->renderFieldParamsPtr->widthLayout=this->renderFieldParamsPtr->widthLayout*this->renderFieldParamsPtr->heightLayout*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y;
    this->renderFieldParamsPtr->heightLayout=1;

	return FIELD_NO_ERR;

	return FIELD_NO_ERR;
};

/**
 * \detail initCPUSubset 
 *
 * \param[in] void
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRenderField::initCPUSubset()
{
	clock_t start, end;
	double msecs=0;
	// check wether we will be able to fit all the rays into our raylist. If not some eror happened earlier and we can not proceed...
	if ((this->renderFieldParamsPtr->GPUSubset_width)<=this->rayListLength)
	{
		// calc the dimensions of the subset
//		long2 l_GPUSubsetDim=calcSubsetDim();
		
		// see if there are any rays to create	
		//if (this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width >= 1)
		if (this->renderFieldParamsPtr->GPUSubset_width*this->renderFieldParamsPtr->GPUSubset_height >= 1)
		{
			// width of ray field in physical dimension
			double physWidth=this->renderFieldParamsPtr->rayPosEnd.x-this->renderFieldParamsPtr->rayPosStart.x;
			// height of ray field in physical dimension
			double physHeight=this->renderFieldParamsPtr->rayPosEnd.y-this->renderFieldParamsPtr->rayPosStart.y;
			// calc centre of ray field 
			double2 rayFieldCentre=make_double2(this->renderFieldParamsPtr->rayPosStart.x+physWidth/2,this->renderFieldParamsPtr->rayPosStart.y+physHeight/2);

			// start timing
			start=clock();

			std::cout << "initalizing random seed" << "...\n";

			int seed = (int)time(0);            // random seed
			RandomInit(seed, x);

			// create random seeds for all the rays
			std::cout << "initializing rays on " << numCPU << " cores of CPU." << "...\n";

			for(signed long long jx=0;jx<this->renderFieldParamsPtr->GPUSubset_width;jx++)
			{
				this->layoutRayList[jx].currentSeed=(uint)BRandom(x);
			}
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize random seeds of " << this->renderFieldParamsPtr->GPUSubset_width << " rays." << "...\n";

			// start timing
			start=clock();

			// create all the rays
			omp_set_num_threads(numCPU);

#pragma omp parallel default(shared)
{
			#pragma omp for schedule(dynamic, 50)//schedule(static)//
			// create all the rays
			for(signed long long jx=0;jx<this->renderFieldParamsPtr->GPUSubset_width;jx++)
			{
				uint32_t x_l[5];
				RandomInit(this->layoutRayList[jx].currentSeed, x_l); // seed random generator

				long long index=0; // loop counter for random rejection method
				double r; // variables for creating random number inside an ellipse

				// increment of rayposition in x and y in case of GridRect definition 
				double deltaW=0;
				double deltaH=0;
				// increment radial ( along x- and y ) and angular direction in GridRad definition
				double deltaRx=0;
				double deltaRy=0;
				double deltaPhi=0;
				// radius in dependence of phi when calculating GRID_RAD
				double R=0;

				// declar variables for randomly distributing ray directions via an importance area
				double2 impAreaHalfWidth;
				double3 dirImpAreaCentre, tmpPos, impAreaRoot, rayAngleCentre,impAreaAxisX,impAreaAxisY;
				double impAreaX, impAreaY, theta;

				geomRenderRayStruct rayData;

				rayData.depth=0;	
				rayData.position.z=this->renderFieldParamsPtr->rayPosStart.z;
				rayData.running=true;
				rayData.currentGeometryID=0;
				rayData.lambda=this->renderFieldParamsPtr->lambda;
				rayData.nImmersed=1;//this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);
				rayData.flux=1;
                rayData.cumFlux=0;
                rayData.secondary=false;
                rayData.secondary_nr=0;

				// map on one dimensional index
				unsigned long long iGes=jx+this->renderFieldParamsPtr->launchOffsetX+this->renderFieldParamsPtr->launchOffsetY*this->renderFieldParamsPtr->width*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y;

				// calc position indices from 1D index
				unsigned long long iPosX=floorf(iGes/(this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y));
				unsigned long long iPosY=floorf(iPosX/this->renderFieldParamsPtr->layout_width);
				iPosX=iPosX % this->renderFieldParamsPtr->layout_width;

				// calc direction indices from 1D index
				unsigned long long iDirX=(iGes-iPosX*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y-iPosY*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y*this->renderFieldParamsPtr->width) % this->renderFieldParamsPtr->nrRayDirections.x;
				unsigned long long iDirY=floorf((iGes-iPosX*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y-iPosY*this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y*this->renderFieldParamsPtr->width)/this->renderFieldParamsPtr->nrRayDirections.x);

				// declare variables for placing a ray randomly inside an ellipse
				double ellipseX;
				double ellipseY;
				double3 exApt;
				double3 eyApt;

				// create rayposition in local coordinate system according to distribution type
				rayData.position.z=0; // all rays start at z=0 in local coordinate system
				// calc increment along x- and y-direction
				if (this->renderFieldParamsPtr->width>0)
					deltaW= (physWidth)/(this->renderFieldParamsPtr->layout_width);
				if (this->renderFieldParamsPtr->height>0)
					// multiple directions per point are listed in y-direction. Therefore the physical height of the rayfield is different from the height of the ray list. This has to be considered here...
					deltaH= (physHeight)/(this->renderFieldParamsPtr->layout_height);
				rayData.position.x=this->renderFieldParamsPtr->rayPosStart.x+deltaW/2+iPosX*deltaW;
				rayData.position.y=this->renderFieldParamsPtr->rayPosStart.y+deltaH/2+iPosY*deltaH;

				// transform rayposition into global coordinate system
				rayData.position=this->renderFieldParamsPtr->Mrot*rayData.position+this->renderFieldParamsPtr->translation;

				double2 rayAngleHalfWidth, phi;

				aimRayTowardsImpArea(rayData.direction, rayData.position, this->renderFieldParamsPtr->importanceAreaRoot, this->renderFieldParamsPtr->importanceAreaHalfWidth, this->renderFieldParamsPtr->importanceAreaTilt, this->renderFieldParamsPtr->importanceAreaApertureType, rayData.currentSeed);

				rayData.currentSeed=(uint)BRandom(x);

				this->setRay(rayData,(unsigned long long)(jx));
				//increment directions counter
			}
} // end omp
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize " << this->renderFieldParamsPtr->GPUSubset_width << " rays." << "...\n";
		}

		else if(this->renderFieldParamsPtr->width*this->renderFieldParamsPtr->height<1)
		{
			//not Possible. Report error or set n=-n
			std::cout << "error in GeometricRenderField.initCPUInstance: negative raynumber" << "...\n";
		}
		this->update=false;
	}	// end if GPUsubsetwidth*height<rayListLength
	else
	{
		std::cout << "error in GeometricRenderField.initCPUInstance: rayList is smaller than simulation subset" << "...\n";
		return FIELD_ERR;
	}
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
fieldError GeometricRenderField::traceStep(Group &oGroup, bool RunOnCPU)
{
	if (!RunOnCPU)
		std::cout << "warning in GeometricRenderField.traceStep(): GPU acceleration is not implemented, continuing on CPU anyways..." << "...\n";

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
		for (signed long long jy=0; jy<this->renderFieldParamsPtr->GPUSubset_height; jy++)
		{
			//int id;
			//id = omp_get_thread_num();

			//printf("Hello World from thread %d\n", id);

			for (signed long long jx=0; jx<this->renderFieldParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
				rayStruct test=layoutRayList[rayListIndex];
				if (this->layoutRayList[rayListIndex].running) 
					oGroup.trace(layoutRayList[rayListIndex]);
			}
		}
//}

	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<" ms to trace " << this->renderFieldParamsPtr->GPUSubset_height*this->renderFieldParamsPtr->GPUSubset_width << " rays." << "...\n";

	return FIELD_NO_ERR;
};

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
unsigned long long GeometricRenderField::getRayListLength(void)
{
	return this->rayListLength;
};

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
fieldError GeometricRenderField::initLayout(Group &oGroup, simAssParams &params)
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
 * \detail setRay 

 *
 * \param[in] rayStruct ray, unsigned long long index
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRenderField::setRay(geomRenderRayStruct ray, unsigned long long index)
{
	if (index <= this->rayListLength)
	{
		layoutRayList[index]=ray;
		return FIELD_NO_ERR;
	}
	else
	{
		return FIELD_INDEXOUTOFRANGE_ERR;
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
geomRenderRayStruct* GeometricRenderField::getRayList(void)
{
	return &layoutRayList[0];	
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
rayStruct* GeometricRenderField::getRay(unsigned long long index)
{
	if (index <= this->rayListLength)
	{
		return &layoutRayList[index];	
	}
	else
	{
		return 0;
	}
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
fieldError GeometricRenderField::initSimulation(Group &oGroup, simAssParams &params)
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
};

fieldError GeometricRenderField::createOptiXContext()
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
		cout << "error in GeometricRenderField.createOptixContext(): creating miss program from ptx at " << path_to_ptx << " failed." << endl;
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
 * \detail createOptixInstance 

 *
 * \param[in] RTcontext &context
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRenderField::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
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
	//	std::cout <<"error in GeometricRenderField.createOptixInstance(): RayField.createOptixInstance() returned an error." << "...\n";
	//	return FIELD_ERR;
	//}

	/* Ray generation program */
	char rayGenName[128];
	sprintf(rayGenName, "rayGeneration_geomRender");

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
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( output_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize2D( output_buffer_obj, GPU_SUBSET_WIDTH_MAX, GPU_SUBSET_WIDTH_MAX ), context ))
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
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize2D( seed_buffer_obj, GPU_SUBSET_WIDTH_MAX, GPU_SUBSET_HEIGHT_MAX ) , context))
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

	this->renderFieldParamsPtr->nImmersed=this->materialList[0]->calcSourceImmersion(this->renderFieldParamsPtr->lambda);

	// transfer the dimension of the whole simulation to GPU
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(renderFieldParams), this->renderFieldParamsPtr), context) )
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
 fieldError GeometricRenderField::initGPUSubset(RTcontext &context, RTbuffer &seed_buffer_obj)
{
	RTvariable offsetX;
	RTvariable offsetY;
	RTvariable seed_buffer;

	long long l_offsetX=this->renderFieldParamsPtr->launchOffsetX;
	long long l_offsetY=this->renderFieldParamsPtr->launchOffsetY;

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
geomRenderRayStruct GeometricRenderField::createRay(unsigned long long jx, unsigned long long jy, unsigned long long jRay)
{
	clock_t start, end;
	double msecs=0;

    geomRenderRayStruct ray;

	// width of ray field in physical dimension
	double physWidth=this->renderFieldParamsPtr->rayPosEnd.x-this->renderFieldParamsPtr->rayPosStart.x;
	// height of ray field in physical dimension
	double physHeight=this->renderFieldParamsPtr->rayPosEnd.y-this->renderFieldParamsPtr->rayPosStart.y;
	// increment of rayposition in x and y in case of GridRect definition 
	double deltaW=0;
	double deltaH=0;
	// calc centre of ray field 
	double2 rayFieldCentre=make_double2(this->renderFieldParamsPtr->rayPosStart.x+physWidth/2,this->renderFieldParamsPtr->rayPosStart.y+physHeight/2);
	// declar variables for randomly distributing ray directions via an importance area
	double2 impAreaHalfWidth;
	double3 dirImpAreaCentre, tmpPos, impAreaRoot;
	double impAreaX, impAreaY, r, theta;
	double3 impAreaAxisX, impAreaAxisY;

	// start timing
	start=clock();

    // create seed
    ray.currentSeed=(uint)BRandom(x);


	long long index=0; // loop counter for random rejection method

	ray.flux=1;
    ray.cumFlux=0;
    ray.secondary=false;
    ray.secondary_nr=0;
	ray.depth=0;
	ray.running=true;
	ray.currentGeometryID=0;
	ray.lambda=this->renderFieldParamsPtr->lambda;
	ray.nImmersed=this->materialList[0]->calcSourceImmersion(this->renderFieldParamsPtr->lambda);
	ray.opl=0;

	// declare variables for placing a ray randomly inside an ellipse
	double ellipseX;
	double ellipseY;
	double3 exApt;
	double3 eyApt;

	// create rayposition in local coordinate system according to distribution type
	ray.position.z=0; // all rays start at z=0 in local coordinate system
	// calc increment along x- and y-direction
	if (this->renderFieldParamsPtr->width>0)
		deltaW= (physWidth)/double(this->renderFieldParamsPtr->width);
    else
    {
        cout << "error in GeometricRenderField.createRay: negative width is not allowed. \n";
        ray.running=false;
    }
	if (this->renderFieldParamsPtr->height>0)
		// multiple directions per point are listed in y-direction. Therefore the physical height of the rayfield is different from the height of the ray list. This has to be considered here...
		deltaH= (physHeight)/double(this->renderFieldParamsPtr->height);
    else
    {
        cout << "error in GeometricRenderField.createRay: negative width is not allowed. \n";
        ray.running=false;
    }
	ray.position.x=this->renderFieldParamsPtr->rayPosStart.x+deltaW/2+jx*deltaW;
	ray.position.y=this->renderFieldParamsPtr->rayPosStart.y+deltaH/2+jy*deltaH;

	if(this->renderFieldParamsPtr->width*this->renderFieldParamsPtr->height==1)
	{
		ray.position=this->renderFieldParamsPtr->rayPosStart;
	}
	// transform rayposition into global coordinate system
	ray.position=this->renderFieldParamsPtr->Mrot*ray.position+this->renderFieldParamsPtr->translation;

    // create ray direction 
	aimRayTowardsImpArea(ray.direction, ray.position, this->renderFieldParamsPtr->importanceAreaRoot, this->renderFieldParamsPtr->importanceAreaHalfWidth, this->renderFieldParamsPtr->importanceAreaTilt, this->renderFieldParamsPtr->importanceAreaApertureType, ray.currentSeed);

    // save current seed to ray
	ray.currentSeed=x[4];//(uint)BRandom(x);

	return ray;
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
fieldError GeometricRenderField::createOptixInstance(RTcontext* context, unsigned long long width, unsigned long long height, double3 rayPosStart, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference)
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
    RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "origin_max", &origin_max ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "origin_min", &origin_min ) );
	//RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "number", &number ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "launch_width", &launch_width ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "launch_height", &launch_height ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "size_xGrad", &l_size_xGrad ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "size_yGrad", &l_size_yGrad ) );
	RT_CHECK_ERROR2( rtProgramDeclareVariable( this->ray_gen_program, "lambda", &l_lambda ) );

	double3 origin_maxVar, origin_minVar;
	unsigned int l_launch_height, l_launch_width;//numberVar;
	
	origin_maxVar.x= end.x;	
	origin_maxVar.y=end.y;	
	origin_maxVar.z=end.z;	
	origin_minVar.x=this->renderFieldParamsPtr->rayPosStart.x;
	origin_minVar.y=this->renderFieldParamsPtr->rayPosStart.y;
	origin_minVar.z=this->renderFieldParamsPtr->rayPosStart.z;

	//numberVar = n;//(unsigned int)rayListLength/100;
	l_launch_height=this->renderFieldParamsPtr->height;
	l_launch_width=this->renderFieldParamsPtr->width;

//	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "radiance_ray_type", &radiance_ray_type ) );
    RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "scene_epsilon", &epsilon ) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "max_depth", &max_depth ) );
	RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "min_flux", &min_flux ) );

//    RT_CHECK_ERROR2( rtVariableSet1ui( radiance_ray_type, 0u ) );
    RT_CHECK_ERROR2( rtVariableSet1f( epsilon, SCENE_EPSILON ) );
	//RT_CHECK_ERROR2( rtVariableSet1i( max_depth, MAX_DEPTH_GPU ) );
	RT_CHECK_ERROR2( rtVariableSet1i( max_depth, MAX_DEPTH_CPU ) );
	RT_CHECK_ERROR2( rtVariableSet1f( min_flux, MIN_FLUX_CPU ) );


	RT_CHECK_ERROR2( rtVariableSetUserData(origin_max, sizeof(double3), &origin_maxVar) );
	RT_CHECK_ERROR2( rtVariableSetUserData(origin_min, sizeof(double3), &origin_minVar) );
	RT_CHECK_ERROR2( rtVariableSetUserData(l_lambda, sizeof(double), &this->renderFieldParamsPtr->lambda) );
	RT_CHECK_ERROR2( rtVariableSet1ui(l_size_xGrad, size_xGrad) );
	RT_CHECK_ERROR2( rtVariableSet1ui(l_size_yGrad, size_yGrad) );
//	RT_CHECK_ERROR2( rtVariableSet1ui(number, numberVar ) );
	RT_CHECK_ERROR2( rtVariableSet1ui(launch_width, l_launch_width ) );
	RT_CHECK_ERROR2( rtVariableSet1ui(launch_height, l_launch_height ) );

    RT_CHECK_ERROR2( rtContextSetRayGenerationProgram( *context,0, this->ray_gen_program ) );

	return FIELD_NO_ERR;
};

//void GeometricRenderField::createCPUSimInstance(unsigned long long nWidth, unsigned long long nHeight,double distance, double3 this->renderFieldParamsPtr->rayDirection, double3 firstRayPosition, double flux, double lambda)
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
//		rayData.direction=this->renderFieldParamsPtr->rayDirection;
//		rayData.position.z=firstRayPosition.z;
//		rayData.running=true;
//		rayData.currentGeometryID=0;
//		rayData.lambda=lambda;
//		rayData.nImmersed=this->renderFieldParamsPtr->nImmersed
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
void GeometricRenderField::setParamsPtr(renderFieldParams *paramsPtr)
{
	this->renderFieldParamsPtr=paramsPtr;
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
renderFieldParams* GeometricRenderField::getParamsPtr(void)
{
	return this->renderFieldParamsPtr;
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
fieldError GeometricRenderField::createCPUSimInstance()
{
	if (this->Iptr != NULL)
	{
		delete this->Iptr;
	}
    Iptr=(double*) calloc(this->renderFieldParamsPtr->width*this->renderFieldParamsPtr->height,sizeof(double));
	if (!Iptr)
	{
		std::cout << "error in GeometricRenderField.createLayoutInstance(): memory for rayList could not be allocated" << "...\n";
		return FIELD_ERR;
	}

	unsigned int l_launch_width, l_launch_height, l_offsetX, l_offsetY;
	if ( this->renderFieldParamsPtr->width < GPU_SUBSET_WIDTH_MAX ) 
	{
		l_launch_width=this->renderFieldParamsPtr->width;
	}
	else
	{
		l_launch_width=GPU_SUBSET_WIDTH_MAX;
	}
	if ( this->renderFieldParamsPtr->height < GPU_SUBSET_HEIGHT_MAX ) 
	{
		l_launch_height=this->renderFieldParamsPtr->height;
	}
	else
	{
		l_launch_height=GPU_SUBSET_HEIGHT_MAX;
	}


	this->renderFieldParamsPtr->launchOffsetX=0;
	this->renderFieldParamsPtr->launchOffsetY=0;

	this->renderFieldParamsPtr->GPUSubset_height=l_launch_height;
	this->renderFieldParamsPtr->GPUSubset_width=l_launch_width;

	return FIELD_NO_ERR;
};

long2 GeometricRenderField::calcSubsetDim(void)
{
	unsigned long long width=this->renderFieldParamsPtr->width;
	unsigned long long height=this->renderFieldParamsPtr->height;

	long2 l_GPUSubsetDim;

	// calc launch_width of current launch
	long long restWidth=width-this->renderFieldParamsPtr->launchOffsetX;
	long long restHeight=height-this->renderFieldParamsPtr->launchOffsetY;
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
	this->renderFieldParamsPtr->GPUSubset_width=l_GPUSubsetDim.x;
	this->renderFieldParamsPtr->GPUSubset_height=l_GPUSubsetDim.y;
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
fieldError GeometricRenderField::traceScene(Group &oGroup, bool RunOnCPU)
{
	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;
	// start timing
	start=clock();

//	long2 l_GPUSubsetDim=calcSubsetDim();
//	this->renderFieldParamsPtr->GPUSubset_width=l_GPUSubsetDim.x;
//	this->renderFieldParamsPtr->GPUSubset_height=l_GPUSubsetDim.y;

	if (RunOnCPU)
	{
		std::cout << "rendering on " << numCPU << " cores of CPU." << "...\n";

		omp_set_num_threads(numCPU);

		//int threadCounter[20];
		//for (unsigned int i=0;i<20;i++)
		//	threadCounter[i]=0;


#pragma omp parallel default(shared) //shared(threadCounter)
{
		#pragma omp for schedule(dynamic, 50)
		//for (signed long long jy=0; jy<this->renderFieldParamsPtr->GPUSubset_height; jy++)
		//{
		for (signed long long j=0; j<this->renderFieldParamsPtr->GPUSubset_height*this->renderFieldParamsPtr->GPUSubset_width; j++)
		{
//			threadCounter[omp_get_thread_num()]=threadCounter[omp_get_thread_num()]+1;
			unsigned long long jx = j % this->renderFieldParamsPtr->GPUSubset_width;
			unsigned long long jy = (j-jx)/this->renderFieldParamsPtr->GPUSubset_width;
            jx=jx+this->renderFieldParamsPtr->launchOffsetX;
            jy=jy+this->renderFieldParamsPtr->launchOffsetY;
            for (unsigned long long jRay=0; jRay<this->renderFieldParamsPtr->nrRayDirections.x*this->renderFieldParamsPtr->nrRayDirections.y; jRay++)
            {
			    geomRenderRayStruct ray=this->createRay(jx,jy,jRay);
			    for(;;) // iterative tracing
			    {
				    if(!ray.running) 
					    break;
				    oGroup.trace(ray);
			    }
                this->Iptr[jx+jy*this->renderFieldParamsPtr->width]+=ray.cumFlux;
            }
		}
}
		//for (int i=0;i<20;i++)
		//{
		//	std::cout << "Thread number " << i << " has run " << threadCounter[i] << " times" << "...\n";
		//}
	}
	else
	{
        void				*data; // pointer to cast output buffer into

		std::cout << "rendering on GPU." << "...\n";

		initGPUSubset(context, seed_buffer_obj);
		// start current launch
		if (!RT_CHECK_ERROR_NOEXIT( rtContextLaunch2D( (context), 0, this->renderFieldParamsPtr->GPUSubset_width, this->renderFieldParamsPtr->GPUSubset_height), context))//this->renderFieldParamsPtr->launchOffsetX, this->renderFieldParamsPtr->launchOffsetY ) );
			return FIELD_ERR;

        // get result from GPU
		if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(output_buffer_obj, &data) , context))
			return FIELD_ERR;
    
        this->copyImagePart((double*)data);
		
		if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( output_buffer_obj ) , context))
			return FIELD_ERR;


	}
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<" ms to render " << this->renderFieldParamsPtr->GPUSubset_height*this->renderFieldParamsPtr->GPUSubset_width << " pixels." << "...\n";

	return FIELD_NO_ERR;
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
fieldError GeometricRenderField::copyImagePart(double *data)
{

	// copy the image line per line
    for (unsigned long long jy=0;jy<this->renderFieldParamsPtr->GPUSubset_height;jy++)
	{
		unsigned long long testIndex=jy*GPU_SUBSET_WIDTH_MAX;
		// memory range of completed lines + offsetX + number of line in current block*width of complete rayblock // we always allocate the max buffer on GPU, therefore we always need to adress the start of the line in this maximum buffer...
        unsigned long long offsetX=this->getParamsPtr()->launchOffsetX;
        unsigned long long offsetY=this->getParamsPtr()->launchOffsetY;
        memcpy(&(this->Iptr[offsetX+jy*this->getParamsPtr()->width]), &data[jy*GPU_SUBSET_WIDTH_MAX], this->renderFieldParamsPtr->GPUSubset_width*sizeof(double));
	}
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
fieldError GeometricRenderField::doSim(Group &oGroup, simAssParams &params, bool &simDone)
{
	unsigned long long width=this->getParamsPtr()->totalLaunch_width;
	unsigned long long height=this->getParamsPtr()->totalLaunch_height;

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
		std::cout << "error in GeometricRenderField.doSim(): GeometricRenderField.traceScene() returned an error" << "...\n";
		return FIELD_ERR;
	}
	this->subsetCounter++;
	// signal simulation progress via callback to gui
	if ((Field::p2ProgCallbackObject != NULL) && (Field::callbackProgress != NULL))
		Field::callbackProgress(Field::p2ProgCallbackObject, floorf(this->subsetCounter*100/roughNrOfSubsets));

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

	void *data; // pointer to cast output buffer into


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
//fieldError GeometricRenderField::writeData2File(FILE *hFile, rayDataOutParams outParams)
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
fieldError GeometricRenderField::write2TextFile(char* filename, detParams &oDetParams)
{
//	char t_filename[512];
//	sprintf(t_filename, "%s%sGeometricRenderField_%i.txt", filename, PATH_SEPARATOR, oDetParams.subSetNr);

	FILE* hFileOut;
	char t_filename[512];
	sprintf(t_filename, "%s%s%i%s", OUTPUT_FILEPATH, PATH_SEPARATOR, oDetParams.subSetNr, oDetParams.filenamePtr);
	hFileOut = fopen( t_filename, "w" ) ;
	if (!hFileOut)
	{
		std::cout << "error in GeometricRenderField.write2TextFile(): could not open output file: " << filename << "...\n";
		return FIELD_ERR;
	}
	if (1) //(oDetParams.reduceData==1)
	{
		for (unsigned long long jy=0; jy<this->renderFieldParamsPtr->GPUSubset_height; jy++)
		{
			for (unsigned long long jx=0; jx<this->renderFieldParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
				//if ((rayList[rayListIndex].currentGeometryID==oDetParams.geomID) || (oDetParams.geomID=-1))
				//{
				//	// write the data in row major format, where width is the size of one row and height is the size of one coloumn
				//	// if the end of a row is reached append a line feed 
				//	fprintf(hFileOut, "%.20lf ;%.20lf ;%.20lf; %.20lf ;%.20lf ;%.20lf ;%.20lf ;%.20lf; %i \n", rayList[rayListIndex].position.x, rayList[rayListIndex].position.y, rayList[rayListIndex].position.z, rayList[rayListIndex].direction.x, rayList[rayListIndex].direction.y, rayList[rayListIndex].direction.z, rayList[rayListIndex].flux, rayList[rayListIndex].opl, rayList[rayListIndex].currentGeometryID);
				//}
			}
		}
	}
	else
	{
		for (unsigned long long jy=0; jy<this->renderFieldParamsPtr->GPUSubset_height; jy++)
		{
			for (unsigned long long jx=0; jx<this->renderFieldParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;

				//if ((rayList[rayListIndex].currentGeometryID==oDetParams.geomID) || (oDetParams.geomID=-1))
				//{
				//	 write the data in row major format, where width is the size of one row and height is the size of one coloumn
				//	 if the end of a row is reached append a line feed 
				//	fprintf(hFileOut, "%.20lf ;%.20lf ;%.20lf; %.20lf ;%.20lf ;%.20lf; %.20lf; %.20lf; %i; %i; %.20lf; \n", rayList[rayListIndex].position.x, rayList[rayListIndex].position.y, rayList[rayListIndex].position.z, rayList[rayListIndex].direction.x, rayList[rayListIndex].direction.y, rayList[rayListIndex].direction.z, rayList[rayListIndex].flux, rayList[rayListIndex].opl, rayList[rayListIndex].currentGeometryID, rayList[rayListIndex].depth, rayList[rayListIndex].nImmersed);
				//}
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
//fieldError GeometricRenderField::write2MatFile(char* filename, detParams &oDetParams)
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
//	sprintf(t_filename, "%s%sGeometricRenderField", filename, PATH_SEPARATOR);
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
fieldError  GeometricRenderField::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams)
{
	Parser_XML l_parser;

    this->setSimMode(simParams.simMode);

	// call base class function
	if (FIELD_NO_ERR != Field::parseXml(field, fieldVec, simParams))
	{
		std::cout << "error in GeometricRenderField.parseXml(): Field.parseXml()  returned an error." << "...\n";
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
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "coherence", this->getParamsPtr()->coherence)))
		return FIELD_ERR;

    if (!this->checkParserError(l_parser.attrByNameToDouble(field, "pupTilt.x", this->getParamsPtr()->importanceAreaTilt.x)))
		return FIELD_ERR;
	this->getParamsPtr()->tilt.x=this->getParamsPtr()->tilt.x/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "pupTilt.y", this->getParamsPtr()->importanceAreaTilt.y)))
		return FIELD_ERR;
	this->getParamsPtr()->tilt.y=this->getParamsPtr()->tilt.y/360*2*PI;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "pupTilt.z", this->getParamsPtr()->importanceAreaTilt.z)))
		return FIELD_ERR;
	this->getParamsPtr()->tilt.z=this->getParamsPtr()->tilt.z/360*2*PI;
   	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "pupRoot.x", this->getParamsPtr()->importanceAreaRoot.x)))
		return FIELD_ERR;
   	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "pupRoot.y", this->getParamsPtr()->importanceAreaRoot.y)))
		return FIELD_ERR;
   	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "pupRoot.z", this->getParamsPtr()->importanceAreaRoot.z)))
		return FIELD_ERR;
   	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "pupAptRad.x", this->getParamsPtr()->importanceAreaHalfWidth.x)))
		return FIELD_ERR;
   	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "pupAptRad.y", this->getParamsPtr()->importanceAreaHalfWidth.y)))
		return FIELD_ERR;
    this->getParamsPtr()->importanceAreaApertureType=AT_ELLIPT;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "coherence", this->getParamsPtr()->coherence)))
		return FIELD_ERR;

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

    this->getParamsPtr()->totalLaunch_height=this->getParamsPtr()->height;
    this->getParamsPtr()->totalLaunch_width=this->getParamsPtr()->width;

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

    if (!this->checkParserError(l_parser.attrByNameToLong(field, "raysPerPixel.x", this->getParamsPtr()->nrRayDirections.x)))
		return FIELD_ERR;
    if (!this->checkParserError(l_parser.attrByNameToLong(field, "raysPerPixel.y", this->getParamsPtr()->nrRayDirections.y)))
		return FIELD_ERR;

    this->getParamsPtr()->nrPseudoLambdas=1;

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
 * \detail convert2Intensity 
 *
 * \param[in] IntensityField* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRenderField::convert2Intensity(Field* imagePtr, detParams &oDetParams)
{
	IntensityField* l_IntensityImagePtr=dynamic_cast<IntensityField*>(imagePtr);
	if (l_IntensityImagePtr == NULL)
	{
		std::cout << "error in GeometricRenderField.convert2Intensity(): imagePtr is not of type IntensityField" << "...\n";
		return FIELD_ERR;
	}

	if (l_IntensityImagePtr == NULL)
	{
		std::cout << "error in GeometricRayField.convert2Intensity(): imagePtr is not of type IntensityField" << "...\n";
		return FIELD_ERR;
	}
    if (this->getParamsPtr()->width*this->getParamsPtr()->height != imagePtr->getParamsPtr()->nrPixels.x*imagePtr->getParamsPtr()->nrPixels.y*imagePtr->getParamsPtr()->nrPixels.z)
    {
	    std::cout << "error in GeometricRenderField.convert2Intensity(): diemsnions of render field do not fit dimensions of detetcor field" << "...\n";
	    return FIELD_ERR;
    }
    // copy intensity field
    memcpy(l_IntensityImagePtr->getIntensityPtr(), this->Iptr, this->getParamsPtr()->width*this->getParamsPtr()->height*sizeof(double));

    return FIELD_NO_ERR;
};