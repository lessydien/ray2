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

/**\file GeometricRayField.cpp
* \brief Rayfield for geometric raytracing
* 
*           
* \author Mauch
*/
#include <omp.h>
#include "GeometricRayField.h"
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
fieldError GeometricRayField::setLambda(double lambda)
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
unsigned long long GeometricRayField::getRayListLength(void)
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
fieldError GeometricRayField::setRay(rayStruct ray, unsigned long long index)
{
	if (index <= this->rayListLength)
	{
		rayList[index]=ray;
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
rayStruct* GeometricRayField::getRay(unsigned long long index)
{
	if (index <= this->rayListLength)
	{
		return &rayList[index];	
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
rayStruct* GeometricRayField::getRayList(void)
{
	return &rayList[0];	
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
void GeometricRayField::setRayList(rayStruct* rayStructPtr)
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
fieldError GeometricRayField::copyRayList(rayStruct *data, long long length)
{
	if (length > this->rayListLength)
	{
		std::cout << "error in GeometricRayField.copyRayList(): subset dimensions exceed rayLIst dimension" << std::endl;
		return FIELD_ERR;
	}

	// copy the ray list line per line
	for (unsigned long long jy=0;jy<this->rayParamsPtr->GPUSubset_height;jy++)
	{
		unsigned long long testIndex=jy*GPU_SUBSET_WIDTH_MAX;
		//                     memory range of completed lines + offsetX + number of line in current block*width of complete rayblock // we always allocate the max buffer on GPU, therefore we always need to adress the start of the line in this maximum buffer...
		memcpy(&(this->rayList[jy*GPU_SUBSET_WIDTH_MAX]), &data[jy*GPU_SUBSET_WIDTH_MAX], this->rayParamsPtr->GPUSubset_width*sizeof(rayStruct));
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
fieldError GeometricRayField::copyRayListSubset(rayStruct *data, long2 launchOffset, long2 subsetDim)
{
//	long2 testOffset=launchOffset;
//	long2 testDim=subsetDim;
	//  ----memory range of completed lines---- + ---memory range blocks in given line---
	if (launchOffset.y*this->rayParamsPtr->width+(subsetDim.x+launchOffset.x)*subsetDim.y > this->rayListLength)
	{
		std::cout << "error in GeometricRayField.copyRayListSubset(): subset dimensions exceed rayLIst dimension" << std::endl;
		return FIELD_ERR;
	}
	// copy the ray list line per line
	for (long long jy=0;jy<subsetDim.y;jy++)
	{
		unsigned long long testIndex=launchOffset.y*this->rayParamsPtr->width+launchOffset.x+jy*this->rayParamsPtr->width;
		//                     memory range of completed lines + offsetX + number of line in current block*width of complete rayblock // we always allocate the max buffer on GPU, therefore we always need to adress the start of the line in this maximum buffer...
		memcpy(&(this->rayList[launchOffset.y*this->rayParamsPtr->width+launchOffset.x+jy*this->rayParamsPtr->width]), &data[jy*GPU_SUBSET_WIDTH_MAX], subsetDim.x*sizeof(rayStruct));
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
void GeometricRayField::setPathToPtx(char* path)
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
const char* GeometricRayField::getPathToPtx(void)
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
fieldError GeometricRayField::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
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
	//	std::cout <<"error in GeometricRayField.createOptixInstance(): RayField.createOptixInstance() returned an error." << std::endl;
	//	return FIELD_ERR;
	//}

	/* Ray generation program */
	char rayGenName[128];
	sprintf(rayGenName, "rayGeneration");
	switch (this->getParamsPtr()->dirDistrType)
	{
	case RAYDIR_RAND_RECT:
		strcat(rayGenName, "_DirRandRect");
		break;
	case RAYDIR_RANDNORM_RECT:
		strcat(rayGenName, "_DirRandNormRect");
		break;
	case RAYDIR_RAND_RAD:
		strcat(rayGenName, "_DirRandRad");
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
		std::cout <<"error in PathTracingRayField.createOptixInstance(): unknown distribution of ray directions." << std::endl;
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
		std::cout <<"error in RayField.createOptixInstance(): unknown distribution of ray positions." << std::endl;
		return FIELD_ERR;
		break;
	}
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
		std::cout <<"error in PathTracingRayField.createOptixInstance(): create CPUSimInstance() returned an error." << std::endl;
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
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(rayFieldParams), this->rayParamsPtr), context) )
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
 fieldError GeometricRayField::initGPUSubset(RTcontext &context, RTbuffer &seed_buffer_obj)
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
fieldError GeometricRayField::initCPUSubset()
{
	clock_t start, end;
	double msecs=0;

	// check wether we will be able to fit all the rays into our raylist. If not some eror happened earlier and we can not proceed...
	if ((this->rayParamsPtr->GPUSubset_width*this->rayParamsPtr->GPUSubset_height)<=this->rayListLength)
	{

		// see if there are any rays to create	
		if (this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width >= 1)
		{
			// width of ray field in physical dimension
			double physWidth=this->rayParamsPtr->rayPosEnd.x-this->rayParamsPtr->rayPosStart.x;
			// height of ray field in physical dimension
			double physHeight=this->rayParamsPtr->rayPosEnd.y-this->rayParamsPtr->rayPosStart.y;
			// increment of rayposition in x and y in case of GridRect definition 
			double deltaW=0;
			double deltaH=0;
			// increment radial ( along x- and y ) and angular direction in GridRad definition
			double deltaRx=0;
			double deltaRy=0;
			double deltaPhi=0;
			// radius in dependence of phi when calculating GRID_RAD
			double R=0;
			// calc centre of ray field 
			double2 rayFieldCentre=make_double2(this->rayParamsPtr->rayPosStart.x+physWidth/2,this->rayParamsPtr->rayPosStart.y+physHeight/2);
			// calc centre angle of opening cone
			double3 rayAngleCentre=make_double3((this->rayParamsPtr->alphaMax.x+this->rayParamsPtr->alphaMin.x)/2,(this->rayParamsPtr->alphaMax.y+this->rayParamsPtr->alphaMin.y)/2,0);
			double angleWidth=(this->rayParamsPtr->alphaMax.x-this->rayParamsPtr->alphaMin.x);
			double angleHeight=(this->rayParamsPtr->alphaMax.y-this->rayParamsPtr->alphaMin.y);
			// declar variables for randomly distributing ray directions via an importance area
			double2 impAreaHalfWidth;
			double3 dirImpAreaCentre, tmpPos, impAreaRoot;
			double impAreaX, impAreaY, r, theta;
			double3 impAreaAxisX, impAreaAxisY;

			double3 alpha=make_double3(0,0,0); // rotation angle of raydirection around x and y

			// start timing
			start=clock();

			std::cout << "initalizing random seed" << std::endl;

			int seed = (int)time(0);            // random seed
			RandomInit(seed, x);

			// create random seeds for all the rays
			for(signed long long j=0;j<this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width;j++)
			{
				this->rayList[j].currentSeed=(uint)BRandom(x);
			}

			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize random seeds of " << this->rayParamsPtr->GPUSubset_width*this->rayParamsPtr->GPUSubset_height << " rays." << std::endl;

			// start timing
			start=clock();

			// create all the rays
		std::cout << "initializing rays on " << numCPU << " cores of CPU." << std::endl;

		omp_set_num_threads(numCPU);

#pragma omp parallel default(shared)
{
			#pragma omp for schedule(dynamic, 50)//schedule(static)//

			for (signed long long j=0; j<this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width; j++)
			{
				unsigned long long jx = j % this->rayParamsPtr->GPUSubset_width;
				unsigned long long iy = (j-jx)/this->rayParamsPtr->GPUSubset_width;
				unsigned long long rayListIndex=jx+iy*GPU_SUBSET_WIDTH_MAX;

				long long index=0; // loop counter for random rejection method
				double r; // variables for creating random number inside an ellipse

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

			// create all the rays
			//for(unsigned long long jx=0;jx<this->rayParamsPtr->GPUSubset_width;jx++)
			//{
			//	for(unsigned long long iy=0;iy<this->rayParamsPtr->GPUSubset_height;iy++)
			//	{
					// consider offsets of current subset
					unsigned long long jGes=jx+this->rayParamsPtr->launchOffsetX;
					unsigned long long iGes=(iy+this->rayParamsPtr->launchOffsetY);

					// consider number of directions per point source. As the multiple directions per point are listed in y direction, we need to create a modified y-index here...
					unsigned long long jPos=floorf(jGes/(this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y));
					unsigned long long iPos=iGes;

					// calc the indices of the directions
					unsigned long long indexDir=iGes % (this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y);

					unsigned long long jDir=floorf(indexDir/this->rayParamsPtr->nrRayDirections.x);
					unsigned long long iDir=indexDir % this->rayParamsPtr->nrRayDirections.x;

					// declare variables for placing a ray randomly inside an ellipse
					double theta;
					double ellipseX;
					double ellipseY;
					double3 exApt;
					double3 eyApt;

					// create rayposition in local coordinate system according to distribution type
					rayData.position.z=0; // all rays start at z=0 in local coordinate system
					switch (this->rayParamsPtr->posDistrType)
					{
						case RAYPOS_GRID_RECT:
							// calc increment along x- and y-direction
							if (this->rayParamsPtr->width>0)
								deltaW= (physWidth)/double(this->rayParamsPtr->width);
							if (this->rayParamsPtr->height>0)
								// multiple directions per point are listed in y-direction. Therefore the physical height of the rayfield is different from the height of the ray list. This has to be considered here...
								deltaH= (physHeight)/double(this->rayParamsPtr->height);
							rayData.position.x=this->rayParamsPtr->rayPosStart.x+deltaW/2+jPos*deltaW;
							rayData.position.y=this->rayParamsPtr->rayPosStart.y+deltaH/2+iPos*deltaH;
							break;
						case RAYPOS_RAND_RECT:							
							rayData.position.x=(Random(x_l)-0.5)*physWidth+rayFieldCentre.x;
							rayData.position.y=(Random(x_l)-0.5)*physHeight+rayFieldCentre.y;
							break;
						case RAYPOS_RAND_RECT_NORM:							
							rayData.position.x=(RandomGauss(x_l))*physWidth/2+rayFieldCentre.x;
							rayData.position.y=(RandomGauss(x_l))*physHeight/2+rayFieldCentre.y;
							break;
						case RAYPOS_GRID_RAD:
							// calc increment along radial and angular direction
							if (this->rayParamsPtr->width>0)
							{
								deltaRx= (physWidth/2)/double(this->rayParamsPtr->width);
								deltaRy= (physHeight/2)/double(this->rayParamsPtr->width);
							}
							if (this->rayParamsPtr->height>0)
								deltaPhi= (2*PI)/(this->rayParamsPtr->height);
							R=(deltaRx/2+deltaRx*jPos)*(deltaRy/2+deltaRy*jPos)/sqrt(pow((deltaRy/2+deltaRy*jPos)*cos(deltaPhi/2+deltaPhi*iPos),2)+pow((deltaRx/2+deltaRx*jPos)*sin(deltaPhi/2+deltaPhi*iPos),2));							
							if (deltaRy==0)
								R=0;
							// now calc rectangular coordinates from polar coordinates
							rayData.position.x=cos(deltaPhi/2+deltaPhi*iPos)*R;
							rayData.position.y=sin(deltaPhi/2+deltaPhi*iPos)*R;

							break;
						case RAYPOS_RAND_RAD:
							// place a point uniformingly randomly inside the importance area
							theta=2*PI*Random(x_l);
							r=sqrt(Random(x_l));
							ellipseX=physWidth/2*r*cos(theta);
							ellipseY=physHeight/2*r*sin(theta);
							exApt=make_double3(1,0,0);
							eyApt=make_double3(0,1,0);
							rayData.position=make_double3(0,0,0)+ellipseX*exApt+ellipseY*eyApt;

							//// create random position until it is inside the ellipse
							//do
							//{
							//	rayData.position.x=(Random(x_l)-0.5)*physWidth;
							//	rayData.position.y=(Random(x_l)-0.5)*physHeight;
							//	r=rayData.position.x*rayData.position.x/(physWidth*physWidth/4)+rayData.position.y*rayData.position.y/(physHeight*physHeight/4);
							//	index++;
							//	if (index>1000000)
							//		break;
							//} while ( (r >= 1.0) );
							break;
						case RAYPOS_RAND_RAD_NORM:
							// place a point uniformingly randomly inside the importance area
							theta=2*PI*RandomGauss(x_l);
							r=sqrt(RandomGauss(x_l));
							ellipseX=physWidth/2*r*cos(theta);
							ellipseY=physHeight/2*r*sin(theta);
							exApt=make_double3(1,0,0);
							eyApt=make_double3(0,1,0);
							rayData.position=make_double3(0,0,0)+ellipseX*exApt+ellipseY*eyApt;
							break;
						default:
							rayData.position=make_double3(0,0,0);
							std::cout << "error in GeometricRayField.initCPUSubset: unknown distribution of rayposition" << std::endl;
							// report error
							break;
					}
					if(this->rayParamsPtr->width*this->rayParamsPtr->height==1)
					{
						rayData.position=this->rayParamsPtr->rayPosStart;
					}
					// transform rayposition into global coordinate system
					rayData.position=this->rayParamsPtr->Mrot*rayData.position+this->rayParamsPtr->translation;


					// declar variables for randomly distributing ray directions via an importance area
					double2 impAreaHalfWidth,phi;
					double3 dirImpAreaCentre, tmpPos, impAreaRoot;
					double impAreaX, impAreaY;
					
					// create raydirection in local coordinate system according to distribution type
					switch (this->rayParamsPtr->dirDistrType)
					{
						case RAYDIR_UNIFORM:
							rayData.direction=this->rayParamsPtr->rayDirection;
							rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
							break;
						case RAYDIR_RAND_RECT:
							// the strategy is to define an importance area that corresponds to the given emission cone. The ray directions are then distributed to aim in thi importance area
							impAreaRoot = rayData.position+rotateRay(this->rayParamsPtr->rayDirection, rayAngleCentre);
							impAreaHalfWidth = make_double2(tan(angleWidth/2), tan(angleHeight/2));
							aimRayTowardsImpArea(rayData.direction, rayData.position, impAreaRoot, impAreaHalfWidth, make_double3(0,0,0), AT_RECT, rayData.currentSeed);
							// transform raydirection into global coordinate system
							rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;

							//rayAngleCentre=make_double3((this->rayParamsPtr->alphaMax.x+this->rayParamsPtr->alphaMin.x)/2,(this->rayParamsPtr->alphaMax.y+this->rayParamsPtr->alphaMin.y)/2,0);
							//rayAngleHalfWidth=make_double2((this->rayParamsPtr->alphaMax.x-this->rayParamsPtr->alphaMin.x)/2,(this->rayParamsPtr->alphaMax.y-this->rayParamsPtr->alphaMin.y)/2);
							//// create random angles inside the given range
							//phi=make_double2(2*(Random(x_l)-0.5)*rayAngleHalfWidth.x+rayAngleCentre.x,2*(Random(x_l)-0.5)*rayAngleHalfWidth.y+rayAngleCentre.y);
							//// create unit vector with the given angles
							//rayData.direction=createObliqueVec(phi);//normalize(make_double3(tan(phi.y),tan(phi.x),1));
							//// transform raydirection into global coordinate system
							//rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;


							//// create points inside importance area to randomly distribute ray direction
							//impAreaHalfWidth.x=(tan(this->rayParamsPtr->alphaMax.x)-tan(this->rayParamsPtr->alphaMin.x))/2;
							//impAreaHalfWidth.y=(tan(this->rayParamsPtr->alphaMax.y)-tan(this->rayParamsPtr->alphaMin.y))/2;
							//dirImpAreaCentre=make_double3(0,0,1);
							//rotateRay(&dirImpAreaCentre, rayAngleCentre);
							//// the centre of the importance area is the root of the current geometry + the direction to the imp area centre normalized such that the importance area is 1mm away from the current geometry
							//impAreaRoot=make_double3(0,0,0)+dirImpAreaCentre/dot(make_double3(0,0,1), dirImpAreaCentre);
							//// now distribute points inside importance area
							//theta=2*PI*Random(x_l);
							//r=sqrt(Random(x_l));
							//impAreaX=impAreaHalfWidth.x*r*cos(theta);
							//impAreaY=impAreaHalfWidth.y*r*sin(theta);
							//tmpPos=impAreaRoot+impAreaX*make_double3(1,0,0)+impAreaY*make_double3(0,1,0);
							//rayData.direction=normalize(tmpPos-make_double3(0,0,0));
							//// transform raydirection into global coordinate system
							//rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
							//rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
							break;
						case RAYDIR_RAND_RAD:
							// the strategy is to define an importance area that corresponds to the given emission cone. The ray directions are then distributed to aim in thi importance area
							dirImpAreaCentre=this->rayParamsPtr->rayDirection;
							rotateRay(&dirImpAreaCentre,rayAngleCentre);
							impAreaRoot = rayData.position+dirImpAreaCentre;
							impAreaHalfWidth = make_double2(tan(angleHeight/2), tan(angleWidth/2));
							aimRayTowardsImpArea(rayData.direction, rayData.position, impAreaRoot, impAreaHalfWidth, make_double3(0,0,0), AT_ELLIPT, rayData.currentSeed);
							// transform raydirection into global coordinate system
							rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
							break;
						case RAYDIR_RANDNORM_RECT:
							// create random angles inside the given range
							phi=make_double2(RandomGauss(x_l)*angleWidth+rayAngleCentre.x,RandomGauss(x_l)*angleHeight+rayAngleCentre.y);
							// create unit vector with the given angles
							rayData.direction=createObliqueVec(phi);//normalize(make_double3(tan(phi.y),tan(phi.x),1));
							// transform raydirection into global coordinate system
							rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
						break;

					case RAYDIR_RANDIMPAREA:

						aimRayTowardsImpArea(rayData.direction, rayData.position, this->rayParamsPtr->importanceAreaRoot, this->rayParamsPtr->importanceAreaHalfWidth, this->rayParamsPtr->importanceAreaTilt, this->rayParamsPtr->importanceAreaApertureType, rayData.currentSeed);

						//if (this->rayParamsPtr->importanceAreaApertureType==AT_RECT)
						//{
						//	// place temporal point uniformingly randomly inside the importance area
						//	impAreaX=(Random(x_l)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.x;
						//	impAreaY=(Random(x_l)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.y; 
						//}
						//else
						//{
						//	if (this->rayParamsPtr->importanceAreaApertureType==AT_ELLIPT)
						//	{
						//		theta=2*PI*Random(x_l);
						//		r=sqrt(Random(x_l));
						//		impAreaX=this->rayParamsPtr->importanceAreaHalfWidth.x*r*cos(theta);
						//		impAreaY=this->rayParamsPtr->importanceAreaHalfWidth.y*r*sin(theta);
						//	}
						//	else
						//	{
						//		std::cout << "error in RayField.createCPUInstance: importance area for defining ray directions of source is only allowed with objects that have rectangular or elliptical apertures" << std::endl;
						//		return FIELD_ERR;
						//	}
						//}

						//impAreaAxisX=make_double3(1,0,0);
						//impAreaAxisY=make_double3(0,1,0);
						//rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
						//rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

						//tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
						//rayData.direction=normalize(tmpPos-rayData.position);
						break;

						case RAYDIR_GRID_RECT:
							// calc increment along x- and y-direction
							if (this->rayParamsPtr->nrRayDirections.x>0)
								deltaW= (2*this->rayParamsPtr->importanceAreaHalfWidth.x)/double(this->rayParamsPtr->nrRayDirections.x);
							if (this->rayParamsPtr->nrRayDirections.y>0)
								deltaH= (2*this->rayParamsPtr->importanceAreaHalfWidth.y)/double(this->rayParamsPtr->nrRayDirections.y);
							impAreaX=-this->rayParamsPtr->importanceAreaHalfWidth.x+deltaW/2+jDir*deltaW; 
							impAreaY=-this->rayParamsPtr->importanceAreaHalfWidth.y+deltaH/2+iDir*deltaH; 
							impAreaAxisX=make_double3(1,0,0);
							impAreaAxisY=make_double3(0,1,0);
							rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
							rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

							tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
							rayData.direction=normalize(tmpPos-rayData.position);
							break;
						case RAYDIR_GRID_RAD:
							// calc increment along radial and angular direction
							if (this->rayParamsPtr->nrRayDirections.x>0)
							{
								deltaRx= (this->rayParamsPtr->importanceAreaHalfWidth.x)/double(this->rayParamsPtr->nrRayDirections.x);
								deltaRy= (this->rayParamsPtr->importanceAreaHalfWidth.y)/double(this->rayParamsPtr->nrRayDirections.x);
							}
							if (this->rayParamsPtr->nrRayDirections.y>0)
								deltaPhi= (2*PI)/this->rayParamsPtr->nrRayDirections.y;
							if (iDir==0)
							{
								R=deltaRx/2*deltaRy/2/sqrt(pow(deltaRy/2,2));
//								impAreaX=0;
//								impAreaY=0;
							}
							else
							{
								// calc r(phi) for given phi and radii of ellipse. see http://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center for reference
								R=deltaRx/2+deltaRx*iDir*deltaRy*iDir/sqrt(pow(deltaRy*iDir*cos(deltaPhi*jDir),2)+pow(deltaRx*iDir*sin(deltaPhi*jDir),2));
							}
							if ( deltaRy==0 )
								R=0;
							// now calc rectangular coordinates from polar coordinates
							impAreaX=cos(deltaPhi*jDir)*R;
							impAreaY=sin(deltaPhi*jDir)*R;

							impAreaAxisX=make_double3(1,0,0);
							impAreaAxisY=make_double3(0,1,0);
							rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
							rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

							tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
							rayData.direction=normalize(tmpPos-rayData.position);

							break;
						default:
							rayData.direction=make_double3(0,0,0);
							std::cout << "error in GeometricRayField.initCPUSubset: unknown raydirection distribution" << std::endl;
							// report error
							break;
					}
					// transform raydirection into global coordinate system
//					rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;

					rayData.currentSeed=x_l[4];//(uint)BRandom(x);
					this->setRay(rayData,(unsigned long long)(jx+iy*GPU_SUBSET_WIDTH_MAX));
				}// end for
} // end omp
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize " << this->rayParamsPtr->GPUSubset_width*this->rayParamsPtr->GPUSubset_height << " rays." << std::endl;

		}
		else if(this->rayParamsPtr->width*this->rayParamsPtr->height<1)
		{
			//not Possible. Report error or set n=-n
			std::cout << "error in GeometricRayField.initCPUInstance: negative raynumber" << std::endl;
			return FIELD_ERR;
		}
		this->update=false;
	}	// end if GPUsubsetwidth*height<rayListLength
	else
	{
		std::cout << "error in GeometricRayField.initCPUInstance: rayList is smaller than simulation subset" << std::endl;
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
fieldError GeometricRayField::createOptixInstance(RTcontext* context, unsigned long long width, unsigned long long height, double3 rayPosStart, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference)
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
	origin_minVar.x=this->rayParamsPtr->rayPosStart.x;
	origin_minVar.y=this->rayParamsPtr->rayPosStart.y;
	origin_minVar.z=this->rayParamsPtr->rayPosStart.z;

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


	RT_CHECK_ERROR2( rtVariableSetUserData(origin_max, sizeof(double3), &origin_maxVar) );
	RT_CHECK_ERROR2( rtVariableSetUserData(origin_min, sizeof(double3), &origin_minVar) );
	RT_CHECK_ERROR2( rtVariableSetUserData(l_lambda, sizeof(double), &this->rayParamsPtr->lambda) );
	RT_CHECK_ERROR2( rtVariableSet1ui(l_size_xGrad, size_xGrad) );
	RT_CHECK_ERROR2( rtVariableSet1ui(l_size_yGrad, size_yGrad) );
//	RT_CHECK_ERROR2( rtVariableSet1ui(number, numberVar ) );
	RT_CHECK_ERROR2( rtVariableSet1ui(launch_width, l_launch_width ) );
	RT_CHECK_ERROR2( rtVariableSet1ui(launch_height, l_launch_height ) );

    RT_CHECK_ERROR2( rtContextSetRayGenerationProgram( *context,0, this->ray_gen_program ) );

	return FIELD_NO_ERR;
};

//void GeometricRayField::createCPUSimInstance(unsigned long long nWidth, unsigned long long nHeight,double distance, double3 this->rayParamsPtr->rayDirection, double3 firstRayPosition, double flux, double lambda)
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
void GeometricRayField::setParamsPtr(rayFieldParams *paramsPtr)
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
rayFieldParams* GeometricRayField::getParamsPtr(void)
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
fieldError GeometricRayField::createCPUSimInstance()
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
		std::cout << "error in GeometricRayField.createLayoutInstance(): memory for rayList could not be allocated" << std::endl;
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
fieldError GeometricRayField::createLayoutInstance()
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
		std::cout << "error in GeometricRayField.createLayoutInstance(): memory for rayList could not be allocated. try to reduce ray tiling size" << std::endl;
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
fieldError GeometricRayField::traceScene(Group &oGroup)
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

long2 GeometricRayField::calcSubsetDim(void)
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
fieldError GeometricRayField::traceScene(Group &oGroup, bool RunOnCPU)
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
		std::cout << "tracing on " << numCPU << " cores of CPU." << std::endl;

		if (FIELD_NO_ERR!= initCPUSubset())
		{
			std::cout << "error in GeometrciRayField.traceScene(): initCPUSubset() returned an error." << std::endl;
			return FIELD_ERR;
		}

		std::cout << "starting the actual trace..." << std::endl;		

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
			//std::cout << "Iteration " << jy << " running Cur Thread " << omp_get_thread_num() << "Num Threads " << omp_get_num_threads() << "Max Threads " << omp_get_max_threads() << " running" << std::endl;
		}
}
		//for (int i=0;i<20;i++)
		//{
		//	std::cout << "Thread number " << i << " has run " << threadCounter[i] << " times" << std::endl;
		//}
	}
	else
	{
		//RTsize				buffer_width, buffer_height; // get size of output buffer
		void				*data; // pointer to cast output buffer into
 		//rayStruct			*bufferData;

		std::cout << "tracing on GPU." << std::endl;

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
		//std::cout << "DEBUG: jx=" << jx << " jy=" << jy << std::endl;
		//copyRayListSubset((rayStruct*)data, l_launchOffset, l_GPUSubsetDim);
		if (FIELD_NO_ERR != copyRayList((rayStruct*)data,this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width) )
		{
			std::cout << "error in GeometricRayField.traceScene(): copyRayList() returned an error" << std::endl;
			return FIELD_NO_ERR;
		}
		
		if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( output_buffer_obj ) , context))
			return FIELD_ERR;
	}
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<" ms to trace " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

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
fieldError GeometricRayField::traceStep(Group &oGroup, bool RunOnCPU)
{
	if (!RunOnCPU)
		std::cout << "warning in GeometricRayField.traceStep(): GPU acceleration is not implemented, continuing on CPU anyways..." << std::endl;

	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;
	// start timing
	start=clock();

//	long2 l_GPUSubsetDim=calcSubsetDim();
//	this->rayParamsPtr->GPUSubset_width=l_GPUSubsetDim.x;
//	this->rayParamsPtr->GPUSubset_height=l_GPUSubsetDim.y;
	std::cout << "tracing on " << numCPU << " cores of CPU." << std::endl;

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
					oGroup.trace(rayList[rayListIndex]);
			}
		}
//}

	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<" ms to trace " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

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
fieldError GeometricRayField::doSim(Group &oGroup, simAssParams &params, bool &simDone)
{
	unsigned long long width=this->getParamsPtr()->totalLaunch_width;//SourceListPtrPtr->getParamsPtr()->width*SourceListPtrPtr->getParamsPtr()->nrRayDirections.x*SourceListPtrPtr->getParamsPtr()->nrRayDirections.y;
	unsigned long long height=this->getParamsPtr()->totalLaunch_height;//SourceListPtrPtr->getParamsPtr()->height;

	unsigned long long roughNrOfSubsets=std::floorf(width*height/(this->getSubsetWidthMax()*this->getSubsetHeightMax()))+1;


	std::cout << "****************************************************** " << std::endl;
	std::cout << "starting subset.......... " << std::endl;
	std::cout << std::endl;
	/***********************************************
	/	trace rays
	/***********************************************/

	long2 l_GPUSubsetDim=this->calcSubsetDim();

	if (FIELD_NO_ERR != this->traceScene(oGroup, params.RunOnCPU) )//, context, output_buffer_obj, seed_buffer_obj) )
	{
		std::cout << "error in GeometricRayField.doSim(): GeometricRayField.traceScene() returned an error" << std::endl;
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
	std::cout << " " << tracedRayNr <<" out of " << width*height << " rays traced in total" << std::endl;

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
//fieldError GeometricRayField::writeData2File(FILE *hFile, rayDataOutParams outParams)
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
fieldError GeometricRayField::write2TextFile(char* filename, detParams &oDetParams)
{
//	char t_filename[512];
//	sprintf(t_filename, "%s%sGeometricRayField_%i.txt", filename, PATH_SEPARATOR, oDetParams.subSetNr);

	FILE* hFileOut;
	char t_filename[512];
	sprintf(t_filename, "%s%s%i%s", OUTPUT_FILEPATH, PATH_SEPARATOR, oDetParams.subSetNr, oDetParams.filenamePtr);
	hFileOut = fopen( t_filename, "w" ) ;
	if (!hFileOut)
	{
		std::cout << "error in GeometricRayField.write2TextFile(): could not open output file: " << filename << std::endl;
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
//fieldError GeometricRayField::write2MatFile(char* filename, detParams &oDetParams)
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
//	sprintf(t_filename, "%s%sGeometricRayField", filename, PATH_SEPARATOR);
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
 * \detail convert2Intensity 

 *
 * \param[in] Field* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRayField::convert2Intensity(Field* imagePtr, detParams &oDetParams)
{
	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;

	// start timing
	start=clock();

	//long2 l_GPUSubsetDim=calcSubsetDim();
	// cast the image to an IntensityField
	IntensityField* l_IntensityImagePtr=dynamic_cast<IntensityField*>(imagePtr);
	if (l_IntensityImagePtr == NULL)
	{
		std::cout << "error in GeometricRayField.convert2Intensity(): imagePtr is not of type IntensityField" << std::endl;
		return FIELD_ERR;
	}
		

	/************************************************************************************************************************** 
	* the idea in calculating the flux per pixel is as following:
	* first we create unit vectors along global coordinate axis. 
	* Then we scale this vectors with the scaling of the respective pixel.
	* Then we rotate and translate these vectors into the local coordinate system of the IntensityField
	* Finally we solve the equation system that expresses the ray position in terms of these rotated and scaled vectors.
	* floor() of the coefficients of these vectors gives the indices we were looking for                                      
	****************************************************************************************************************************/
	double3 scale=l_IntensityImagePtr->getParamsPtr()->scale;
	long3 nrPixels=l_IntensityImagePtr->getParamsPtr()->nrPixels;
	scale.z=2*oDetParams.apertureHalfWidth.z/nrPixels.z; // we need to set this here as the IntensityField coming from the Detector is set for PseudoBandwidth...

	// create unit vectors
	double3 t_ez = make_double3(0,0,1);
	double3 t_ey=make_double3(0,1,0);
	double3 t_ex=make_double3(1,0,0);
	// transform unit vectors into local coordinate system of IntensityField
	rotateRay(&t_ez,oDetParams.tilt);
	rotateRay(&t_ey,oDetParams.tilt);
	rotateRay(&t_ex,oDetParams.tilt);

	// the origin of the IntensityField is at the outer edge of the detector rather than at the origin
	double3 offset;
	offset=oDetParams.root-oDetParams.apertureHalfWidth.x*t_ex;//+0.5*l_IntensityImagePtr->getParamsPtr()->scale*t_ex;
	offset=offset-oDetParams.apertureHalfWidth.y*t_ey;//+0.5*l_IntensityImagePtr->getParamsPtr()->scale*t_ey;
	offset=offset-oDetParams.apertureHalfWidth.z*t_ez;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ez;

	short solutionIndex;

	unsigned long long hitNr=0;

	double3 posMinOffset;
	double3 indexFloat;
	long3 index;
	if (this->rayParamsPtr->coherence==1) // sum coherently
	{
		complex<double> i_compl=complex<double>(0,1); // define complex number "i"

				//double phi1=2*PI/this->rayList[0].lambda*20;
				//complex<double> l_U1=complex<double>(this->rayList[0].flux*cos(phi1),this->rayList[0].flux*sin(phi1));
				//double phi2=2*PI/this->rayList[0].lambda*20.02;
				//complex<double> l_U2=complex<double>(this->rayList[0].flux*cos(phi2),this->rayList[0].flux*sin(phi2));
				//l_IntensityImagePtr->getComplexAmplPtr()[0]=l_U1+l_U2;

		for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		{
			for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
				// transform to local coordinate system
				double3 tmpPos=this->rayList[rayListIndex].position-offset;
				rotateRayInv(&tmpPos,oDetParams.tilt);

				index.x=floor((tmpPos.x)/scale.x);
				index.y=floor((tmpPos.y)/scale.y);
				index.z=floor((tmpPos.z)/scale.z);


				// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
				if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) )  && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
				{
					// use this ray only if it agrees with the ignoreDepth
					if ( this->rayList[rayListIndex].depth > oDetParams.ignoreDepth )
					{
						hitNr++;
						for (unsigned long long jWvl=0; jWvl<this->rayParamsPtr->nrPseudoLambdas; jWvl++)
						{
							double wvl=(this->rayList[rayListIndex].lambda-this->rayParamsPtr->pseudoBandwidth/2+this->rayParamsPtr->pseudoBandwidth/this->rayParamsPtr->nrPseudoLambdas*jWvl);
							double phi=2*PI/wvl*this->rayList[rayListIndex].opl;
							complex<double> l_U=complex<double>(this->rayList[rayListIndex].flux*cos(phi),this->rayList[rayListIndex].flux*sin(phi));
							l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+l_U; // create a complex amplitude from the rays flux and opl and sum them coherently
						}
					}
				}
			}
		}

		//for (unsigned long long j=0;j<rayListLength;j++)
		//{
		//	posMinOffset=this->rayList[j].position-offset;
		//	indexFloat=MatrixInv*posMinOffset;
		//	index.x=floor(indexFloat.x+0.5);
		//	index.y=floor(indexFloat.y+0.5);
		//	index.z=floor(indexFloat.z+0.5);
		//	
		//	// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
		//	if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
		//	{
		//		complex<double> l_exp=complex<double>(0,2*PI/this->rayList[j].lambda*this->rayList[j].opl);
		//		l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+this->rayList[j].flux*c_exp(l_exp); // create a complex amplitude from the rays flux and opl and sum them coherently
		//	}
		//}
		// loop through the pixels and calc intensity from complex amplitudes
		for (unsigned long long jx=0;jx<nrPixels.x;jx++)
		{
			for (unsigned long long jy=0;jy<nrPixels.y;jy++)
			{
				for (unsigned long long jz=0;jz<nrPixels.z;jz++)
				{
					// intensity is square of modulus of complex amplitude
					(l_IntensityImagePtr->getIntensityPtr())[jx+jy*nrPixels.x+jz*nrPixels.x*nrPixels.y]=pow(abs(l_IntensityImagePtr->getComplexAmplPtr()[jx+jy*nrPixels.x+jz*nrPixels.x*nrPixels.y]),2);
				}
			}
		}
	}
	else 
	{
		if (this->rayParamsPtr->coherence == 0)// sum incoherently
		{
			
			for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
			{
				for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
				{

					unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
					// transform to local coordinate system
					double3 tmpPos=this->rayList[rayListIndex].position-offset;
					rotateRayInv(&tmpPos,oDetParams.tilt);

					rayStruct rayTest=this->rayList[rayListIndex];
					//posMinOffset=this->rayList[rayListIndex].position-offset;
					//indexFloat=MatrixInv*posMinOffset;
					// subtract half a pixel (0.5*scale.x). This way the centre of our pixels do not lie on the edge of the aperture but rather half a pixel inside...
					// then round to nearest neighbour
					//index.x=floor((indexFloat.x-0.5*scale.x)/scale.x+0.5);
					index.x=floor((tmpPos.x)/scale.x);
					index.y=floor((tmpPos.y)/scale.y);
					index.z=floor((tmpPos.z)/scale.z);
					
					// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
					if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
					{
						// use this ray only if it agrees with the ignoreDepth
						if ( this->rayList[rayListIndex].depth > oDetParams.ignoreDepth )
						{
							hitNr++;
							(l_IntensityImagePtr->getIntensityPtr())[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+=this->rayList[rayListIndex].flux;								
						}
					}
				}
			}


			//for (unsigned long long j=0;j<rayListLength;j++)
			//{
			//	posMinOffset=this->rayList[j].position-offset;
			//	indexFloat=MatrixInv*posMinOffset;
			//	index.x=floor(indexFloat.x+0.5);
			//	index.y=floor(indexFloat.y+0.5);
			//	index.z=floor(indexFloat.z+0.5);
			//	
			//	// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
			//	if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
			//	{
			//		hitNr++;
			//		(l_IntensityImagePtr->getIntensityPtr())[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+=this->rayList[j].flux;
			//	}
			//	//else
			//	//{
			//	//	std::cout <<  "ray number " << j << " did not hit target." << "x: " << rayList[j].position.x << ";y: " << rayList[j].position.y << "z: " << rayList[j].position.z << ";geometryID " << rayList[j].currentGeometryID << std::endl;
			//	//}
			//}
		}
		else
		{
			std::cout << "error in GeometricRayField.convert2Intensity(): partial coherence not implemented yet" << std::endl;
			return FIELD_ERR;
		}

	}
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << " " << msecs <<"ms to process " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

	std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays in target" << std::endl;

	return FIELD_NO_ERR;
};

/**
 * \detail convert2ScalarField 

 *
 * \param[in] Field* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRayField::convert2ScalarField(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in GeometricRayField.convert2ScalarField(): conversion to scalar field not yet implemented" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail convert2VecField 

 *
 * \param[in] Field* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRayField::convert2VecField(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in GeometricRayField.convert2VecField(): conversion to vectorial field not yet implemented" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail convert2PhaseSpaceField 

 *
 * \param[in] Field* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRayField::convert2PhaseSpace(Field* imagePtr, detParams &oDetParams)
{
	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;

	// start timing
	start=clock();

	detPhaseSpaceParams *oDetPhaseSpaceParamsPtr=static_cast<detPhaseSpaceParams*>(&oDetParams);

	//long2 l_GPUSubsetDim=calcSubsetDim();
	// cast the image to an IntensityField
	PhaseSpaceField* l_PhaseSpacePtr=dynamic_cast<PhaseSpaceField*>(imagePtr);
	if (l_PhaseSpacePtr == NULL)
	{
		std::cout << "error in GeometricRayField.convert2PhaseSpaceField(): imagePtr is not of type IntensityField" << std::endl;
		return FIELD_ERR;
	}
		
	/************************************************************************************************************************** 
	* the idea in calculating the flux per pixel is as following:
	* first we create unit vectors along global coordinate axis. 
	* Then we scale this vectors with the scaling of the respective pixel.
	* Then we rotate and translate these vectors into the local coordinate system of the IntensityField
	* Finally we solve the equation system that expresses the ray position in terms of these rotated and scaled vectors.
	* floor() of the coefficients of these vectors gives the indices we were looking for                                      
	****************************************************************************************************************************/
	double3 scale=l_PhaseSpacePtr->getParamsPtr()->scale;
	long3 nrPixels=l_PhaseSpacePtr->getParamsPtr()->nrPixels;
	scale.z=2*oDetParams.apertureHalfWidth.z/nrPixels.z; // we need to set this here as the IntensityField coming from the Detector is set for PseudoBandwidth...

	// create unit vectors
	double3 t_ez = make_double3(0,0,1);
	double3 t_ey=make_double3(0,1,0);
	double3 t_ex=make_double3(1,0,0);
	// transform unit vectors into local coordinate system of IntensityField
	rotateRay(&t_ez,oDetParams.tilt);
	rotateRay(&t_ey,oDetParams.tilt);
	rotateRay(&t_ex,oDetParams.tilt);

	// the origin of the IntensityField is at the outer edge of the detector rather than at the origin
	double3 offset;
	offset=oDetParams.root-oDetParams.apertureHalfWidth.x*t_ex;//+0.5*l_IntensityImagePtr->getParamsPtr()->scale*t_ex;
	offset=offset-oDetParams.apertureHalfWidth.y*t_ey;//+0.5*l_IntensityImagePtr->getParamsPtr()->scale*t_ey;
	offset=offset-oDetParams.apertureHalfWidth.z*t_ez;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ez;

	short solutionIndex;

	unsigned long long hitNr=0;

	double3 posMinOffset;
	double3 indexFloat;
	long3 index;

	if (this->rayParamsPtr->coherence==1) // sum coherently
	{
		std::cout << "error in GeometricRayField.convert2PhaseSpaceField(): coherent conversion is not defined!!" << std::endl;
		return FIELD_ERR; //matrix singular
	}
	else 
	{
		if (this->rayParamsPtr->coherence == 0)// sum incoherently
		{
			unsigned long long hitNr=0;

			for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
			{
				for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
				{
					// calc spatial coordinates
					unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
				    // transform to local coordinate system
				    double3 tmpPos=this->rayList[rayListIndex].position-offset;
				    rotateRayInv(&tmpPos,oDetParams.tilt);

				    index.x=floor((tmpPos.x)/scale.x);
				    index.y=floor((tmpPos.y)/scale.y);
				    index.z=floor((tmpPos.z)/scale.z);

					// calc directional coordinates
					// calc the angle of the current ray with respect to the local coordinate axis
					// we assume that we are interested in angles with respect to x- and y-axis
					// calc projection of ray onto local x-axis
					double t_x=dot(t_ex,this->rayList[rayListIndex].direction);
					// remove x-component from ray
					double3 t_ray_y=normalize(this->rayList[rayListIndex].direction-t_x*t_ex);
					// calc rotation angle around x with respect to z axis
					double phi_x =acos(dot(t_ray_y,t_ez));
					// calc projection of ray onto local y-axis
					double t_y=dot(t_ey,this->rayList[rayListIndex].direction);
					// in order to get the sign right we need to check the sign of the projection on th y-axis
					if (t_y>0)
						phi_x=-phi_x;
					// remove y-component from ray
					double3 t_ray_x=normalize(this->rayList[rayListIndex].direction-t_y*t_ey);
					// calc rotation angle around y with respect to z axis
					double phi_y=acos(dot(t_ray_x,t_ez));
					// in order to get the sign right we need to check the sign of the projection on th y-axis
					if (t_x>0)
						phi_y=-phi_y;
					double2 index_dir_float=make_double2((phi_x+oDetPhaseSpaceParamsPtr->dirHalfWidth.x-l_PhaseSpacePtr->getParamsPtr()->scale_dir.x/2)/l_PhaseSpacePtr->getParamsPtr()->scale_dir.x,(phi_y+oDetPhaseSpaceParamsPtr->dirHalfWidth.y-l_PhaseSpacePtr->getParamsPtr()->scale_dir.y/2)/l_PhaseSpacePtr->getParamsPtr()->scale_dir.y);
					ulong2 index_dir=make_ulong2(floor(index_dir_float.x+0.5),floor(index_dir_float.y+0.5));


					// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
					if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) && ( (index_dir.x<oDetPhaseSpaceParamsPtr->detPixel_PhaseSpace.x)&&(index_dir.x>=0) ) && ( (index_dir.y<oDetPhaseSpaceParamsPtr->detPixel_PhaseSpace.y)&&(index.y>=0) ) )
					{
						hitNr++;
						//(l_PhaseSpacePtr->getPhaseSpacePtr())[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+=this->rayList[rayListIndex].flux;
						double t_flux=l_PhaseSpacePtr->getPix(make_ulong2(index.x,index.y),index_dir)+this->rayList[rayListIndex].flux;
						l_PhaseSpacePtr->setPix(make_ulong2(index.x,index.y),index_dir,t_flux);
					}
				}
			}


			//for (unsigned long long j=0;j<rayListLength;j++)
			//{
			//	posMinOffset=this->rayList[j].position-offset;
			//	indexFloat=MatrixInv*posMinOffset;
			//	index.x=floor(indexFloat.x+0.5);
			//	index.y=floor(indexFloat.y+0.5);
			//	index.z=floor(indexFloat.z+0.5);
			//	
			//	// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
			//	if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
			//	{
			//		hitNr++;
			//		(l_PhaseSpacePtr->getIntensityPtr())[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+=this->rayList[j].flux;
			//	}
			//	//else
			//	//{
			//	//	std::cout <<  "ray number " << j << " did not hit target." << "x: " << rayList[j].position.x << ";y: " << rayList[j].position.y << "z: " << rayList[j].position.z << ";geometryID " << rayList[j].currentGeometryID << std::endl;
			//	//}
			//}
			std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays in target" << std::endl;
		}
		else
		{
			std::cout << "error in GeometricRayField.convert2PhaseSpaceField(): partial coherence not implemented yet" << std::endl;
			return FIELD_ERR;
		}

	}
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << " " << msecs <<"ms to process " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

	return FIELD_NO_ERR;
};

/**
 * \detail convert2RayData 

 *
 * \param[in] Field** imagePtrPtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRayField::convert2RayData(Field** imagePtrPtr, detParams &oDetParams)
{
	GeometricRayField* l_ptr;
	// if there is no image yet, create one
	if (*imagePtrPtr == NULL)
	{
		*imagePtrPtr=new GeometricRayField(this->rayListLength);
		l_ptr=dynamic_cast<GeometricRayField*>(*imagePtrPtr);
		(*imagePtrPtr)->setSubsetHeightMax(this->GPU_SUBSET_HEIGHT_MAX);
		(*imagePtrPtr)->setSubsetWidthMax(this->GPU_SUBSET_WIDTH_MAX);
	}
	else
	{
		l_ptr=dynamic_cast<GeometricRayField*>(*imagePtrPtr);
		if ( l_ptr->rayListLength < this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width )
		{
			std::cout << "error in GeometricRayField.convert2RayData(): dimensions of image does not fit dimensions of raylist subset" << std::endl;
			return FIELD_ERR;
		}
	}
	// copy the rayList and the respective parameters
	memcpy(l_ptr->getRayList(), this->rayList, (GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX)*sizeof(rayStruct));
	rayFieldParams *l_rayParamsPtr=new rayFieldParams();
	memcpy(l_rayParamsPtr,this->rayParamsPtr, sizeof(rayFieldParams));
	l_ptr->setParamsPtr(l_rayParamsPtr);
	
	return FIELD_NO_ERR;
};

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] FieldParseParamStruct &parseResults_Src, MaterialParseParamStruct *parseResults_MatPtr
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRayField::processParseResults(FieldParseParamStruct &parseResults_Src, parseGlassResultStruct* parseResults_GlassPtr)
{
	if ( (parseResults_Src.rayDirDistr == RAYDIR_GRID_RECT) || (parseResults_Src.rayDirDistr == RAYDIR_GRID_RAD) )
	{
		std::cout << "error in GeometricRayField.processParseResults(): RAYDIR_GRID_RAD and RAYDIR_GRID_RECT are not allowed for geometric ray fields" << std::endl;
		return FIELD_ERR;
	}
//	this->rayParamsPtr=new rayFieldParams;
	this->rayParamsPtr->translation=parseResults_Src.root;
	double rotX=parseResults_Src.tilt.x;
	double rotY=parseResults_Src.tilt.y;
	double rotZ=parseResults_Src.tilt.z;
	double3x3 MrotX, MrotY, MrotZ, Mrot;
	MrotX=make_double3x3(1,0,0, 0,cos(rotX),-sin(rotX), 0,sin(rotX),cos(rotX));
	MrotY=make_double3x3(cos(rotY),0,sin(rotY), 0,1,0, -sin(rotY),0,cos(rotY));
	MrotZ=make_double3x3(cos(rotZ),-sin(rotZ),0, sin(rotZ),cos(rotZ),0, 0,0,1);
	Mrot=MrotX*MrotY;
	this->rayParamsPtr->Mrot=Mrot*MrotZ;
	this->rayParamsPtr->tilt=parseResults_Src.tilt;
	this->rayParamsPtr->coherence=parseResults_Src.coherence;
	this->rayParamsPtr->rayPosStart=make_double3(-parseResults_Src.apertureHalfWidth1.x,
									 -parseResults_Src.apertureHalfWidth1.y,
									 0);
	this->rayParamsPtr->rayPosEnd=make_double3(parseResults_Src.apertureHalfWidth1.x,
									 parseResults_Src.apertureHalfWidth1.y,
									 0);
	this->rayParamsPtr->rayDirection=parseResults_Src.rayDirection;
	this->rayParamsPtr->width=parseResults_Src.width;
	this->rayParamsPtr->height=parseResults_Src.height;//*parseResults_Src.nrRayDirections.x*parseResults_Src.nrRayDirections.y;
	this->rayParamsPtr->widthLayout=parseResults_Src.widthLayout;
	this->rayParamsPtr->heightLayout=parseResults_Src.heightLayout;
	this->rayParamsPtr->totalLaunch_height=parseResults_Src.height;
	this->rayParamsPtr->totalLaunch_width=parseResults_Src.width;
	this->rayParamsPtr->flux=parseResults_Src.power;
	this->rayParamsPtr->lambda=parseResults_Src.lambda*1e-3;
	this->rayParamsPtr->posDistrType=parseResults_Src.rayPosDistr;
	this->rayParamsPtr->dirDistrType=parseResults_Src.rayDirDistr;
	this->rayParamsPtr->alphaMax=parseResults_Src.alphaMax;
	this->rayParamsPtr->alphaMin=parseResults_Src.alphaMin;
	this->rayParamsPtr->nrRayDirections=make_ulong2(1,1);//parseResults_Src.nrRayDirections;
	this->rayParamsPtr->importanceAreaHalfWidth=parseResults_Src.importanceAreaHalfWidth;
	this->rayParamsPtr->importanceAreaRoot=parseResults_Src.importanceAreaRoot;
	this->rayParamsPtr->importanceAreaTilt=parseResults_Src.importanceAreaTilt;
	this->rayParamsPtr->importanceAreaApertureType=parseResults_Src.importanceAreaApertureType;


	// create refracting material
	this->setMaterialListLength(1);
	MaterialRefracting* l_matRefrPtr=new MaterialRefracting();
	this->setMaterial(l_matRefrPtr, 0);

	// define params of the refracting material
	MatRefracting_params refrParams;
	MatRefracting_DispersionParams* glassDispersionParamsPtr;
	MatRefracting_DispersionParams* immersionDispersionParamsPtr;

	/* create immersion material */
	switch (parseResults_Src.materialParams.matType)
	{
		case MT_REFRMATERIAL:
			// process the parsing results for the refracting material
			l_matRefrPtr->processParseResults(parseResults_Src.materialParams, parseResults_GlassPtr, parseResults_GlassPtr);

			break;
		default:
			// create a refracting material with n=1 per default

			glassDispersionParamsPtr=new MatRefracting_DispersionParams();
			/* default we set the immersion material to user defined refractive index of 1 */
			refrParams.n1=1;
			//refrParams.n2=parseResults->geometryParams[k].materialParams.nRefr.y;
			glassDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
			glassDispersionParamsPtr->lambdaMin=0;
			l_matRefrPtr->setParams(refrParams);
			glassDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
			l_matRefrPtr->setGlassDispersionParams(glassDispersionParamsPtr);
			immersionDispersionParamsPtr=new MatRefracting_DispersionParams();
			immersionDispersionParamsPtr->lambdaMax=DOUBLE_MAX;
			immersionDispersionParamsPtr->lambdaMin=0;
			immersionDispersionParamsPtr->dispersionFormula=MAT_DISPFORMULA_NODISP;
			l_matRefrPtr->setImmersionDispersionParams(immersionDispersionParamsPtr); // we don't use an immersion medium here but we need to set some value...
			std::cout <<"warning in GeometricRayField.processParseResults(): unknown material. Rafracting material with n=1 assumed." << std::endl;
			break;
	}
	return FIELD_NO_ERR;
};

void GeometricRayField::setSimMode(SimMode &simMode)
{
	simMode=SIM_GEOM_RT;
};

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
fieldError  GeometricRayField::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams)
{
	Parser_XML l_parser;

    this->setSimMode(simParams.simMode);

	// call base class function
	if (FIELD_NO_ERR != RayField::parseXml(field, fieldVec, simParams))
	{
		std::cout << "error in GeometricRayField.parseXml(): RayField.parseXml()  returned an error." << std::endl;
		return FIELD_ERR;
	}

	if ( (this->rayParamsPtr->dirDistrType == RAYDIR_GRID_RECT) || (this->rayParamsPtr->dirDistrType == RAYDIR_GRID_RAD) )
	{
		std::cout << "error in GeometricRayField.parseXml(): RAYDIR_GRID_RAD and RAYDIR_GRID_RECT are not allowed for geometric ray fields" << std::endl;
		return FIELD_ERR;
	}

	this->rayParamsPtr->totalLaunch_height=this->rayParamsPtr->height;
	this->rayParamsPtr->totalLaunch_width=this->rayParamsPtr->width;
	this->rayParamsPtr->nrRayDirections=make_ulong2(1,1);

	this->getParamsPtr()->pseudoBandwidth=0; // default to zero
	this->getParamsPtr()->nrPseudoLambdas=1; // default to one

	return FIELD_NO_ERR;
};