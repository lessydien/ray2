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

/**\file PathTracingRayField.cpp
* \brief Rayfield for geometric raytracing
* 
*           
* \author Mauch
*/
#include <omp.h>
#include "PathTracingRayField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include "Converter.h"
#include "MatlabInterface.h"
#include <ctime>

using namespace optix;


/**
 * \detail setRay 

 *
 * \param[in] rayStruct_PathTracing_PathTracing ray, unsigned long long index
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathTracingRayField::setRay(rayStruct_PathTracing ray, unsigned long long index)
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
 * \return rayStruct_PathTracing*
 * \sa 
 * \remarks 
 * \author Mauch
 */
rayStruct_PathTracing* PathTracingRayField::getRay(unsigned long long index)
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
 * \return rayStruct_PathTracing*
 * \sa 
 * \remarks 
 * \author Mauch
 */
rayStruct_PathTracing* PathTracingRayField::getRayList(void)
{
	return &rayList[0];	
};

/**
 * \detail setRayList 

 *
 * \param[in] rayStruct_PathTracing* rayStruct_PathTracingPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void PathTracingRayField::setRayList(rayStruct_PathTracing* rayStruct_PathTracingPtr)
{
	if (this->rayList!=NULL)
		free (rayList);
	this->rayList=rayStruct_PathTracingPtr;
};

/**
 * \detail copyRayList 

 *
 * \param[in] rayStruct_PathTracing *data, long long length
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathTracingRayField::copyRayList(rayStruct_PathTracing *data, long long length)
{
	if (length > this->rayListLength)
	{
		std::cout << "error in PathTracingRayField.copyRayList(): subset dimensions exceed rayLIst dimension" << std::endl;
		return FIELD_ERR;
	}
	memcpy(this->rayList, data, this->rayParamsPtr->GPUSubset_width*sizeof(rayStruct_PathTracing));

	return FIELD_NO_ERR;
};

/**
 * \detail copyRayListSubset 

 *
 * \param[in] rayStruct_PathTracing *data, long2 launchOffset, long2 subsetDim
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathTracingRayField::copyRayListSubset(rayStruct_PathTracing *data, long2 launchOffset, long2 subsetDim)
{
//	long2 testOffset=launchOffset;
//	long2 testDim=subsetDim;
	//  ----memory range of completed lines---- + ---memory range blocks in given line---
	if (launchOffset.y*this->rayParamsPtr->width+(subsetDim.x+launchOffset.x)*subsetDim.y > this->rayListLength)
	{
		std::cout << "error in PathTracingRayField.copyRayListSubset(): subset dimensions exceed rayLIst dimension" << std::endl;
		return FIELD_ERR;
	}
	// copy the ray list line per line
	for (long long jy=0;jy<subsetDim.y;jy++)
	{
		unsigned long long testIndex=launchOffset.y*this->rayParamsPtr->width+launchOffset.x+jy*this->rayParamsPtr->width;
		//                     memory range of completed lines + offsetX + number of line in current block*width of complete rayblock // we always allocate the max buffer on GPU, therefore we always need to adress the start of the line in this maximum buffer...
		memcpy(&(this->rayList[launchOffset.y*this->rayParamsPtr->width+launchOffset.x+jy*this->rayParamsPtr->width]), &data[jy*GPU_SUBSET_WIDTH_MAX], subsetDim.x*sizeof(rayStruct_PathTracing));
	}
	//memcpy(this->rayList, data, length*sizeof(rayStruct_PathTracing));
	return FIELD_NO_ERR;
};

/**
 * \detail calcSubsetDim 
 *
 * \param[in] void
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
long2 PathTracingRayField::calcSubsetDim()
{
	unsigned long long width=this->rayParamsPtr->width*this->rayParamsPtr->height*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;

	long2 l_GPUSubsetDim;

	// calc launch_width of current launch
	long long restWidth=width-this->rayParamsPtr->launchOffsetX-this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;
	// if the restWidth is smaller than the maximum subset-width. Take restWidth
	if (restWidth < this->rayParamsPtr->GPUSubset_width)
	{
		l_GPUSubsetDim.x=restWidth;
	}
	else
	{
		l_GPUSubsetDim.x=this->rayParamsPtr->GPUSubset_width;
	}
	// we need to set to one
	l_GPUSubsetDim.y=1;
	this->rayParamsPtr->GPUSubset_height=l_GPUSubsetDim.y;
	this->rayParamsPtr->GPUSubset_width=l_GPUSubsetDim.x;
	return l_GPUSubsetDim;
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
fieldError PathTracingRayField::createCPUSimInstance()
{
	if (this->rayList != NULL)
	{
		delete this->rayList;
		this->rayListLength=0;
		rayList=NULL;
	}
	rayList=(rayStruct_PathTracing*) malloc(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX*sizeof(rayStruct_PathTracing));
	if (!rayList)
	{
		std::cout << "error in PathTracingRayField.createLayoutInstance(): memory for rayList could not be allocated. try to reduce ray tiling size" << std::endl;
		return FIELD_ERR;
	}
	this->rayListLength=GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX;

	unsigned int l_launch_width, l_launch_height, l_offsetX, l_offsetY;
	// calc the dimensions of the simulation subset
	if ( this->rayParamsPtr->width*this->rayParamsPtr->height*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y < GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX ) 
	{
		this->rayParamsPtr->GPUSubset_width=this->rayParamsPtr->width*this->rayParamsPtr->height*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;
	}
	else
	{
		this->rayParamsPtr->GPUSubset_width=GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX;
	}
	// differential rays are traced in a 1D launch, so we set the subset height to the height of the ray field. This is necessary as geometric ray fields are traced in 2D launches and SimAssistant.doSim() doesn't know wether it is simulating differential or geometric rayfields !!
	this->rayParamsPtr->GPUSubset_height=1;//this->rayParamsPtr->height;

	l_offsetX=0;
	l_offsetY=0;
	this->rayParamsPtr->launchOffsetX=l_offsetX;
	this->rayParamsPtr->launchOffsetY=l_offsetY;

	return FIELD_NO_ERR;
};

/* functions for GPU usage */

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
fieldError PathTracingRayField::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	if (FIELD_NO_ERR != RayField::createOptixInstance(context, output_buffer_obj, seed_buffer_obj))
	{
		std::cout <<"error in PathTracingRayField.createOptixInstance(): RayField.creatOptiXInstance() returned an error." << std::endl;
		return FIELD_ERR;
	}

	RTvariable output_buffer;
    /* variables for ray gen program */

	RTvariable params;

	/* declare result buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "output_buffer", &output_buffer ), context ))
		return FIELD_ERR;
    /* Render result buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_OUTPUT, &output_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( output_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( output_buffer_obj, sizeof(rayStruct_PathTracing) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( output_buffer_obj, GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_WIDTH_MAX ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( output_buffer, output_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare variables */
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "params", &params ), context ))
		return FIELD_ERR;

	this->rayParamsPtr->nImmersed=this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);

	if (FIELD_NO_ERR!=this->createCPUSimInstance())
	{
		std::cout <<"error in PathTracingRayField.createOptixInstance(): create CPUSimInstance() returned an error." << std::endl;
		return FIELD_ERR;
	}

	//// calc the dimensions of the simulation subset
	//if ( this->rayParamsPtr->width*this->rayParamsPtr->height*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y < GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX ) 
	//{
	//	this->rayParamsPtr->GPUSubset_width=this->rayParamsPtr->width*this->rayParamsPtr->height*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;
	//}
	//else
	//{
	//	this->rayParamsPtr->GPUSubset_width=GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX;
	//}
	//// pathTracing Rays rays are traced in a 1D launch, so we set the subset height to the height of the ray field. This is necessary as geometric ray fields are traced in 2D launches and SimAssistant.doSim() doesn't know wether it is simulating differential or geometric rayfields !!
	//this->rayParamsPtr->GPUSubset_height=1;//this->rayParamsPtr->height;

	//this->rayParamsPtr->launchOffsetX=0;//l_offsetX;
	//this->rayParamsPtr->launchOffsetY=0;//l_offsetY;

	// transfer the dimension of the whole simulation to GPU
	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(rayFieldParams), this->rayParamsPtr), context) )
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
fieldError PathTracingRayField::initCPUSubset()
{
	clock_t start, end;
	double msecs=0;

	// check wether we will be able to fit all the rays into our raylist. If not some eror happened earlier and we can not proceed...
	if ((this->rayParamsPtr->GPUSubset_width)<=this->rayListLength)
	{

		// see if there are any rays to create	
		if (this->rayParamsPtr->GPUSubset_width >= 1)
		{
			// width of ray field in physical dimension
			double physWidth=this->rayParamsPtr->rayPosEnd.x-this->rayParamsPtr->rayPosStart.x;
			// height of ray field in physical dimension
			double physHeight=this->rayParamsPtr->rayPosEnd.y-this->rayParamsPtr->rayPosStart.y;
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

			// create all the rays
			for(signed long long jx=0;jx<this->rayParamsPtr->GPUSubset_width;jx++)
			{
				uint32_t x_l[5];
				RandomInit(this->rayList[jx].currentSeed, x_l); // seed random generator

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

				rayStruct_PathTracing rayData;
				rayData.flux=this->rayParamsPtr->flux;
				rayData.depth=0;	
				rayData.position.z=this->rayParamsPtr->rayPosStart.z;
				rayData.running=true;
				rayData.currentGeometryID=0;
				rayData.lambda=this->rayParamsPtr->lambda;
				rayData.nImmersed=1;//this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);
				rayData.opl=0;
				rayData.result=0;
				rayData.secondary=false;
				rayData.secondary_nr=1;

				// map on one dimensional index
				//unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*( floorf(this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y/this->rayParamsPtr->GPUSubset_width+1)*this->rayParamsPtr->GPUSubset_width);
				unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;

				//std::cout << "iGes: " << iGes << std::endl;

				// calc position indices from 1D index
				unsigned long long iPosX=floorf(iGes/(this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y));
				unsigned long long iPosY=floorf(iPosX/this->rayParamsPtr->width);
				iPosX=iPosX % this->rayParamsPtr->width;
				

				// calc direction indices from 1D index
				unsigned long long iDirX=(iGes-iPosX*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y-iPosY*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y*this->rayParamsPtr->width) % this->rayParamsPtr->nrRayDirections.x;
				unsigned long long iDirY=floorf((iGes-iPosX*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y-iPosY*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y*this->rayParamsPtr->width)/this->rayParamsPtr->nrRayDirections.x);

				// declare variables for placing a ray randomly inside an ellipse
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
							deltaW= (physWidth)/(this->rayParamsPtr->width);
						if (this->rayParamsPtr->height>0)
							// multiple directions per point are listed in y-direction. Therefore the physical height of the rayfield is different from the height of the ray list. This has to be considered here...
							deltaH= (physHeight)/(this->rayParamsPtr->height);
						rayData.position.x=this->rayParamsPtr->rayPosStart.x+deltaW/2+iPosX*deltaW;
						rayData.position.y=this->rayParamsPtr->rayPosStart.y+deltaH/2+iPosY*deltaH;
						break;
					case RAYPOS_RAND_RECT:
						if ( (iGes % (this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y)) == 0 )
						{
							rayData.position.x=(Random(x_l)-0.5)*physWidth+rayFieldCentre.x;
							rayData.position.y=(Random(x_l)-0.5)*physHeight+rayFieldCentre.y;
							this->oldPosition=rayData.position; // save current position for next launch
						}
						else
						{
							rayData.position=this->oldPosition; // we use tha same position as before...
						}
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
						R=(deltaRx/2+deltaRx*iPosX)*(deltaRy/2+deltaRy*iPosX)/sqrt(pow((deltaRy/2+deltaRy*iPosX)*cos(deltaPhi/2+deltaPhi*iPosY),2)+pow((deltaRx/2+deltaRx*iPosX)*sin(deltaPhi/2+deltaPhi*iPosY),2));							
						if (deltaRy==0)
							R=0;
						// now calc rectangular coordinates from polar coordinates
						rayData.position.x=cos(deltaPhi/2+deltaPhi*iPosY)*R;
						rayData.position.y=sin(deltaPhi/2+deltaPhi*iPosY)*R;

						break;
					case RAYPOS_RAND_RAD:
						// place a point uniformingly randomly inside the importance area
						if ( (iGes % this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y) == 0 )
						{
							theta=2*PI*Random(x_l);
							r=sqrt(Random(x_l));
						}
						ellipseX=physWidth*r*cos(theta);
						ellipseY=physHeight*r*sin(theta);
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
					default:
						rayData.position=make_double3(0,0,0);
						std::cout << "error in PathTraceRayField.initCPUSubset: unknown distribution of rayposition" << std::endl;
						// report error
						break;
				}
				// transform rayposition into global coordinate system
				rayData.position=this->rayParamsPtr->Mrot*rayData.position+this->rayParamsPtr->translation;


				switch (this->rayParamsPtr->dirDistrType)
				{
					case RAYDIR_UNIFORM:
						rayData.direction=this->rayParamsPtr->rayDirection;
						break;
					case RAYDIR_RAND:
						// create points inside importance area to randomly distribute ray direction
						rayAngleCentre=make_double3((this->rayParamsPtr->alphaMax.x+this->rayParamsPtr->alphaMin.x)/2,(this->rayParamsPtr->alphaMax.y+this->rayParamsPtr->alphaMin.y)/2,0);
						impAreaHalfWidth.x=(tan(this->rayParamsPtr->alphaMax.x)-tan(this->rayParamsPtr->alphaMin.x))/2;
						impAreaHalfWidth.y=(tan(this->rayParamsPtr->alphaMax.y)-tan(this->rayParamsPtr->alphaMin.y))/2;
						dirImpAreaCentre=make_double3(0,0,1);
						rotateRay(&dirImpAreaCentre, rayAngleCentre);
						// the centre of the importance area is the root of the current geometry + the direction to the imp area centre normalized such that the importance area is 1mm away from the current geometry
						impAreaRoot=make_double3(0,0,0)+dirImpAreaCentre/dot(make_double3(0,0,1), dirImpAreaCentre);
						// now distribute points inside importance area
						theta=2*PI*Random(x_l);
						r=sqrt(Random(x_l));
						impAreaX=impAreaHalfWidth.x*r*cos(theta);
						impAreaY=impAreaHalfWidth.y*r*sin(theta);
						tmpPos=impAreaRoot+impAreaX*make_double3(1,0,0)+impAreaY*make_double3(0,1,0);
						rayData.direction=normalize(tmpPos-make_double3(0,0,0));
						// transform raydirection into global coordinate system
						rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
						// save seed for next randomization
						rayData.currentSeed=x[4];
						break;
					case RAYDIR_RANDIMPAREA:
						if (this->rayParamsPtr->importanceAreaApertureType==AT_RECT)
						{
							// place temporal point uniformingly randomly inside the importance area
							impAreaX=(Random(x_l)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.x;
							impAreaY=(Random(x_l)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.y; 
						}
						else
						{
							if (this->rayParamsPtr->importanceAreaApertureType==AT_ELLIPT)
							{
								theta=2*PI*Random(x_l);
								r=sqrt(Random(x_l));
								impAreaX=this->rayParamsPtr->importanceAreaHalfWidth.x*r*cos(theta);
								impAreaY=this->rayParamsPtr->importanceAreaHalfWidth.y*r*sin(theta);
							}
							else
							{
								std::cout << "error in PathTraceRayField.initCPUSubset: importance area for defining ray directions of source is only allowed with objects that have rectangular or elliptical apertures" << std::endl;
								// report error
								//return FIELD_ERR; // return is not allowed inside opneMP block!!!
							}
						}

						impAreaAxisX=make_double3(1,0,0);
						impAreaAxisY=make_double3(0,1,0);
						rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
						rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

						tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
						rayData.direction=normalize(tmpPos-rayData.position);
						break;
						
						break;
					case RAYDIR_GRID_RECT:
						// calc increment along x- and y-direction
						if (this->rayParamsPtr->nrRayDirections.x>0)
							deltaW= (2*this->rayParamsPtr->importanceAreaHalfWidth.x)/double(this->rayParamsPtr->nrRayDirections.x);
						if (this->rayParamsPtr->nrRayDirections.y>0)
							deltaH= (2*this->rayParamsPtr->importanceAreaHalfWidth.y)/double(this->rayParamsPtr->nrRayDirections.y);
						impAreaX=-this->rayParamsPtr->importanceAreaHalfWidth.x+deltaW/2+iDirX*deltaW; 
						impAreaY=-this->rayParamsPtr->importanceAreaHalfWidth.y+deltaH/2+iDirY*deltaH; 
						impAreaAxisX=make_double3(1,0,0);
						impAreaAxisY=make_double3(0,1,0);
						rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
						rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

						tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
						rayData.direction=normalize(tmpPos-rayData.position);
						break;
					//case RAYDIR_RAND_RECT:
					//	// place temporal point uniformingly randomly inside the importance area
					//	impAreaX=(Random(x_l)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.x;
					//	impAreaY=(Random(x_l)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.y; 
					//	impAreaAxisX=make_double3(1,0,0);
					//	impAreaAxisY=make_double3(0,1,0);
					//	rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
					//	rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

					//	tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
					//	rayData.direction=normalize(tmpPos-rayData.position);
					//	break;
					case RAYDIR_GRID_RAD:
						// calc increment along radial and angular direction
						if (this->rayParamsPtr->nrRayDirections.x>0)
						{
							deltaRx= (this->rayParamsPtr->importanceAreaHalfWidth.x)/double(this->rayParamsPtr->nrRayDirections.x);
							deltaRy= (this->rayParamsPtr->importanceAreaHalfWidth.y)/double(this->rayParamsPtr->nrRayDirections.x);
						}
						if (this->rayParamsPtr->nrRayDirections.y>0)
							deltaPhi= (2*PI)/this->rayParamsPtr->nrRayDirections.y;
						// calc r(phi) for given phi and radii of ellipse. see http://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center for reference
						R=(deltaRx/2+deltaRx*iDirY)*(deltaRy/2+deltaRy*iDirY)/sqrt(pow((deltaRy/2+deltaRy*iDirY)*cos(deltaPhi/2+deltaPhi*iDirX),2)+pow((deltaRx/2+deltaRx*iDirY)*sin(deltaPhi/2+deltaPhi*iDirX),2));
						if ( (deltaRx==0) || (deltaRy==0) )
							R=0;
						// now calc rectangular coordinates from polar coordinates
						impAreaX=cos(deltaPhi/2+deltaPhi*iDirX)*R;
						impAreaY=sin(deltaPhi/2+deltaPhi*iDirX)*R;

						impAreaAxisX=make_double3(1,0,0);
						impAreaAxisY=make_double3(0,1,0);
						rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
						rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

						tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
						rayData.direction=normalize(tmpPos-rayData.position);
						break;
					//case RAYDIR_RAND_RAD:
					//	// place temporal point uniformingly randomly inside the importance area
					//	// place a point uniformingly randomly inside the importance area
					//	theta=2*PI*Random(x_l);
					//	r=sqrt(Random(x_l));
					//	impAreaX=this->rayParamsPtr->importanceAreaHalfWidth.x*r*cos(theta);
					//	impAreaY=this->rayParamsPtr->importanceAreaHalfWidth.y*r*sin(theta);
					//	impAreaAxisX=make_double3(1,0,0);
					//	impAreaAxisY=make_double3(0,1,0);
					//	rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
					//	rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

					//	tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
					//	rayData.direction=normalize(tmpPos-rayData.position);
					//	break;

					default:
						rayData.direction=make_double3(0,0,0);
						std::cout << "error in PathTracingRayField.initCPUSubset: unknown raydirection distribution" << std::endl;
						// report error
						break;
				}

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

				rayData.currentSeed=(uint)BRandom(x);
				this->setRay(rayData,(unsigned long long)(jx));
				//increment directions counter
			}
} // end omp
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize " << this->rayParamsPtr->GPUSubset_width*this->rayParamsPtr->GPUSubset_height << " rays." << std::endl;

		}
		else if(this->rayParamsPtr->width*this->rayParamsPtr->height<1)
		{
			//not Possible. Report error or set n=-n
			std::cout << "error in PathTracingRayField.initCPUInstance: negative raynumber" << std::endl;
		}
		this->update=false;
	}	// end if GPUsubsetwidth*height<rayListLength
	else
	{
		std::cout << "error in DiffRayField.initCPUInstance: rayList is smaller than simulation subset" << std::endl;
		return FIELD_ERR;
	}
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
fieldError PathTracingRayField::traceScene(Group &oGroup, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
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

		omp_set_num_threads(numCPU);

		if (FIELD_NO_ERR!= initCPUSubset())
		{
			std::cout << "error in PathTracingRayField.traceScene: initCPUSubset returned an error" << std::endl;
			return FIELD_ERR;
		}
		
		std::cout << "starting the actual trace..." << std::endl;
#pragma omp parallel default(shared)
{
		#pragma omp for schedule(dynamic, 50)
		//for (signed long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		//{
		for (signed long long j=0; j<this->rayParamsPtr->GPUSubset_width; j++)
		{

			for(;;) // iterative tracing
			{
				if(!this->rayList[j].running) 
					break;
				oGroup.trace(rayList[j]);
			}
			//std::cout << "Iteration " << jy << " running Cur Thread " << omp_get_thread_num() << "Num Threads " << omp_get_num_threads() << "Max Threads " << omp_get_max_threads() << " running" << std::endl;
		}
}
	}
	else
	{
		//RTsize				buffer_width, buffer_height; // get size of output buffer
		void				*data; // pointer to cast output buffer into
 		//rayStruct_PathTracing			*bufferData;

		std::cout << "tracing on GPU." << std::endl;

		initGPUSubset(context, seed_buffer_obj);
		// start current launch
		if (!RT_CHECK_ERROR_NOEXIT( rtContextLaunch1D( (context), 0, this->rayParamsPtr->GPUSubset_width), context))
			return FIELD_ERR;

		// update scene
//		oGroup.updateOptixInstance(context, mode, lambda);
				
//		RT_CHECK_ERROR_NOEXIT( rtContextLaunch2D( context, 0, width, height ) );
		/* unmap output-buffer */
		//RT_CHECK_ERROR_NOEXIT( rtBufferGetSize2D(output_buffer_obj, &buffer_width, &buffer_height) );
		// recast from Optix RTsize to standard int
		//unsigned long long l_bufferWidth = (unsigned long long)(buffer_width);
		//unsigned long long l_bufferHeight = (unsigned long long)(buffer_height);//static_cast<int>(buffer_height);

		if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(output_buffer_obj, &data), context ))
			return FIELD_ERR;
//			end=clock();
		// cast data pointer to format of the output buffer
		//bufferData=(rayStruct_PathTracing*)data;
		//rayStruct_PathTracing test=bufferData[250];
		//SourceList->setRayList((rayStruct_PathTracing*)data);
		//std::cout << "DEBUG: jx=" << jx << " jy=" << jy << std::endl;
		//copyRayListSubset((rayStruct_PathTracing*)data, l_launchOffset, l_GPUSubsetDim);
		if (FIELD_NO_ERR != copyRayList((rayStruct_PathTracing*)data,this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width) )
		{
			std::cout << "error in PathTracingRayField.traceScene(): copyRayList() returned an error" << std::endl;
			return FIELD_NO_ERR;
		}
		
		if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( output_buffer_obj ), context ))
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
fieldError PathTracingRayField::traceStep(Group &oGroup, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
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

//	std::cout << "tracing on " << numCPU << " cores of CPU." << std::endl;

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
				rayStruct_PathTracing test=rayList[rayListIndex];
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
 * \detail writeData2File 

 *
 * \param[in] FILE *hFile, rayDataOutParams outParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
//fieldError PathTracingRayField::writeData2File(FILE *hFile, rayDataOutParams outParams)
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
fieldError PathTracingRayField::write2TextFile(char* filename, detParams &oDetParams)
{
	char t_filename[512];
	sprintf(t_filename, "%s%sPathTracingRayField_%i.txt", filename, PATH_SEPARATOR, oDetParams.subSetNr);

	FILE* hFileOut;
	hFileOut = fopen( t_filename, "w" ) ;
	if (!hFileOut)
	{
		std::cout << "error in PathTracingRayField.write2TextFile(): could not open output file: " << filename << std::endl;
		return FIELD_ERR;
	}
	if (1) // (reducedData==1)
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
//fieldError PathTracingRayField::write2MatFile(char* filename, detParams &oDetParams)
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
//	sprintf(t_filename, "%s%sPathTracingRayField", filename, PATH_SEPARATOR);
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
fieldError PathTracingRayField::convert2Intensity(Field* imagePtr, detParams &oDetParams)
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
		std::cout << "error in PathTracingRayField.convert2Intensity(): imagePtr is not of type IntensityField" << std::endl;
		return FIELD_ERR;
	}
		
	if (this->rayParamsPtr->coherence==1) // sum coherently
	{
		std::cout << "error in PathTracingRayField.convert2Intensity(): coherent summing not implemented yet" << std::endl;
		return FIELD_ERR;

		//complex<double> i_compl=complex<double>(0,1); // define complex number "i"

		//for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
		//{
		//	// map on one dimensional index
		//	//unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*( floorf(this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y/this->rayParamsPtr->GPUSubset_width+1)*this->rayParamsPtr->GPUSubset_width);
		//	unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;

		//	//std::cout << "iGes: " << iGes << std::endl;

		//	// calc position indices from 1D index
		//	unsigned long long iPosX=floorf(iGes/(this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y));
		//	unsigned long long iPosY=floorf(iPosX/this->rayParamsPtr->width);
		//	iPosX=iPosX % this->rayParamsPtr->width;
		//		
		//	double phi=2*PI/this->rayList[jx].lambda*this->rayList[jx].opl;
		//	complex<double> l_U=complex<double>(this->rayList[jx].result*cos(phi),this->rayList[jx].flux*sin(phi));
		//	l_IntensityImagePtr->getComplexAmplPtr()[iPosX+iPosY*nrPixels.x]=l_IntensityImagePtr->getComplexAmplPtr()[iPosX+iPosY*nrPixels.x]+l_U; // create a complex amplitude from the rays flux and opl and sum them coherently
		//}

		//// loop through the pixels and calc intensity from complex amplitudes
		//for (unsigned long long jx=0;jx<nrPixels.x;jx++)
		//{
		//	for (unsigned long long jy=0;jy<nrPixels.y;jy++)
		//	{
		//		for (unsigned long long jz=0;jz<nrPixels.z;jz++)
		//		{
		//			// intensity is square of modulus of complex amplitude
		//			(l_IntensityImagePtr->getIntensityPtr())[jx+jy*nrPixels.x]=pow(abs(l_IntensityImagePtr->getComplexAmplPtr()[jx+jy*nrPixels.x]),2);
		//		}
		//	}
		//}

	}
	else 
	{
		if (this->rayParamsPtr->coherence == 0)// sum incoherently
		{
			unsigned long long hitNr=0;

			for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				// map on one dimensional index
				//unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*( floorf(this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y/this->rayParamsPtr->GPUSubset_width+1)*this->rayParamsPtr->GPUSubset_width);
				unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;

				//std::cout << "iGes: " << iGes << std::endl;

				// calc position indices from 1D index
				unsigned long long iPosX=floorf(iGes/(this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y));
				unsigned long long iPosY=floorf(iPosX/this->rayParamsPtr->width);
				iPosX=iPosX % this->rayParamsPtr->width;

				if (this->rayList[jx].result != 0)
				{
					hitNr++;
					// sum them incoherently and weighted by their number of secondary rays that were launched dureing the trace
					l_IntensityImagePtr->getIntensityPtr()[iPosX+iPosY*l_IntensityImagePtr->getParamsPtr()->nrPixels.x]=l_IntensityImagePtr->getIntensityPtr()[iPosX+iPosY*l_IntensityImagePtr->getParamsPtr()->nrPixels.x]+this->rayList[jx].result/this->rayList[jx].secondary_nr; 
				}
			}
			std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays in target" << std::endl;
		}
		else
		{
			std::cout << "error in PathTracingRayField.convert2Intensity(): partial coherence not implemented yet" << std::endl;
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
 * \detail convert2ScalarField 

 *
 * \param[in] Field* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PathTracingRayField::convert2ScalarField(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in PathTracingRayField.convert2ScalarField(): conversion to scalar field not yet implemented" << std::endl;
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
fieldError PathTracingRayField::convert2VecField(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in PathTracingRayField.convert2VecField(): conversion to vectorial field not yet implemented" << std::endl;
	return FIELD_ERR;
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
fieldError PathTracingRayField::convert2RayData(Field** imagePtrPtr, detParams &oDetParams)
{
	PathTracingRayField* l_ptr;
	// if there is no image yet, create one
	if (*imagePtrPtr == NULL)
	{
		*imagePtrPtr=new PathTracingRayField(this->rayListLength);
		l_ptr=dynamic_cast<PathTracingRayField*>(*imagePtrPtr);
	}
	else
	{
		l_ptr=dynamic_cast<PathTracingRayField*>(*imagePtrPtr);
		if ( l_ptr->rayListLength < this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width )
		{
			std::cout << "error in PathTracingRayField.convert2RayData(): dimensions of image does not fit dimensions of raylist subset" << std::endl;
			return FIELD_ERR;
		}
	}
	// copy the rayList and the respective parameters
	memcpy(l_ptr->getRayList(), this->rayList, (GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX)*sizeof(rayStruct_PathTracing));
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
fieldError PathTracingRayField::processParseResults(FieldParseParamStruct &parseResults_Src, parseGlassResultStruct* parseResults_GlassPtr)
{
	if ( (parseResults_Src.rayDirDistr == RAYDIR_GRID_RECT) || (parseResults_Src.rayDirDistr == RAYDIR_GRID_RAD) )
	{
		std::cout << "error in PathTracingRayField.processParseResults(): RAYDIR_GRID_RAD and RAYDIR_GRID_RECT are not allowed for geometric ray fields" << std::endl;
		return FIELD_ERR;
	}
	this->rayParamsPtr=new rayFieldParams;
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
	this->rayParamsPtr->totalLaunch_height=1;
	this->rayParamsPtr->totalLaunch_width=parseResults_Src.height*parseResults_Src.width*parseResults_Src.nrRayDirections.x*parseResults_Src.nrRayDirections.y;
	this->rayParamsPtr->flux=parseResults_Src.power;
	this->rayParamsPtr->lambda=parseResults_Src.lambda*1e-3;
	this->rayParamsPtr->posDistrType=parseResults_Src.rayPosDistr;
	this->rayParamsPtr->dirDistrType=parseResults_Src.rayDirDistr;
	this->rayParamsPtr->alphaMax=parseResults_Src.alphaMax;
	this->rayParamsPtr->alphaMin=parseResults_Src.alphaMin;
	this->rayParamsPtr->nrRayDirections=parseResults_Src.nrRayDirections;//parseResults_Src.nrRayDirections;
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
			std::cout <<"warning in PathTracingRayField.processParseResults(): unknown material. Rafracting material with n=1 assumed." << std::endl;
			break;
	}
	return FIELD_NO_ERR;
};