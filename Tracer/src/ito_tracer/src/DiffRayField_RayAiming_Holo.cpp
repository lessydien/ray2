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

/**\file DiffRayField_RayAiming_Holo.cpp
* \brief Rayfield for differential raytracing
* 
*           
* \author Mauch
*/

#include <fstream>
#include <iostream>
#include <iomanip>

#include <omp.h>
#include "DiffRayField_RayAiming_Holo.h"
#include "DetectorLib.h"
#include "myUtil.h"
#include <complex>
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"

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
fieldError DiffRayField_RayAiming_Holo::setLambda(double lambda)
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
unsigned long long DiffRayField_RayAiming_Holo::getRayListLength(void)
{
	return this->rayListLength;
};

/**
 * \detail setRay 
 *
 * \param[in] diffRayStruct ray, unsigned long long index
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError DiffRayField_RayAiming_Holo::setRay(diffRayStruct ray, unsigned long long index)
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
 * \return diffRayStruct*
 * \sa 
 * \remarks 
 * \author Mauch
 */
diffRayStruct* DiffRayField_RayAiming_Holo::getRay(unsigned long long index)
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
 * \return diffRayStruct*
 * \sa 
 * \remarks 
 * \author Mauch
 */
diffRayStruct* DiffRayField_RayAiming_Holo::getRayList(void)
{
	return &rayList[0];	
};

/**
 * \detail setParamsPtr 
 *
 * \param[in] DiffRayField_RayAiming_HoloParams *paramsPtr
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
void DiffRayField_RayAiming_Holo::setParamsPtr(DiffRayField_RayAiming_HoloParams *paramsPtr)
{
	this->rayParamsPtr=paramsPtr;
	this->update=true;
};

/**
 * \detail getParamsPtr 
 *
 * \param[in] void
 * 
 * \return DiffRayField_RayAiming_HoloParams*
 * \sa 
 * \remarks 
 * \author Mauch
 */
DiffRayField_RayAiming_HoloParams* DiffRayField_RayAiming_Holo::getParamsPtr(void)
{
	return this->rayParamsPtr;
};


/* functions for GPU usage */

//void DiffRayField_RayAiming_Holo::setPathToPtx(char* path)
//{
//	memcpy(this->path_to_ptx_rayGeneration, path, sizeof(this->path_to_ptx_rayGeneration));
//};
//
//const char* DiffRayField_RayAiming_Holo::getPathToPtx(void)
//{
//	return this->path_to_ptx_rayGeneration;
//};

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
fieldError DiffRayField_RayAiming_Holo::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	if (FIELD_NO_ERR != RayField::createOptixInstance(context, output_buffer_obj, seed_buffer_obj))
	{
		std::cout <<"error in DiffRayField_RayAiming_Holo.createOptixInstance(): RayField.creatOptiXInstance() returned an error." << std::endl;
		return FIELD_ERR;
	}
	RTvariable output_buffer;
	RTvariable holoAngle_buffer;
	RTvariable holoAbs_buffer;

	RTvariable params;

	/* declare result buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "output_buffer", &output_buffer ), context ))
		return FIELD_ERR;
    /* Render result buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_OUTPUT, &output_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( output_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( output_buffer_obj, sizeof(diffRayStruct) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( output_buffer_obj, GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_WIDTH_MAX ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( output_buffer, output_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare holo angle buffer */
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "holoAngle_buffer", &holoAngle_buffer ), context ))
		return FIELD_ERR;
    /* Render holo angle buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &holoAngle_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( holoAngle_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( holoAngle_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize2D( holoAngle_buffer_obj, WIDTH_HOLO_BUFFER, HEIGHT_HOLO_BUFFER ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( holoAngle_buffer, holoAngle_buffer_obj ), context ))
		return FIELD_ERR;

	/* fill holo angle buffer */
	char holoAngle[128];
	sprintf( holoAngle, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "holoTobi512.txt" );
	std::ifstream inFile(holoAngle);

	// make sure the filestream is good
	if (!inFile)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.createOptiXInstance(): failed to open holo angle file " << std::endl;
		return FIELD_ERR;
	}

	void *data;
	// read the seed buffer from the GPU
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(holoAngle_buffer_obj, &data), context ))
		return FIELD_ERR;
	holoAngle_buffer_CPU = reinterpret_cast<double*>( data );
	//holoAngle_buffer_CPU=(double*)malloc(100*100*sizeof(double));

	char test;
	RTsize buffer_width;
	RTsize buffer_height;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferGetSize2D(holoAngle_buffer_obj, &buffer_width, &buffer_height), context ))
		return FIELD_ERR;
	for ( unsigned int i = 0; i < (unsigned int)buffer_height; ++i )
	{
		for (unsigned int j=0; j<(unsigned int)buffer_width; ++j)
		{
			if (inFile.eof())
			{
				std::cout << "error in DiffRayField_RayAiming_Holo.createOptiXInstance(): end of file of freeform file before all points were read " << std::endl;
				return FIELD_ERR;
			}
			inFile >> holoAngle_buffer_CPU[j+i*buffer_width];
			//inFile >> test;
			holoAngle_buffer_CPU[j+i*buffer_width]=holoAngle_buffer_CPU[j+i*buffer_width];
		}
	}
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( holoAngle_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare holo abs buffer */
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "holoAbs_buffer", &holoAbs_buffer ), context ))
		return FIELD_ERR;
    /* Render holo angle buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &holoAbs_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( holoAbs_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( holoAbs_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize2D( holoAbs_buffer_obj, WIDTH_HOLO_BUFFER, HEIGHT_HOLO_BUFFER ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( holoAbs_buffer, holoAbs_buffer_obj ), context ))
		return FIELD_ERR;

	/* fill holo abs buffer */
	char holoAbs[128];
	sprintf( holoAbs, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "holoLine512_0_3Grad_800nm_Abs.txt" );
	std::ifstream inFileAbs(holoAbs);

	// make sure the filestream is good
	if (!inFileAbs)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.createOptiXInstance(): failed to open holo abs file " << std::endl;
		return FIELD_ERR;
	}

	// read the holoAbs buffer from the GPU
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(holoAbs_buffer_obj, &data), context ))
		return FIELD_ERR;
	holoAbs_buffer_CPU = reinterpret_cast<double*>( data );

	if (!RT_CHECK_ERROR_NOEXIT( rtBufferGetSize2D(holoAbs_buffer_obj, &buffer_width, &buffer_height) , context))
		return FIELD_ERR;
	for ( unsigned int i = 0; i < (unsigned int)buffer_height; ++i )
	{
		for (unsigned int j=0; j<(unsigned int)buffer_width; ++j)
		{
			if (inFileAbs.eof())
			{
				std::cout << "error in DiffRayField_RayAiming_Holo.createOptiXInstance(): end of file of holo abs file before all points were read " << std::endl;
				return FIELD_ERR;
			}
			inFileAbs >> holoAbs_buffer_CPU[j+i*buffer_width];
			//inFile >> test;
			holoAbs_buffer_CPU[j+i*buffer_width]=holoAbs_buffer_CPU[j+i*buffer_width];
		}
	}
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( holoAbs_buffer_obj ) , context))
		return FIELD_ERR;

	//RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "diff_epsilon", &diff_epsilon ) );
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "params", &params ), context ))
		return FIELD_ERR;

	this->rayParamsPtr->nImmersed=this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);

	if (FIELD_NO_ERR!=this->createCPUSimInstance())
	{
		std::cout <<"error in DiffRayField_RayAiming_Holo.createOptixInstance(): create CPUSimInstance() returned an error." << std::endl;
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
	//// differential rays are traced in a 1D launch, so we set the subset height to the height of the ray field. This is necessary as geometric ray fields are traced in 2D launches and SimAssistant.doSim() doesn't know wether it is simulating differential or geometric rayfields !!
	//this->rayParamsPtr->GPUSubset_height=1;//this->rayParamsPtr->height;

	//this->rayParamsPtr->launchOffsetX=0;//l_offsetX;
	//this->rayParamsPtr->launchOffsetY=0;//l_offsetY;

	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(DiffRayField_RayAiming_HoloParams), this->rayParamsPtr), context) )
		return FIELD_ERR;

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
fieldError DiffRayField_RayAiming_Holo::initCPUSubset()
{
	clock_t start, end;
	double msecs=0;
	// check wether we will be able to fit all the rays into our raylist. If not some eror happened earlier and we can not proceed...
	if ((this->rayParamsPtr->GPUSubset_width)<=this->rayListLength)
	{
		// calc the dimensions of the subset
//		long2 l_GPUSubsetDim=calcSubsetDim();
		
		// see if there are any rays to create	
		//if (this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width >= 1)
		if (this->rayParamsPtr->GPUSubset_width*this->rayParamsPtr->GPUSubset_height >= 1)
		{
			// width of ray field in physical dimension
			double physWidth=this->rayParamsPtr->rayPosEnd.x-this->rayParamsPtr->rayPosStart.x;
			// height of ray field in physical dimension
			double physHeight=this->rayParamsPtr->rayPosEnd.y-this->rayParamsPtr->rayPosStart.y;
			// calc centre of ray field 
			double2 rayFieldCentre=make_double2(this->rayParamsPtr->rayPosStart.x+physWidth/2,this->rayParamsPtr->rayPosStart.y+physHeight/2);

			// calc centre angle of opening cone
			//rayParamsPtr->alphaMax=make_double2(PI/2,PI/2);
			//rayParamsPtr->alphaMin=make_double2(-PI/2,-PI/2);
			double3 rayAngleCentre=make_double3((this->rayParamsPtr->alphaMax.x+this->rayParamsPtr->alphaMin.x)/2,(this->rayParamsPtr->alphaMax.y+this->rayParamsPtr->alphaMin.y)/2,0);
			double angleWidth=(this->rayParamsPtr->alphaMax.x-this->rayParamsPtr->alphaMin.x);
			double angleHeight=(this->rayParamsPtr->alphaMax.y-this->rayParamsPtr->alphaMin.y);

			// declar variables for randomly distributing ray directions via an importance area
			//double2 impAreaHalfWidth;
			//double3 dirImpAreaCentre, tmpPos, impAreaRoot;
			//double impAreaX, impAreaY, r, theta;
			//double3 impAreaAxisX, impAreaAxisY;

			//double3 alpha=make_double3(0,0,0); // rotation angle of raydirection around x and y

			// start timing
			start=clock();

			std::cout << "initalizing random seed" << std::endl;

			int seed = (int)time(0);            // random seed
			RandomInit(seed, x);

			// create random seeds for all the rays

			for(signed long long jx=0;jx<this->rayParamsPtr->GPUSubset_width;jx++)
			{
				this->rayList[jx].currentSeed=(uint)BRandom(x);
			}
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize random seeds of " << this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

			// start timing
			start=clock();

			// create all the rays
			omp_set_num_threads(numCPU);

#pragma omp parallel default(shared)
{
			#pragma omp for schedule(dynamic, 50)//schedule(static)//
			// create all the rays
			for(signed long long jx=0;jx<this->rayParamsPtr->GPUSubset_width;jx++)
			{
				uint32_t x_l[5];
				RandomInit(this->rayList[jx].currentSeed, x_l); // seed random generator

//				long long index=0; // loop counter for random rejection method
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

				// calc data of detector
				double deltaW_det=0;
				double deltaH_det=0;
				// calc increment along x- and y-direction
				if (this->rayParamsPtr->oDetParams.detPixel.x>0)
					deltaW_det= (2*this->rayParamsPtr->oDetParams.apertureHalfWidth.x)/double(this->rayParamsPtr->oDetParams.detPixel.x);
				if (this->rayParamsPtr->oDetParams.detPixel.y>0)
					deltaH_det= (2*this->rayParamsPtr->oDetParams.apertureHalfWidth.y)/double(this->rayParamsPtr->oDetParams.detPixel.y);


				double3 detAxisX=make_double3(1,0,0);
				double3 detAxisY=make_double3(0,1,0);
				rotateRay(&detAxisX,this->rayParamsPtr->oDetParams.tilt);
				rotateRay(&detAxisY,this->rayParamsPtr->oDetParams.tilt);


				// declare variables for randomly distributing ray directions via an importance area
				double2 impAreaHalfWidth;
				double3 dirImpAreaCentre, tmpPos, impAreaRoot, rayAngleCentre,impAreaAxisX,impAreaAxisY;
				double impAreaX, impAreaY, theta;

				diffRayStruct rayData;
				rayData.flux=this->rayParamsPtr->flux;
				rayData.depth=0;	
				rayData.position.z=this->rayParamsPtr->rayPosStart.z;
				rayData.running=true;
				rayData.currentGeometryID=0;
				rayData.lambda=this->rayParamsPtr->lambda;
				rayData.nImmersed=1;//this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);
				double epsilon=this->rayParamsPtr->epsilon;//DIFF_EPSILON; // small distance. The ray is moved out of the caustic by this distance
				rayData.mainDirX=make_double3(1,0,0);
				rayData.mainDirY=make_double3(0,1,0);
				rayData.flux=this->rayParamsPtr->flux;

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
						std::cout << "error in DiffRayField_RayAiming_Holo.initCPUSubset: unknown distribution of rayposition" << std::endl;
						// report error
						break;
				}
				// interpolate phase from data table
				double l_phase;
				double buffer_deltaX=physWidth/WIDTH_HOLO_BUFFER;
				double buffer_deltaY=physHeight/HEIGHT_HOLO_BUFFER;
				this->oInterpPtr->nearestNeighbour(-WIDTH_HOLO_BUFFER/2*buffer_deltaX+buffer_deltaX/2,-HEIGHT_HOLO_BUFFER/2*buffer_deltaY+buffer_deltaY/2,buffer_deltaX,buffer_deltaY,this->holoAngle_buffer_CPU,WIDTH_HOLO_BUFFER,HEIGHT_HOLO_BUFFER,rayData.position.x,rayData.position.y,&l_phase);
				// transfer phase value that is contained in the hologram file to opl
				rayData.opl=rayData.opl+l_phase*this->rayParamsPtr->lambda/(2*M_PI);

				// transform rayposition into global coordinate system
				rayData.position=this->rayParamsPtr->Mrot*rayData.position+this->rayParamsPtr->translation;

				//// calc target of current ray
				//double detX=-this->rayParamsPtr->oDetParams.apertureHalfWidth.x+deltaW_det/2+iDirX*deltaW_det; 
				//double detY=-this->rayParamsPtr->importanceAreaHalfWidth.y+deltaH_det/2+iDirY*deltaH_det; 

				//// this finally is the point on the detector that we want to hit
				//double3 hitPos=this->rayParamsPtr->oDetParams.root+detX*detAxisX+detY*detAxisY;

				double2 rayAngleHalfWidth, phi;

				switch (this->rayParamsPtr->dirDistrType)
				{
					case RAYDIR_UNIFORM:
//						rayData.direction=this->rayParamsPtr->rayDirection;
//						// transform direction into global coordinate system
//						rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
						rayData.direction=normalize(this->rayParamsPtr->oDetParams.root-rayData.position);
						break;
					case RAYDIR_RAND_RECT:
							rayAngleCentre=make_double3((this->rayParamsPtr->alphaMax.x+this->rayParamsPtr->alphaMin.x)/2,(this->rayParamsPtr->alphaMax.y+this->rayParamsPtr->alphaMin.y)/2,0);
							rayAngleHalfWidth=make_double2((this->rayParamsPtr->alphaMax.x-this->rayParamsPtr->alphaMin.x)/2,(this->rayParamsPtr->alphaMax.y-this->rayParamsPtr->alphaMin.y)/2);
							// create random angles inside the given range
							phi=make_double2(2*(Random(x_l)-0.5)*rayAngleHalfWidth.x+rayAngleCentre.x,2*(Random(x_l)-0.5)*rayAngleHalfWidth.y+rayAngleCentre.y);
							// create unit vector with the given angles
							rayData.direction=createObliqueVec(phi);//normalize(make_double3(tan(phi.y),tan(phi.x),1));
							// transform raydirection into global coordinate system
							rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;


						//// create points inside importance area to randomly distribute ray direction
						//rayAngleCentre=make_double3((this->rayParamsPtr->alphaMax.x+this->rayParamsPtr->alphaMin.x)/2,(this->rayParamsPtr->alphaMax.y+this->rayParamsPtr->alphaMin.y)/2,0);
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

						// transform raydirection into global coordinate system
						rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
						// save seed for next randomization
						rayData.currentSeed=x[4];
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
						//		std::cout << "error in DiffRayField.initCPUSubset: importance area for defining ray directions of source is only allowed with objects that have rectangular or elliptical apertures" << std::endl;
						//		// report error
						//		//return FIELD_ERR; // return is not allowed inside opneMP block!!!
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
						impAreaX=-this->rayParamsPtr->importanceAreaHalfWidth.x+deltaW/2+iDirX*deltaW; 
						impAreaY=-this->rayParamsPtr->importanceAreaHalfWidth.y+deltaH/2+iDirY*deltaH; 
						impAreaAxisX=make_double3(1,0,0);
						impAreaAxisY=make_double3(0,1,0);
						rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
						rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);

						tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
						rayData.direction=normalize(tmpPos-rayData.position);

						break;

					case RAYDIR_GRID_RECT_FARFIELD:
						// calc increment along x- and y-direction
						if (this->rayParamsPtr->nrRayDirections.x>0)
							deltaW= (this->rayParamsPtr->alphaMax.y-this->rayParamsPtr->alphaMin.y)/double(this->rayParamsPtr->nrRayDirections.y);
						if (this->rayParamsPtr->nrRayDirections.y>0)
							deltaH= (this->rayParamsPtr->alphaMax.x-this->rayParamsPtr->alphaMin.x)/double(this->rayParamsPtr->nrRayDirections.x);
						phi.y=this->rayParamsPtr->alphaMax.y-deltaW/2-iDirY*deltaW; 
						phi.x=this->rayParamsPtr->alphaMax.x-deltaH/2-iDirX*deltaH; 

						rayData.direction=createObliqueVec(phi);//normalize(make_double3(tan(phi.y),tan(phi.x),1));

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
						std::cout << "error in Diff.initCPUSubset: unknown raydirection distribution" << std::endl;
						// report error
						break;
				}

				// move ray out of caustic
				rayData.position=rayData.position+epsilon*rayData.direction;
				rayData.currentSeed=(uint)BRandom(x);
				rayData.opl=epsilon;
				rayData.wavefrontRad=make_double2(-epsilon,-epsilon); // init wavefront radius according to small distance
				// adjust flux
				rayData.flux=1/(epsilon*epsilon)*rayData.flux;
				//further adjust flux
				//rayData.flux=rayData.flux*abs(dot(rayData.direction,make_double3(0,0,1)));


				// create main directions
				// calc angles with respect to global x- and y-axis
				phi=calcAnglesFromVector(rayData.direction,this->rayParamsPtr->tilt);
				rayData.mainDirX=make_double3(1,0,0);
				rotateRay(&(rayData.mainDirX),this->rayParamsPtr->tilt+make_double3(phi.x,phi.y,0));
				rayData.mainDirY=make_double3(0,1,0);
				rotateRay(&(rayData.mainDirY),this->rayParamsPtr->tilt+make_double3(phi.x,phi.y,0));


				//rayData.mainDirX.y=0;
				//rayData.mainDirY.x=0;
				//if (rayData.direction.z!=0)
				//{
				//	rayData.mainDirX.x=1/sqrt(1-rayData.direction.x/rayData.direction.z);
				//	rayData.mainDirX.z=-rayData.mainDirX.x*rayData.direction.x/rayData.direction.z;
				//	rayData.mainDirY.y=1/sqrt(1-rayData.direction.y/rayData.direction.z);
				//	rayData.mainDirY.z=-rayData.mainDirY.y*rayData.direction.x/rayData.direction.z;
				//}
				//else
				//{
				//	if (rayData.direction.x != 0)
				//	{
				//		rayData.mainDirX.z=1/sqrt(1-rayData.direction.z/rayData.direction.x);
				//		rayData.mainDirX.x=-rayData.mainDirX.z*rayData.direction.z/rayData.direction.x;
				//	}
				//	else
				//		rayData.mainDirX=make_double3(1,0,0);
				//	if (rayData.direction.y != 0)
				//	{
				//		rayData.mainDirY.z=1/sqrt(1-rayData.direction.z/rayData.direction.y);
				//		rayData.mainDirY.y=-rayData.mainDirY.z*rayData.direction.z/rayData.direction.y;
				//	}
				//	else
				//		rayData.mainDirY=make_double3(0,1,0);
				//}
				rayData.currentSeed=(uint)BRandom(x);
				this->setRay(rayData,(unsigned long long)(jx));
				//increment directions counter
			}
} // end omp
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize " << this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;
		}

		else if(this->rayParamsPtr->width*this->rayParamsPtr->height<1)
		{
			//not Possible. Report error or set n=-n
			std::cout << "error in DiffRayField_RayAiming_Holo.initCPUInstance: negative raynumber" << std::endl;
		}
		this->update=false;
	}	// end if GPUsubsetwidth*height<rayListLength
	else
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.initCPUInstance: rayList is smaller than simulation subset" << std::endl;
		return FIELD_ERR;
	}
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
long2 DiffRayField_RayAiming_Holo::calcSubsetDim()
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
fieldError DiffRayField_RayAiming_Holo::createCPUSimInstance()
{
	if (this->rayList != NULL)
	{
		delete this->rayList;
		this->rayListLength=0;
		rayList=NULL;
	}
	rayList=(diffRayStruct*) malloc(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX*sizeof(diffRayStruct));
	if (!rayList)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.createLayoutInstance(): memory for rayList could not be allocated. try to reduce ray tiling size" << std::endl;
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

	/* fill holo buffer */
	char holoAngle[128];
	sprintf( holoAngle, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "hologramLine512_12Grad.txt" );
	std::ifstream inFile(holoAngle);

	// make sure the filestream is good
	if (!inFile)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.createCPUSimInstance(): failed to open freeform file " << std::endl;
		//return FIELD_ERR;
	}

	// create array for freeform data
	holoAngle_buffer_CPU=(double*)malloc(WIDTH_HOLO_BUFFER*HEIGHT_HOLO_BUFFER*sizeof(double));
	char test;
	for ( unsigned int i = 0; i < HEIGHT_HOLO_BUFFER; ++i )
	{
		for (unsigned int j=0; j<WIDTH_HOLO_BUFFER; ++j)
		{
			if (inFile.eof())
			{
				std::cout << "error in DiffRayField_RayAiming_Holo.createCPUSimInstance(): end of file of freeform file before all points were read " << std::endl;
				//return FIELD_ERR;
			}
			inFile >> holoAngle_buffer_CPU[j+i*WIDTH_HOLO_BUFFER];
//			inFile >> test;
		}
	}

	/* fill holo buffer */
	char holoAbs[128];
	sprintf( holoAbs, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "hologramLine512_12_6_0Grad_Abs.txt" );
	std::ifstream inFileAbs(holoAbs);

	// make sure the filestream is good
	if (!inFileAbs)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.createCPUSimInstance(): failed to open holo abs file " << std::endl;
		//return FIELD_ERR;
	}

	// create array for freeform data
	holoAbs_buffer_CPU=(double*)malloc(WIDTH_HOLO_BUFFER*HEIGHT_HOLO_BUFFER*sizeof(double));
	for ( unsigned int i = 0; i < HEIGHT_HOLO_BUFFER; ++i )
	{
		for (unsigned int j=0; j<WIDTH_HOLO_BUFFER; ++j)
		{
			if (inFileAbs.eof())
			{
				std::cout << "error in DiffRayField_RayAiming_Holo.createCPUSimInstance(): end of file of holo abs file before all points were read " << std::endl;
				//return FIELD_ERR;
			}
			inFile >> holoAbs_buffer_CPU[j+i*WIDTH_HOLO_BUFFER];
//			inFile >> test;
		}
	}

	return FIELD_NO_ERR;
};

/**
 * \detail createCPUSimInstance 
 *
 *
 * \param[in] void
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError DiffRayField_RayAiming_Holo::createLayoutInstance()
{
	unsigned int l_launch_width, l_launch_height, l_offsetX, l_offsetY;
	// calc the dimensions of the simulation subset
	if ( this->rayParamsPtr->widthLayout*this->rayParamsPtr->heightLayout*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y < GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX ) 
	{
		this->rayParamsPtr->GPUSubset_width=this->rayParamsPtr->widthLayout*this->rayParamsPtr->heightLayout*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;
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

	this->rayParamsPtr->totalLaunch_height=1;
	this->rayParamsPtr->totalLaunch_width=this->rayParamsPtr->widthLayout*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;

	return FIELD_NO_ERR;

	//initCPUSubset();
};

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
fieldError DiffRayField_RayAiming_Holo::write2TextFile(char* filename, detParams &oDetParams)
{
	char t_filename[512];
	//sprintf(t_filename, "%s%sDiffRayField_RayAiming_Holo_%i.txt", filename, PATH_SEPARATOR, oDetParams.subSetNr);
	sprintf(t_filename, "%s%sDiffRayField_RayAiming_Holo.txt", filename, PATH_SEPARATOR);

	FILE* hFileOut;
	hFileOut = fopen( t_filename, "a" ) ;
	if (!hFileOut)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.write2TextFile(): could not open output file: " << filename << std::endl;
		return FIELD_ERR;
	}
	// calc the dimensions of the subset
//	long2 l_GPUSubsetDim=calcSubsetDim();

	if (1) // (reducedData==1)
	{
		//for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		{
			//for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
				if ((rayList[rayListIndex].currentGeometryID==oDetParams.geomID) || (oDetParams.geomID=-1))
				{
					// write the data in row major format, where width is the size of one row and height is the size of one coloumn
					// if the end of a row is reached append a line feed 
					fprintf(hFileOut, "%.20lf ;%.20lf ;%.20lf; %.20lf ;%.20lf ;%.20lf ;%.20lf ;%.20lf;%i ;%.20lf ;%.20lf \n", rayList[rayListIndex].position.x, rayList[rayListIndex].position.y, rayList[rayListIndex].position.z, rayList[rayListIndex].direction.x, rayList[rayListIndex].direction.y, rayList[rayListIndex].direction.z, rayList[rayListIndex].flux, rayList[rayListIndex].opl, rayList[rayListIndex].currentGeometryID, rayList[rayListIndex].wavefrontRad.x, rayList[rayListIndex].wavefrontRad.y);
				}
			}
		}
	}
	else
	{
		//for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		{
			//for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
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
 * \detail traceScene 
 *
 * \param[in] Group &oGroup
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError DiffRayField_RayAiming_Holo::traceScene(Group &oGroup, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	clock_t start, end;
	double msecs=0;

	// start timing
	start=clock();

//	long2 l_GPUSubsetDim=calcSubsetDim();

	if (RunOnCPU)
	{

		omp_set_num_threads(numCPU);

		std::cout << "tracing on " << numCPU << " cores of CPU." << std::endl;

		if (FIELD_NO_ERR!= initCPUSubset())
		{
			std::cout << "error in DiffRayField_RayAiming_Holo.traceScene: initCPUSubset returned an error" << std::endl;
			return FIELD_ERR;
		}

		std::cout << "starting the actual trace..." << std::endl;

#pragma omp parallel default(shared)
{

		// calc data of detector
		double deltaW=0;
		double deltaH=0;
		// calc increment along x- and y-direction
		if (this->rayParamsPtr->oDetParams.detPixel.x>0)
			deltaW= (2*this->rayParamsPtr->oDetParams.apertureHalfWidth.x)/double(this->rayParamsPtr->oDetParams.detPixel.x);
		if (this->rayParamsPtr->oDetParams.detPixel.y>0)
			deltaH= (2*this->rayParamsPtr->oDetParams.apertureHalfWidth.y)/double(this->rayParamsPtr->oDetParams.detPixel.y);


		double3 detAxisX=make_double3(1,0,0);
		double3 detAxisY=make_double3(0,1,0);
		rotateRay(&detAxisX,this->rayParamsPtr->oDetParams.tilt);
		rotateRay(&detAxisY,this->rayParamsPtr->oDetParams.tilt);


		#pragma omp for schedule(dynamic, 500)

		for (signed long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				// see where this ray is supposed to hit the detector
				unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;

				// calc position indices from 1D index
				unsigned long long iPosX=floorf(iGes/(this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y));
				unsigned long long iPosY=floorf(iPosX/this->rayParamsPtr->width);
				iPosX=iPosX % this->rayParamsPtr->width;
					
				// calc direction indices from 1D index
				unsigned long long iDirX=(iGes-iPosX*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y-iPosY*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y*this->rayParamsPtr->width) % this->rayParamsPtr->nrRayDirections.x;
				unsigned long long iDirY=floorf((iGes-iPosX*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y-iPosY*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y*this->rayParamsPtr->width)/this->rayParamsPtr->nrRayDirections.x);

				double detX=this->rayParamsPtr->oDetParams.apertureHalfWidth.x-deltaW/2-iDirY*deltaW; 
				double detY=this->rayParamsPtr->oDetParams.apertureHalfWidth.y-deltaH/2-iDirX*deltaH; 

				// this finally is the point on the detector that we want to hit
				double3 targetHitPos=this->rayParamsPtr->oDetParams.root+detX*detAxisX+detY*detAxisY;

				// define variables for the ray aiming loop
				bool firstRun=true;

				//**********************************************************************
				// init the values for debugging
//				this->rayList[jx].position=make_double3(0.0,0.0,-22.030743);
//				this->rayList[jx].direction=normalize(make_double3(0.0,0,0.1));//make_double3(0.0055833990273328350,0,0.99998441270616889);
//				this->rayList[jx].opl=0;
				//targetHitPos=make_double3(3.99511718750,0,29.774);
				//deltaW=0.0001;
				//deltaH=0.0001;
				//**********************************************************************

				// init old ray
				diffRayStruct startingRay=this->rayList[jx];

				double3 startingDirLastIteration=this->rayList[jx].direction;

				// init derivative of direction of starting ray with respect to change in position of hit point
				// use a small value to be safe in case the sign is wron in the beginning...
				double3 dDir_dPos=make_double3(0,0,0); 
				double3 diffPosOld=make_double3(DOUBLE_MAX,DOUBLE_MAX,DOUBLE_MAX);
				double3 hitPosOld=make_double3(0,0,0);

				unsigned long long index=0;
				double3 recentHitPos;
				// do ray aiming
				do
				{
					index++;
//					std::cout << "index: " << index << std::endl;
					if (index>151)
					{
						std::cout << "error in DiffRayField_RayAiming_Holo.traceScene(): ray aiming loop canceled after " << index << " iterations for ray " << jx << std::endl;
						std::cout << "old hitpos: " << recentHitPos.x << " " << recentHitPos.y << " " << recentHitPos.z << std::endl;
						std::cout << "targetpos: " << targetHitPos.x << " " << targetHitPos.y << " " << targetHitPos.z << std::endl;
//						return FIELD_ERR;
						break;
					}

//					std::cout << "startpos: " << this->rayList[jx].position.x << " " << this->rayList[jx].position.y << " " <<this->rayList[jx].position.z << std::endl;
//					std::cout << "startdir: " << this->rayList[jx].direction.x << " " << this->rayList[jx].direction.y << " " << this->rayList[jx].direction.z << std::endl;

					// save raydata for next iteration
					startingRay=this->rayList[jx];
					
					for(;;) // iterative tracing
					{
						if(!this->rayList[jx].running) 
							break;
						oGroup.trace(rayList[jx]);
//						std::cout << std::endl;
					}
//					std::cout << "hitpos: " << this->rayList[jx].position.x << " " << this->rayList[jx].position.y << " " << this->rayList[jx].position.z << std::endl;
//					std::cout << "targetpos: " << targetHitPos.x << " " << targetHitPos.y << " " << targetHitPos.z << std::endl;
					// calc difference of current hitPos to target hitPos
					double3 diffPos=targetHitPos-this->rayList[jx].position;
					recentHitPos=this->rayList[jx].position;

					if ( (abs(dot(diffPos,detAxisX)) > deltaW/2) || (abs(dot(diffPos,detAxisY)) > deltaH/2 ) )
					{

						//// update ray for next trace
						//// init ray with data from last trace
						//this->rayList[jx]=startingRay;
						//// move ray back into caustic
						//this->rayList[jx].position=startingRay.position-this->rayParamsPtr->epsilon*startingRay.direction;
						//this->rayList[jx].direction=normalize(this->rayList[jx].direction+make_double3(0.01,0,0));

						// update ray for next trace
						// init ray with data from last trace
						this->rayList[jx]=startingRay;
						// move ray back into caustic
						this->rayList[jx].position=this->rayList[jx].position-this->rayList[jx].direction*this->rayParamsPtr->epsilon;
						this->rayList[jx].opl=this->rayList[jx].opl-this->rayParamsPtr->epsilon;
						this->rayList[jx].wavefrontRad=make_double2(0,0); // init wavefront radius according to small distance
						// adjust flux
						this->rayList[jx].flux=(this->rayParamsPtr->epsilon*this->rayParamsPtr->epsilon)*this->rayList[jx].flux;

						// update ray direction
						if (firstRun)
						{
							firstRun=false;
							this->rayList[jx].direction=this->rayList[jx].direction+make_double3(0.01,0.01,0);
						}
						else
						{
							// if the new diffPos is worse than the last one in one of the dimensions, we set the direction in that dimenision to the half between the old direction and this direction
							if ( 0 )//abs(diffPos.x)>abs(diffPosOld.x) || abs(diffPos.y)>abs(diffPosOld.y) )
								this->rayList[jx].direction=0.5*(startingRay.direction+startingDirLastIteration);
							else
							{
								// save diffPos
								diffPosOld=diffPos;

								// calc change in hit position
								double3 deltaPos=recentHitPos-hitPosOld;

								// save current hit pos
								hitPosOld=recentHitPos;

								// derivative ray direction with respect to hit position on detector
								// therefore calc the difference of the starting direction of this iteration (startingRay.direction) and the starting direction of the last iteration (oldDir)
								dDir_dPos=(startingRay.direction-startingDirLastIteration)/deltaPos;
								// save starting direction of this iteration
								startingDirLastIteration=startingRay.direction;

								// check for division by zero
								if (abs(deltaPos.x)<0.0000001)
									dDir_dPos.x=0;
								if (abs(deltaPos.y)<0.0000001)
									dDir_dPos.y=0;
								if (abs(deltaPos.z)<0.0000001)
									dDir_dPos.z=0;

								this->rayList[jx].direction=normalize(this->rayList[jx].direction+diffPos*dDir_dPos);
							}

						}
						// move ray out of caustic
						this->rayList[jx].position=this->rayList[jx].position+this->rayParamsPtr->epsilon*this->rayList[jx].direction;

						this->rayList[jx].opl=this->rayList[jx].opl+this->rayParamsPtr->epsilon;
						this->rayList[jx].wavefrontRad=make_double2(this->rayParamsPtr->epsilon,this->rayParamsPtr->epsilon); // init wavefront radius according to small distance
						// adjust flux
						this->rayList[jx].flux=1/(this->rayParamsPtr->epsilon*this->rayParamsPtr->epsilon)*this->rayList[jx].flux;

						// create main directions according to new direction
						// calc angles with respect to global x- and y-axis
						double2 phi=calcAnglesFromVector(this->rayList[jx].direction,this->rayParamsPtr->tilt);
						this->rayList[jx].mainDirX=createObliqueVec(make_double2(phi.x,phi.y+M_PI/2));
						this->rayList[jx].mainDirY=createObliqueVec(make_double2(phi.x+M_PI/2,phi.y));
					}
					else
					{
						diffPosOld=diffPos;
					}

				}
				// now check wether we hit it, i.e. wether we are inside the pixel we were aiming at.
				while ( (abs(dot(diffPosOld,detAxisX)) > deltaW/2) || (abs(dot(diffPosOld,detAxisY)) > deltaH/2 ) ); 
			}
}
	}
	else
	{
		//RTsize				buffer_width, buffer_height; // get size of output buffer
		void				*data; // pointer to cast output buffer into
 		//rayStruct			*bufferData;
		std::cout << "tracing on GPU." << std::endl;

		initGPUSubset(context, seed_buffer_obj);

		// start current launch. We make a one dimensional launch to process all the rays starting from one point into several directions simultaneously
		if (!RT_CHECK_ERROR_NOEXIT( rtContextLaunch1D( (context), 0, this->rayParamsPtr->GPUSubset_width), context))//this->rayParamsPtr->launchOffsetX, this->rayParamsPtr->launchOffsetY ) );
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
		if (FIELD_NO_ERR != copyRayList((diffRayStruct*)data,this->rayParamsPtr->GPUSubset_width) )
		{
			std::cout << "error in GeometricRayField.traceScene(): copyRayList() returned an error" << std::endl;
			return FIELD_NO_ERR;
		}
		
		if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( output_buffer_obj ) , context))
			return FIELD_ERR;
	}

	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << " " << msecs <<" ms to trace " << this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

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
fieldError DiffRayField_RayAiming_Holo::traceStep(Group &oGroup, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	clock_t start, end;
	double msecs=0;

	// start timing
	start=clock();

		std::cout << "tracing on " << numCPU << " cores of CPU." << std::endl;

#pragma omp parallel default(shared)
{
		#pragma omp for schedule(dynamic, 50)
			//int id;
			//id = omp_get_thread_num();

			//printf("Hello World from thread %d\n", id);

		for (signed long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
		{
			if (this->rayList[jx].running) 
				oGroup.trace(rayList[jx]);
		}
}
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << " " << msecs <<" ms to trace " << this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

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
fieldError DiffRayField_RayAiming_Holo::copyRayList(diffRayStruct *data, long long length)
{
	if (length > this->rayListLength)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.copyRayList(): subset dimensions exceed rayLIst dimension" << std::endl;
		return FIELD_ERR;
	}
	memcpy(this->rayList, data, this->rayParamsPtr->GPUSubset_width*sizeof(diffRayStruct));

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
fieldError DiffRayField_RayAiming_Holo::copyRayListSubset(diffRayStruct *data, long2 launchOffset, long2 subsetDim)
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
		memcpy(&(this->rayList[launchOffset.y*this->rayParamsPtr->width+launchOffset.x+jy*this->rayParamsPtr->width]), &data[jy*GPU_SUBSET_WIDTH_MAX], subsetDim.x*sizeof(diffRayStruct));
	}
	//memcpy(this->rayList, data, length*sizeof(rayStruct));
	return FIELD_NO_ERR;
};

/**
 * \detail writeData2File 
 *
 * \param[in] FILE *hFile, rayDataOutParams outParams
 * 
 * \return fieldError
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
fieldError DiffRayField_RayAiming_Holo::writeData2File(FILE *hFile, rayDataOutParams outParams)
{
//	writeGeomRayData2File(hFile, this->rayList, this->rayListLength, outParams);
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
fieldError DiffRayField_RayAiming_Holo::convert2RayData(Field** imagePtrPtr, detParams &oDetParams)
{
	DiffRayField_RayAiming_Holo* l_ptr;
	// if there is no image yet, create one
	if (*imagePtrPtr == NULL)
	{
		*imagePtrPtr=new DiffRayField_RayAiming_Holo(this->rayListLength);
		l_ptr=dynamic_cast<DiffRayField_RayAiming_Holo*>(*imagePtrPtr);
	}
	else
	{
		l_ptr=dynamic_cast<DiffRayField_RayAiming_Holo*>(*imagePtrPtr);
		if ( l_ptr->rayListLength < this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width )
		{
			std::cout << "error in GeometricRayField.convert2RayData(): dimensions of image does not fit dimensions of raylist subset" << std::endl;
			return FIELD_ERR;
		}
	}
	// copy the rayList and the respective parameters
	memcpy(l_ptr->getRayList(), this->rayList, (GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX)*sizeof(diffRayStruct));
	DiffRayField_RayAiming_HoloParams *l_diffRayParamsPtr=new DiffRayField_RayAiming_HoloParams();
	memcpy(l_diffRayParamsPtr,this->rayParamsPtr, sizeof(DiffRayField_RayAiming_HoloParams));
	l_ptr->setParamsPtr(l_diffRayParamsPtr);
	
	return FIELD_NO_ERR;
};

/**
 * \detail convert2Intensity 
 *
 * \param[in] IntensityField* imagePtr
 * 
 * \return fieldError
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
fieldError DiffRayField_RayAiming_Holo::convert2Intensity(Field* imagePtr, detParams &oDetParams)
{
	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;

	// start timing
	start=clock();

//	long2 l_GPUSubsetDim=calcSubsetDim();

	// cast the image to an IntensityField
	IntensityField* l_IntensityImagePtr=dynamic_cast<IntensityField*>(imagePtr);

	if (l_IntensityImagePtr == NULL)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.convert2Intensity(): imagePtr is not of type IntensityField" << std::endl;
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
	double4x4 MTransform=oDetParams.MTransform;
	long3 nrPixels=l_IntensityImagePtr->getParamsPtr()->nrPixels;
	// save the offset from the transformation matrix
	double3 offset=make_double3(MTransform.m14, MTransform.m24, MTransform.m34);
	// set offset in transformation matrix to zero for rotation of the scaled unit vectors
	MTransform.m14=0;
	MTransform.m24=0;
	MTransform.m34=0;
	// create unit vectors
	double3 t_ez = make_double3(0,0,1);
	double3 t_ey=make_double3(0,1,0);
	double3 t_ex=make_double3(1,0,0);
	// transform unit vectors into local coordinate system of IntensityField
	t_ez=MTransform*t_ez;
	t_ey=MTransform*t_ey;
	t_ex=MTransform*t_ex;

	// save normal
	double3 t_normal=t_ez;

	// the origin of the IntensityField is at the outer edge of the detector rather than at the origin
//	offset=offset-oDetParams.apertureHalfWidth.x*t_ex-oDetParams.apertureHalfWidth.y*t_ey;
	// the origin of the IntensityField is at the outer edge of the detector rather than at the origin
	offset=offset-oDetParams.apertureHalfWidth.x*t_ex;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ex;
	offset=offset-oDetParams.apertureHalfWidth.y*t_ey;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ey;
	offset=offset-0.005*t_ez;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ez;

	// scale unit vectors
	double3 t_ez_scaled = t_ez*scale.z; 
	double3 t_ey_scaled = t_ey*scale.y; 
	double3 t_ex_scaled = t_ex*scale.x; 

	//// we need the offset to be shifted half of a pixel...
	//offset=offset+0.5*t_ex+0.5*t_ey;

	short solutionIndex;

	double3x3 Matrix=make_double3x3(t_ex,t_ey,t_ez);
	if (optix::det(Matrix)==0)
	{
		std::cout << "error in GeometricRayField.convert2Intensity(): Matrix is unitary!!" << std::endl;
		return FIELD_ERR; //matrix singular
	}
	double3x3 MatrixInv=inv(Matrix);
	double3 posMinOffset;
	double3 indexFloat;
	long3 index;

	complex<double> i_compl=complex<double>(0,1); // define complex number "i"

	unsigned long long hitNr=0;

//	std::cout << "processing on " << numCPU << " cores of CPU." << std::endl;

//#pragma omp parallel default(shared)
//{
//	#pragma omp for schedule(dynamic, 50)

	for ( long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
	{
		posMinOffset=this->rayList[jx].position-offset;
		indexFloat=MatrixInv*posMinOffset;
		// subtract half a pixel (0.5*scale.x). This way the centre of our pixels do not lie on the edge of the aperture but rather half a pixel inside...
		// then round to nearest neighbour
		//index.x=floor((indexFloat.x-0.5*scale.x)/scale.x+0.5);
		index.x=floor((indexFloat.x)/scale.x);
		index.y=floor((indexFloat.y)/scale.y);
		index.z=floor((indexFloat.z)/scale.z);
			
		// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
		if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
		{
			// phase from OPL of differential ray
			double phi=std::fmod(2*PI/this->rayList[jx].lambda*this->rayList[jx].opl,2*M_PI);
			// we want to compute the field value at the centre of the pixel. Therefore we need to make som corrections in case the ray doesn't hit the Pixel at its centre
			// calc vector from differential ray to centre of pixel
			double3 PixelOffset=posMinOffset-(index.x*t_ex_scaled+index.y*t_ey_scaled+index.z*t_ez_scaled);
			// calc projection of this vector onto the ray direction
			double dz=dot(this->rayList[jx].direction,PixelOffset);
			// calc additional phase at centre of pixel from linear approximation to local wavefront
			phi=phi+dz*this->rayList[jx].lambda;
			// calc projection of this vector onto main directions of local wavefront
			double dx=dot(this->rayList[jx].mainDirX,PixelOffset);
			double dy=dot(this->rayList[jx].mainDirY,PixelOffset);
			//calc additional phase at centre of pixel from spherical approximation to local wavefront
			phi=phi+pow(dx,2)*1/this->rayList[jx].wavefrontRad.x+pow(dy,2)*1/this->rayList[jx].wavefrontRad.y;
			// we approximate the flux at the centre of the ray to be the same as at the ray position
			// nevertheless we need to take into account the angle between the ray direction and the normal of the pixel surface
			double t_fac=dot(this->rayList[jx].direction,t_normal);
			// add current field contribution to field at current pixel
			complex<double> l_U=polar(t_fac*this->rayList[jx].flux,phi);
			l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+l_U; // create a complex amplitude from the rays flux and opl and sum them coherently
			hitNr++;
		}
		else
		{
			std::cout << " ray " << jx << " not in target: " << rayList[jx].position.x << "; " << rayList[jx].position.y << "; " << rayList[jx].position.z << std::endl;
		}

	}
//}
	// if this is the last subset of the current launch, convert complex amplitude to intensity
	if ( this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->GPUSubset_width+this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y >= this->rayParamsPtr->width*this->rayParamsPtr->height*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y )
	{
		std::cout << " finally converting scalar field to intensity" << std::endl;
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
	std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_width << " rays in target" << std::endl;

	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<"ms to process " << this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

	return FIELD_NO_ERR;
};

/**
 * \detail convert2ScalarField 
 *
 * \param[in] ScalarLightField* imagePtr
 * 
 * \return fieldError
 * \sa 
 * \remarks not implemented yet
 * \author Mauch
 */
fieldError DiffRayField_RayAiming_Holo::convert2ScalarField(Field* imagePtr, detParams &oDetParams)
{
	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;

	// start timing
	start=clock();

//	long2 l_GPUSubsetDim=calcSubsetDim();

	// cast the image to an IntensityField
	ScalarLightField* l_ScalarImagePtr=dynamic_cast<ScalarLightField*>(imagePtr);

	if (l_ScalarImagePtr == NULL)
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.convert2ScalarField(): imagePtr is not of type ScalarField" << std::endl;
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
	double3 scale=l_ScalarImagePtr->getParamsPtr()->scale;
	double4x4 MTransform=oDetParams.MTransform;
	long3 nrPixels=l_ScalarImagePtr->getParamsPtr()->nrPixels;
	// save the offset from the transformation matrix
	double3 offset=make_double3(MTransform.m14, MTransform.m24, MTransform.m34);
	// set offset in transformation matrix to zero for rotation of the scaled unit vectors
	MTransform.m14=0;
	MTransform.m24=0;
	MTransform.m34=0;
	// create unit vectors
	double3 t_ez = make_double3(0,0,1);
	double3 t_ey=make_double3(0,1,0);
	double3 t_ex=make_double3(1,0,0);
	// transform unit vectors into local coordinate system of IntensityField
	t_ez=MTransform*t_ez;
	t_ey=MTransform*t_ey;
	t_ex=MTransform*t_ex;

	// save normal
	double3 t_normal=t_ez;

	//// the origin of the IntensityField is at the outer edge of the detector rather than at the origin
	//offset=offset-oDetParams.apertureHalfWidth.x*t_ex-oDetParams.apertureHalfWidth.y*t_ey;
	// the origin of the IntensityField is at the outer edge of the detector rather than at the origin
	offset=offset-oDetParams.apertureHalfWidth.x*t_ex;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ex;
	offset=offset-oDetParams.apertureHalfWidth.y*t_ey;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ey;
	offset=offset-0.005*t_ez;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ez;

	// we don't scale the vectors here anymore in order to ensure a good condition number of the matrix. Instead we scale the indices later
	// scale unit vectors
	double3 t_ez_scaled = t_ez*scale.z; 
	double3 t_ey_scaled = t_ey*scale.y; 
	double3 t_ex_scaled = t_ex*scale.x; 

	// we need the offset to be shifted half of a pixel...
//	offset=offset+0.5*t_ex+0.5*t_ey;

	short solutionIndex;

	double3x3 Matrix=make_double3x3(t_ex,t_ey,t_ez);
	if (optix::det(Matrix)==0)
	{
		std::cout << "error in GeometricRayField.convert2Intensity(): Matrix is unitary!!" << std::endl;
		return FIELD_ERR; //matrix singular
	}
	double3x3 MatrixInv=inv(Matrix);
	double3 posMinOffset;
	double3 indexFloat;
	long3 index;

	complex<double> i_compl=complex<double>(0,1); // define complex number "i"

	unsigned long long hitNr=0;

//	std::cout << "processing on " << numCPU << " cores of CPU." << std::endl;

//#pragma omp parallel default(shared)
//{
//	#pragma omp for schedule(dynamic, 50)

	for (long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
	{
		posMinOffset=this->rayList[jx].position-offset;
		indexFloat=MatrixInv*posMinOffset;
		// subtract half a pixel (0.5*scale.x). This way the centre of our pixels do not lie on the edge of the aperture but rather half a pixel inside...
		// then round to nearest neighbour
		//index.x=floor((indexFloat.x-0.5*scale.x)/scale.x+0.5);
		index.x=floor((indexFloat.x)/scale.x);
		index.y=floor((indexFloat.y)/scale.y);
		index.z=floor((indexFloat.z)/scale.z);
			
		// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
		if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
		{
			// phase from OPL of differential ray
			double phi=std::fmod((2*M_PI/this->rayList[jx].lambda*this->rayList[jx].opl),(2*M_PI));
			// we want to compute the field value at the centre of the pixel. Therefore we need to make som corrections in case the ray doesn't hit the Pixel at its centre
			// calc vector from differential ray to centre of pixel
			//double3 PixelOffset=posMinOffset-(index.x*t_ex_scaled+index.y*t_ey_scaled+index.z*t_ez_scaled);
			//// calc projection of this vector onto the ray direction
			//double dz=dot(this->rayList[jx].direction,PixelOffset);
			//// calc additional phase at centre of pixel from linear approximation to local wavefront
			//phi=phi+dz*this->rayList[jx].lambda;
			//// calc projection of this vector onto main directions of local wavefront
			//double dx=dot(this->rayList[jx].mainDirX,PixelOffset);
			//double dy=dot(this->rayList[jx].mainDirY,PixelOffset);
			////calc additional phase at centre of pixel from spherical approximation to local wavefront
			//phi=phi+pow(dx,2)*1/this->rayList[jx].wavefrontRad.x+pow(dy,2)*1/this->rayList[jx].wavefrontRad.y;
			// we approximate the flux at the centre of the ray to be the same as at the ray position
			// nevertheless we need to take into account the angle between the ray direction and the normal of the pixel surface
			double t_fac=1;//dot(this->rayList[jx].direction,t_normal);
			// add current field contribution to field at current pixel
			complex<double> l_U=polar(t_fac*this->rayList[jx].flux,phi);
			//complex<double> l_U=polar(t_fac*this->rayList[jx].opl,phi);
			//l_ScalarImagePtr->getFieldPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_U;
			l_ScalarImagePtr->getFieldPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_ScalarImagePtr->getFieldPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+l_U; // create a complex amplitude from the rays flux and opl and sum them coherently
			hitNr++;
		}
		else
		{
			std::cout << " ray " << jx << " not in target: " << rayList[jx].position.x << "; " << rayList[jx].position.y << "; " << rayList[jx].position.z << std::endl;
		}

	}
//}
	//l_ScalarImagePtr->getFieldPtr()[98]=polar(100,0);
	//l_ScalarImagePtr->getFieldPtr()[99]=polar(100,0);
	std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_width << " rays in target" << std::endl;
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<"ms to process " << this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;


	return FIELD_NO_ERR;
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
fieldError DiffRayField_RayAiming_Holo::convert2PhaseSpace(Field* imagePtr, detParams &oDetParams)
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
	double4x4 MTransform=oDetParams.MTransform;
	long3 nrPixels=l_PhaseSpacePtr->getParamsPtr()->nrPixels;
	// save the offset from the transformation matrix
	double3 offset=make_double3(MTransform.m14, MTransform.m24, MTransform.m34);
	// set offset in transformation matrix to zero for rotation of the scaled unit vectors
	MTransform.m14=0;
	MTransform.m24=0;
	MTransform.m34=0;
	// create unit vectors
	double3 t_ez = make_double3(0,0,1);
	double3 t_ey=make_double3(0,1,0);
	double3 t_ex=make_double3(1,0,0);
	// transform unit vectors into local coordinate system of IntensityField
	t_ez=MTransform*t_ez;
	t_ey=MTransform*t_ey;
	t_ex=MTransform*t_ex;

	// the origin of the IntensityField is at the outer edge of the detector rather than at the origin
	offset=offset-oDetParams.apertureHalfWidth.x*t_ex;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ex;
	offset=offset-oDetParams.apertureHalfWidth.y*t_ey;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ey;
	offset=offset-0.005*t_ez;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ez;

	// scale unit vectors
	double3 t_ez_scaled = t_ez*scale.z; 
	double3 t_ey_scaled = t_ey*scale.y; 
	double3 t_ex_scaled = t_ex*scale.x; 

	short solutionIndex;

	double3x3 Matrix=make_double3x3(t_ex_scaled,t_ey_scaled,t_ez_scaled);
	if (optix::det(Matrix)==0)
	{
		std::cout << "error in GeometricRayField.convert2PhaseSpaceField(): Matrix is unitary!!" << std::endl;
		return FIELD_ERR; //matrix singular
	}
	double3x3 MatrixInv=inv(Matrix);
//	double3 posMinOffset;
//	double3 indexFloat;
//	long3 index;
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
					rayStruct rayTest=this->rayList[rayListIndex];
					double3 posMinOffset=this->rayList[rayListIndex].position-offset;
					double3 indexFloat=MatrixInv*posMinOffset;
					long3 index;
					index.x=floor(indexFloat.x+0.5);
					index.y=floor(indexFloat.y+0.5);
					index.z=floor(indexFloat.z+0.5);

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
					double2 index_dir_float=make_double2((phi_x+M_PI/2-l_PhaseSpacePtr->getParamsPtr()->scale_dir.x/2)/l_PhaseSpacePtr->getParamsPtr()->scale_dir.x,(phi_y+M_PI/2-l_PhaseSpacePtr->getParamsPtr()->scale_dir.y/2)/l_PhaseSpacePtr->getParamsPtr()->scale_dir.y);
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

///**
// * \detail convert2VecField 
// *
// * \param[in] VectorLightField* imagePtr
// * 
// * \return fieldError
// * \sa 
// * \remarks not implemented yet
// * \author Mauch
// */
//fieldError DiffRayField_RayAiming_Holo::convert2VecField(Field* imagePtr, detParams &oDetParams)
//{
//	return FIELD_NO_ERR;
//};

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
fieldError DiffRayField_RayAiming_Holo::processParseResults(FieldParseParamStruct &parseResults_Src, parseGlassResultStruct* parseResults_GlassPtr, DetectorParseParamStruct &parseResults_Det)
{
	// no importance area and direction distribution GRID_RECT is not allowed
	if ( !parseResults_Src.importanceArea && ( (parseResults_Src.rayDirDistr==RAYDIR_GRID_RECT)||(parseResults_Src.rayDirDistr==RAYDIR_GRID_RAD) ) )
	{
		std::cout <<"error in DiffRayField_RayAiming_Holo.processParseResults(): direction distribution GRID_RECT and GRID_RAD are not allowed if no importance area is defined" << std::endl;
		return FIELD_ERR;
	}
	//if (parseResults_Src.rayDirDistr == RAYDIR_UNIFORM)
	//{
	//	std::cout <<"error in DiffRayField_RayAiming_Holo.processParseResults(): direction distribution UNIFORM is not allowed for differential ray fields" << std::endl;
	//	return FIELD_ERR;
	//}
	if ( (parseResults_Src.nrRayDirections.x*parseResults_Src.nrRayDirections.y > GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_WIDTH_MAX) && (parseResults_Src.rayDirDistr == RAYDIR_RAND_RECT) && ( (parseResults_Src.rayPosDistr == RAYPOS_RAND_RECT) || (parseResults_Src.rayPosDistr == RAYPOS_RAND_RAD) ) )
	{
		std::cout <<"warning in DiffRayField_RayAiming_Holo.processParseResults(): a number of ray directions that is bigger than the size of a GPU subset in combination with random position and direction distribution leads to a situation where some rays per point source point into the same direction when tracing on GPU." << std::endl;
	}

	//this->rayParamsPtr=new DiffRayField_RayAiming_HoloParams;
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
	this->rayParamsPtr->height=parseResults_Src.height;
	this->rayParamsPtr->totalLaunch_height=1;
	this->rayParamsPtr->totalLaunch_width=parseResults_Src.height*parseResults_Src.width*parseResults_Src.nrRayDirections.x*parseResults_Src.nrRayDirections.y;
	this->rayParamsPtr->flux=parseResults_Src.power;
	this->rayParamsPtr->lambda=parseResults_Src.lambda*1e-3;
	this->rayParamsPtr->posDistrType=parseResults_Src.rayPosDistr;
	this->rayParamsPtr->dirDistrType=parseResults_Src.rayDirDistr;
	this->rayParamsPtr->alphaMax=parseResults_Src.alphaMax;
	this->rayParamsPtr->alphaMin=parseResults_Src.alphaMin;
	this->rayParamsPtr->nrRayDirections=parseResults_Src.nrRayDirections;
	this->rayParamsPtr->importanceAreaHalfWidth=parseResults_Src.importanceAreaHalfWidth;
	this->rayParamsPtr->importanceAreaRoot=parseResults_Src.importanceAreaRoot;
	this->rayParamsPtr->importanceAreaTilt=parseResults_Src.importanceAreaTilt;
	this->rayParamsPtr->importanceAreaApertureType=parseResults_Src.importanceAreaApertureType;
	this->rayParamsPtr->epsilon=10*this->rayParamsPtr->lambda;//parseResults_Src.epsilon;
//	this->rayParamsPtr->oDetParams.apertureHalfWidth=make_double2(0.25,0.006);//parseResults_Det.apertureHalfWidth;//
//	this->rayParamsPtr->oDetParams.detPixel=parseResults_Det.detPixel;
	this->rayParamsPtr->oDetParams.MTransform=createTransformationMatrix(parseResults_Det.tilt, parseResults_Det.root);
	this->rayParamsPtr->oDetParams.tilt=parseResults_Det.tilt;
	this->rayParamsPtr->oDetParams.normal=parseResults_Det.normal;
	this->rayParamsPtr->oDetParams.geomID=parseResults_Det.geomID;
	this->rayParamsPtr->oDetParams.root=make_double3(0.25,0.018,11.4551040);//parseResults_Det.root;
	this->rayParamsPtr->oDetParams.outFormat=DET_OUT_MAT;
//	this->rayParamsPtr->importanceAreaType=

	// create refracting material
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
}

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
fieldError  DiffRayField_RayAiming_Holo::parseXml(pugi::xml_node &det, vector<Field*> &fieldVec)
{
	// call base class function
	if (FIELD_NO_ERR != DiffRayField_RayAiming::parseXml(det, fieldVec))
	{
		std::cout << "error in DiffRayField_RayAiming_Holo.parseXml(): RayField_RayAiming.parseXml()  returned an error." << std::endl;
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
};