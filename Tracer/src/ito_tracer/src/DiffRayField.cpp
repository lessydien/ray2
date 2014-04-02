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

/**\file DiffRayField.cpp
* \brief Rayfield for differential raytracing
* 
*           
* \author Mauch
*/

#include <omp.h>
#include "DiffRayField.h"
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
fieldError DiffRayField::setLambda(double lambda)
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
unsigned long long DiffRayField::getRayListLength(void)
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
fieldError DiffRayField::setRay(diffRayStruct ray, unsigned long long index)
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
diffRayStruct* DiffRayField::getRay(unsigned long long index)
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
diffRayStruct* DiffRayField::getRayList(void)
{
	return &rayList[0];	
};

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
void DiffRayField::setParamsPtr(diffRayFieldParams *paramsPtr)
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
diffRayFieldParams* DiffRayField::getParamsPtr(void)
{
	return this->rayParamsPtr;
};


/* functions for GPU usage */

//void DiffRayField::setPathToPtx(char* path)
//{
//	memcpy(this->path_to_ptx_rayGeneration, path, sizeof(this->path_to_ptx_rayGeneration));
//};
//
//const char* DiffRayField::getPathToPtx(void)
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
fieldError DiffRayField::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	RTvariable output_buffer;
	RTvariable seed_buffer;

	RTvariable params;
//    RTvariable radiance_ray_type;
	RTvariable max_depth;
	RTvariable min_flux;
    /* variables for ray gen program */
 //   RTvariable origin_max;
	//RTvariable origin_min;
	//RTvariable launch_width;
	//RTvariable launch_height;
	//RTvariable nrDirs;
	//RTvariable flux;
	//RTvariable lambda;
	//RTvariable rayDir;
	//RTvariable nImmersed;
	//RTvariable posDistrType;
	//RTvariable Mrot;
	//RTvariable translation;
	//RTvariable alphaMax;
	//RTvariable alphaMin;
	RTvariable offsetX;
	RTvariable offsetY;
	RTvariable epsilon;
//	RTvariable diff_epsilon;

	/* Ray generation program */
	char rayGenName[128];
	sprintf(rayGenName, "rayGeneration");
	switch (this->rayParamsPtr->dirDistrType)
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
		std::cout <<"error in DiffRayField.createOptixInstance(): unknown distribution of ray directions." << "...\n";
		return FIELD_ERR;
		break;
	}
	switch (this->rayParamsPtr->posDistrType)
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
		std::cout <<"error in DiffRayField.createOptixInstance(): unknown distribution of ray positions." << "...\n";
		return FIELD_ERR;
		break;
	}
	RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_rayGeneration, rayGenName, &this->ray_gen_program ), context);
	//RT_CHECK_ERROR_NOEXIT( rtProgramCreateFromPTXFile( context, this->path_to_ptx_rayGeneration, rayGenName, &this->ray_gen_program ) );

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
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetX, sizeof(long long), &(this->rayParamsPtr->launchOffsetX)), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetY, sizeof(long long), &(this->rayParamsPtr->launchOffsetY)), context ))
		return FIELD_ERR;

	//RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "diff_epsilon", &diff_epsilon ) );
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "params", &params ), context ))
		return FIELD_ERR;

	//double3 rayDirVar, origin_maxVar, origin_minVar;
	//
	//origin_maxVar=this->rayParamsPtr->rayPosEnd;	
	//origin_minVar=this->rayParamsPtr->rayPosStart;	

	//rayDirVar=this->rayParamsPtr->rayDirection;

	//double lambdaVar, fluxVar, nImmersedVar;
	//lambdaVar=this->rayParamsPtr->lambda;
	//fluxVar=this->rayParamsPtr->flux;
//	nImmersedVar=this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);
	this->rayParamsPtr->nImmersed=this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);

	if (FIELD_NO_ERR!=this->createCPUSimInstance())
	{
		std::cout <<"error in DiffRayField.createOptixInstance(): create CPUSimInstance() returned an error." << "...\n";
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

	////l_offsetX=0;
	////l_offsetY=0;
	//this->rayParamsPtr->launchOffsetX=0;//l_offsetX;
	//this->rayParamsPtr->launchOffsetY=0;//l_offsetY;

//	RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "radiance_ray_type", &radiance_ray_type ) );
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "scene_epsilon", &epsilon ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "max_depth", &max_depth ), context ))
		return FIELD_ERR;
	RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "min_flux", &min_flux ), context );

//    RT_CHECK_ERROR_NOEXIT( rtVariableSet1ui( radiance_ray_type, 0u ) );
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSet1f( epsilon, EPSILON ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSet1i( max_depth, MAX_DEPTH_CPU ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtVariableSet1f( min_flux, MIN_FLUX_CPU ), context ))
		return FIELD_ERR;

	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(diffRayFieldParams), this->rayParamsPtr), context) )
		return FIELD_ERR;

	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(origin_max, sizeof(double3), &origin_maxVar) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(origin_min, sizeof(double3), &origin_minVar) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(flux, sizeof(double), &fluxVar) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(lambda, sizeof(double), &lambdaVar) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(nImmersed, sizeof(double), &nImmersedVar) );
	////RT_CHECK_ERROR_NOEXIT( rtVariableSet1ui(number, numberVar ) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSet1ui(launch_width, l_launch_width ) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSet1ui(launch_height, l_launch_height ) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSet1ui(nrDirs, l_nrDirs ) );
	//
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(rayDir, sizeof(double3), &rayDirVar) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(posDistrType, sizeof(rayPosDistrType), &(this->rayParamsPtr->posDistrType)) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(Mrot, sizeof(double3x3), &(this->rayParamsPtr->Mrot)) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(translation, sizeof(double3), &(this->rayParamsPtr->translation)) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(alphaMax, sizeof(double2), &(this->rayParamsPtr->alphaMax)) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(alphaMin, sizeof(double2), &(this->rayParamsPtr->alphaMin)) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetX, sizeof(long long), &l_offsetX) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(offsetY, sizeof(long long), &l_offsetY) );
	//RT_CHECK_ERROR_NOEXIT( rtVariableSet1f( diff_epsilon, DIFF_EPSILON ) );

    if (!RT_CHECK_ERROR_NOEXIT( rtContextSetRayGenerationProgram( context,0, this->ray_gen_program ), context ))
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
fieldError DiffRayField::initCPUSubset()
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

			std::cout << "initalizing random seed" << "...\n";

			int seed = (int)time(0);            // random seed
			RandomInit(seed, x);

			// create random seeds for all the rays
			std::cout << "initializing rays on " << numCPU << " cores of CPU." << "...\n";

			for(signed long long jx=0;jx<this->rayParamsPtr->GPUSubset_width;jx++)
			{
				this->rayList[jx].currentSeed=(uint)BRandom(x);
			}
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize random seeds of " << this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";

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

				diffRayStruct rayData;
				rayData.flux=this->rayParamsPtr->flux;
				rayData.depth=0;	
				rayData.position.z=this->rayParamsPtr->rayPosStart.z;
				rayData.running=true;
				rayData.currentGeometryID=0;
				rayData.lambda=this->rayParamsPtr->lambda;
				rayData.nImmersed=1;//this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);
				double epsilon=this->rayParamsPtr->epsilon;//DIFF_EPSILON; // small distance. The ray is moved out of the caustic by this distance
				rayData.opl=epsilon;
				rayData.wavefrontRad=make_double2(-epsilon,-epsilon); // init wavefron radius according to small distance
				rayData.mainDirX=make_double3(1,0,0);
				rayData.mainDirY=make_double3(0,1,0);
				rayData.flux=this->rayParamsPtr->flux;

				// map on one dimensional index
				//unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*( floorf(this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y/this->rayParamsPtr->GPUSubset_width+1)*this->rayParamsPtr->GPUSubset_width);
				unsigned long long iGes=jx+this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;

				//std::cout << "iGes: " << iGes << "...\n";

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
						std::cout << "error in DiffRayField.initCPUSubset: unknown distribution of rayposition" << "...\n";
						// report error
						break;
				}
				// transform rayposition into global coordinate system
				rayData.position=this->rayParamsPtr->Mrot*rayData.position+this->rayParamsPtr->translation;

				double2 rayAngleHalfWidth, phi;

				switch (this->rayParamsPtr->dirDistrType)
				{
					case RAYDIR_UNIFORM:
						rayData.direction=this->rayParamsPtr->rayDirection;
						// transform raydirection into global coordinate system
						rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;

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
						//		std::cout << "error in DiffRayField.initCPUSubset: importance area for defining ray directions of source is only allowed with objects that have rectangular or elliptical apertures" << "...\n";
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
						std::cout << "error in Diff.initCPUSubset: unknown raydirection distribution" << "...\n";
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

				// move ray out of caustic
				rayData.position=rayData.position+epsilon*rayData.direction;
				rayData.currentSeed=(uint)BRandom(x);
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
			std::cout << " " << msecs <<" ms to initialize " << this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";
		}

		else if(this->rayParamsPtr->width*this->rayParamsPtr->height<1)
		{
			//not Possible. Report error or set n=-n
			std::cout << "error in DiffRayField.initCPUInstance: negative raynumber" << "...\n";
		}
		this->update=false;
	}	// end if GPUsubsetwidth*height<rayListLength
	else
	{
		std::cout << "error in DiffRayField.initCPUInstance: rayList is smaller than simulation subset" << "...\n";
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
long2 DiffRayField::calcSubsetDim()
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
fieldError DiffRayField::createCPUSimInstance()
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
		std::cout << "error in DiffRayField.createLayoutInstance(): memory for rayList could not be allocated. try to reduce ray tiling size" << "...\n";
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
fieldError DiffRayField::createLayoutInstance()
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
fieldError DiffRayField::write2TextFile(char* filename, detParams &oDetParams)
{
	char t_filename[512];
	//sprintf(t_filename, "%s%sDiffRayField_%i.txt", filename, PATH_SEPARATOR, oDetParams.subSetNr);
	sprintf(t_filename, "%s%sDiffRayField.txt", filename, PATH_SEPARATOR);

	FILE* hFileOut;
	hFileOut = fopen( t_filename, "a" ) ;
	if (!hFileOut)
	{
		std::cout << "error in DiffRayField.write2TextFile(): could not open output file: " << filename << "...\n";
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
					fprintf(hFileOut, "%.20lf ;%.20lf ;%.20lf; %.20lf ;%.20lf ;%.20lf ;%.20lf ;%.20lf; %i \n", rayList[rayListIndex].position.x, rayList[rayListIndex].position.y, rayList[rayListIndex].position.z, rayList[rayListIndex].direction.x, rayList[rayListIndex].direction.y, rayList[rayListIndex].direction.z, rayList[rayListIndex].flux, rayList[rayListIndex].opl, rayList[rayListIndex].currentGeometryID);
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
fieldError DiffRayField::traceScene(Group &oGroup, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	clock_t start, end;
	double msecs=0;

	// start timing
	start=clock();

//	long2 l_GPUSubsetDim=calcSubsetDim();

	if (RunOnCPU)
	{

		omp_set_num_threads(numCPU);

		std::cout << "tracing on " << numCPU << " cores of CPU." << "...\n";

		if (FIELD_NO_ERR!= initCPUSubset())
		{
			std::cout << "error in DiffRayField.traceScene: initCPUSubset returned an error" << "...\n";
			return FIELD_ERR;
		}

		std::cout << "starting the actual trace..." << "...\n";

#pragma omp parallel default(shared)
{
//			int id;
//			id = omp_get_thread_num();

//			std::cout << "Hello World from thread: " << id << "...\n";

//			std::cout << "test: " << "...\n";

		#pragma omp for schedule(dynamic, 500)

		for (signed long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				for(;;) // iterative tracing
				{
					if(!this->rayList[jx].running) 
						break;
					oGroup.trace(rayList[jx]);
				}
			}
}
	}
	else
	{
		//RTsize				buffer_width, buffer_height; // get size of output buffer
		void				*data; // pointer to cast output buffer into
 		//rayStruct			*bufferData;
		std::cout << "tracing on GPU." << "...\n";

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
		//std::cout << "DEBUG: jx=" << jx << " jy=" << jy << "...\n";
		//copyRayListSubset((rayStruct*)data, l_launchOffset, l_GPUSubsetDim);
		if (FIELD_NO_ERR != copyRayList((diffRayStruct*)data,this->rayParamsPtr->GPUSubset_width) )
		{
			std::cout << "error in GeometricRayField.traceScene(): copyRayList() returned an error" << "...\n";
			return FIELD_NO_ERR;
		}
		
		if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( output_buffer_obj ), context ))
			return FIELD_ERR;
	}

	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	std::cout << " " << msecs <<" ms to trace " << this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";

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
fieldError DiffRayField::traceStep(Group &oGroup, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	clock_t start, end;
	double msecs=0;

	// start timing
	start=clock();

		std::cout << "tracing steps on " << numCPU << " cores of CPU." << "...\n";

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
	std::cout << " " << msecs <<" ms to trace " << this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";

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
fieldError DiffRayField::copyRayList(diffRayStruct *data, long long length)
{
	if (length > this->rayListLength)
	{
		std::cout << "error in DiffRayField.copyRayList(): subset dimensions exceed rayLIst dimension" << "...\n";
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
fieldError DiffRayField::copyRayListSubset(diffRayStruct *data, long2 launchOffset, long2 subsetDim)
{
//	long2 testOffset=launchOffset;
//	long2 testDim=subsetDim;
	//  ----memory range of completed lines---- + ---memory range blocks in given line---
	if (launchOffset.y*this->rayParamsPtr->width+(subsetDim.x+launchOffset.x)*subsetDim.y > this->rayListLength)
	{
		std::cout << "error in GeometricRayField.copyRayListSubset(): subset dimensions exceed rayLIst dimension" << "...\n";
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
fieldError DiffRayField::writeData2File(FILE *hFile, rayDataOutParams outParams)
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
fieldError DiffRayField::convert2RayData(Field** imagePtrPtr, detParams &oDetParams)
{
	DiffRayField* l_ptr;
	// if there is no image yet, create one
	if (*imagePtrPtr == NULL)
	{
		*imagePtrPtr=new DiffRayField(this->rayListLength);
		l_ptr=dynamic_cast<DiffRayField*>(*imagePtrPtr);
	}
	else
	{
		l_ptr=dynamic_cast<DiffRayField*>(*imagePtrPtr);
		if ( l_ptr->rayListLength < this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width )
		{
			std::cout << "error in GeometricRayField.convert2RayData(): dimensions of image does not fit dimensions of raylist subset" << "...\n";
			return FIELD_ERR;
		}
	}
	// copy the rayList and the respective parameters
	memcpy(l_ptr->getRayList(), this->rayList, (GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX)*sizeof(diffRayStruct));
	diffRayFieldParams *l_diffRayParamsPtr=new diffRayFieldParams();
	memcpy(l_diffRayParamsPtr,this->rayParamsPtr, sizeof(diffRayFieldParams));
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
fieldError DiffRayField::convert2Intensity(Field* imagePtr, detParams &oDetParams)
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
		std::cout << "error in DiffRayField.convert2Intensity(): imagePtr is not of type IntensityField" << "...\n";
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

	complex<double> i_compl=complex<double>(0,1); // define complex number "i"

//	std::cout << "processing on " << numCPU << " cores of CPU." << "...\n";

//#pragma omp parallel default(shared)
//{
//	#pragma omp for schedule(dynamic, 50)

	for ( long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
	{
		// transform to local coordinate system
		double3 tmpPos=this->rayList[jx].position-offset;
		rotateRayInv(&tmpPos,oDetParams.tilt);

		index.x=floor((tmpPos.x)/scale.x);
		index.y=floor((tmpPos.y)/scale.y);
		index.z=floor((tmpPos.z)/scale.z);
			
		// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
		if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
		{
			double phi=2*PI/this->rayList[jx].lambda*this->rayList[jx].opl;
			complex<double> l_U=complex<double>(this->rayList[jx].flux*cos(phi),this->rayList[jx].flux*sin(phi));
			l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+l_U; // create a complex amplitude from the rays flux and opl and sum them coherently
			hitNr++;
		}
		else
		{
			std::cout << " ray " << jx << " not in target: " << rayList[jx].position.x << "; " << rayList[jx].position.y << "; " << rayList[jx].position.z << "...\n";
		}

	}
//}
	// if this is the last subset of the current launch, convert complex amplitude to intensity
	if ( this->rayParamsPtr->launchOffsetX+this->rayParamsPtr->GPUSubset_width+this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y >= this->rayParamsPtr->width*this->rayParamsPtr->height*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y )
	{
		std::cout << " finally converting scalar field to intensity" << "...\n";
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
	std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_width << " rays in target" << "...\n";

	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<"ms to process " << this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";

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
fieldError DiffRayField::convert2ScalarField(Field* imagePtr, detParams &oDetParams)
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
		std::cout << "error in DiffRayField.convert2ScalarField(): imagePtr is not of type ScalarField" << "...\n";
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
	long3 nrPixels=l_ScalarImagePtr->getParamsPtr()->nrPixels;
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

	// save normal
	double3 t_normal=t_ez;

	complex<double> i_compl=complex<double>(0,1); // define complex number "i"

//	std::cout << "processing on " << numCPU << " cores of CPU." << "...\n";

//#pragma omp parallel default(shared)
//{
//	#pragma omp for schedule(dynamic, 50)

	for (long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
	{
		// transform to local coordinate system
		double3 tmpPos=this->rayList[jx].position-offset;
		rotateRayInv(&tmpPos,oDetParams.tilt);

		index.x=floor((tmpPos.x)/scale.x);
		index.y=floor((tmpPos.y)/scale.y);
		index.z=floor((tmpPos.z)/scale.z);
			
		// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
		if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
		{
			// phase from OPL of differential ray
			double phi=std::fmod((2*PI/this->rayList[jx].lambda*this->rayList[jx].opl),(2*M_PI));
			// we want to compute the field value at the centre of the pixel. Therefore we need to make som corrections in case the ray doesn't hit the Pixel at its centre
			// calc vector from differential ray to centre of pixel
			double3 PixelOffset=posMinOffset-(index.x*t_ex+index.y*t_ey+index.z*t_ez);
			// calc projection of this vector onto the ray direction
			double dz=dot(this->rayList[jx].direction,PixelOffset);
			// calc additional phase at centre of pixel from linear approximation to local wavefront
			phi=phi+dz*2*M_PI/this->rayList[jx].lambda;
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
			//l_ScalarImagePtr->getFieldPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_U;
			l_ScalarImagePtr->getFieldPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_ScalarImagePtr->getFieldPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+l_U; // create a complex amplitude from the rays flux and opl and sum them coherently
			hitNr++;
		}
		else
		{
			std::cout << " ray " << jx << " not in target: " << rayList[jx].position.x << "; " << rayList[jx].position.y << "; " << rayList[jx].position.z << "...\n";
		}

	}
//}
	//l_ScalarImagePtr->getFieldPtr()[98]=polar(100,0);
	//l_ScalarImagePtr->getFieldPtr()[99]=polar(100,0);
	std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_width << " rays in target" << "...\n";
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << msecs <<"ms to process " << this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";


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
fieldError DiffRayField::convert2PhaseSpace(Field* imagePtr, detParams &oDetParams)
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
		std::cout << "error in GeometricRayField.convert2PhaseSpaceField(): imagePtr is not of type IntensityField" << "...\n";
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
		std::cout << "error in GeometricRayField.convert2PhaseSpaceField(): coherent conversion is not defined!!" << "...\n";
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
			//	//	std::cout <<  "ray number " << j << " did not hit target." << "x: " << rayList[j].position.x << ";y: " << rayList[j].position.y << "z: " << rayList[j].position.z << ";geometryID " << rayList[j].currentGeometryID << "...\n";
			//	//}
			//}
			std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays in target" << "...\n";
		}
		else
		{
			std::cout << "error in GeometricRayField.convert2PhaseSpaceField(): partial coherence not implemented yet" << "...\n";
			return FIELD_ERR;
		}

	}
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << " " << msecs <<"ms to process " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";

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
//fieldError DiffRayField::convert2VecField(Field* imagePtr, detParams &oDetParams)
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
fieldError DiffRayField::processParseResults(FieldParseParamStruct &parseResults_Src, parseGlassResultStruct* parseResults_GlassPtr)
{
	// no importance area and direction distribution GRID_RECT is not allowed
	if ( !parseResults_Src.importanceArea && ( (parseResults_Src.rayDirDistr==RAYDIR_GRID_RECT)||(parseResults_Src.rayDirDistr==RAYDIR_GRID_RAD) ) )
	{
		std::cout <<"error in DiffRayField.processParseResults(): direction distribution GRID_RECT and GRID_RAD are not allowed if no importance area is defined" << "...\n";
		return FIELD_ERR;
	}
	//if (parseResults_Src.rayDirDistr == RAYDIR_UNIFORM)
	//{
	//	std::cout <<"error in DiffRayField.processParseResults(): direction distribution UNIFORM is not allowed for differential ray fields" << "...\n";
	//	return FIELD_ERR;
	//}
	if ( (parseResults_Src.nrRayDirections.x*parseResults_Src.nrRayDirections.y > GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_WIDTH_MAX) && (parseResults_Src.rayDirDistr == RAYDIR_RAND_RECT) && ( (parseResults_Src.rayPosDistr == RAYPOS_RAND_RECT) || (parseResults_Src.rayPosDistr == RAYPOS_RAND_RAD) ) )
	{
		std::cout <<"warning in DiffRayField.processParseResults(): a number of ray directions that is bigger than the size of a GPU subset in combination with random position and direction distribution leads to a situation where some rays per point source point into the same direction when tracing on GPU." << "...\n";
	}

//	this->rayParamsPtr=new diffRayFieldParams;
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
	this->rayParamsPtr->widthLayout=parseResults_Src.widthLayout;
	this->rayParamsPtr->heightLayout=parseResults_Src.heightLayout;
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
			std::cout <<"warning in GeometricRayField.processParseResults(): unknown material. Rafracting material with n=1 assumed." << "...\n";
			break;
	}
	return FIELD_NO_ERR;
}

void DiffRayField::setSimMode(SimMode &simMode)
{
    simMode=SIM_DIFF_RT;
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
fieldError  DiffRayField::parseXml(pugi::xml_node &det, vector<Field*> &fieldVec, SimParams simParams)
{
    this->setSimMode(simParams.simMode);

    this->getParamsPtr()->epsilon=EPSILON;

	// call base class function
	if (FIELD_NO_ERR != RayField::parseXml(det, fieldVec, simParams))
	{
		std::cout << "error in DiffRayField.parseXml(): RayField.parseXml()  returned an error." << "...\n";
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
};