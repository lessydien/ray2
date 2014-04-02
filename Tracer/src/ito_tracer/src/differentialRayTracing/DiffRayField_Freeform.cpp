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

/**\file DiffRayField_Freeform.cpp
* \brief Rayfield for differential raytracing
* 
*           
* \author Mauch
*/

#include <fstream>
#include <iostream>
#include <iomanip>

#include <omp.h>
#include "DiffRayField_Freeform.h"
#include "../myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "../Geometry.h"
#include "math.h"
#include "../randomGenerator.h"

using namespace optix;
///**
// * \detail getRayListLength 
// *
// * \param[in] void
// * 
// * \return unsigned long long
// * \sa 
// * \remarks 
// * \author Mauch
// */
//unsigned long long DiffRayField_Freeform::getRayListLength(void)
//{
//	return this->rayListLength;
//};

///**
// * \detail setRay 
// *
// * \param[in] diffRayStruct ray, unsigned long long index
// * 
// * \return fieldError
// * \sa 
// * \remarks 
// * \author Mauch
// */
//fieldError DiffRayField_Freeform::setRay(diffRayStruct ray, unsigned long long index)
//{
//	if (index <= this->rayListLength)
//	{
//		rayList[index]=ray;
//		return FIELD_NO_ERR;
//	}
//	else
//	{
//		return FIELD_INDEXOUTOFRANGE_ERR;
//	}
//};

///**
// * \detail getRay 
// *
// * \param[in] unsigned long long index
// * 
// * \return diffRayStruct*
// * \sa 
// * \remarks 
// * \author Mauch
// */
//diffRayStruct* DiffRayField_Freeform::getRay(unsigned long long index)
//{
//	if (index <= this->rayListLength)
//	{
//		return &rayList[index];	
//	}
//	else
//	{
//		return 0;
//	}
//};

///**
// * \detail getRayList 
// *
// * \param[in] void
// * 
// * \return diffRayStruct*
// * \sa 
// * \remarks 
// * \author Mauch
// */
//diffRayStruct* DiffRayField_Freeform::getRayList(void)
//{
//	return &rayList[0];	
//};
//
///**
// * \detail setParamsPtr 
// *
// * \param[in] diffRayFieldParams *paramsPtr
// * 
// * \return void
// * \sa 
// * \remarks 
// * \author Mauch
// */
//void DiffRayField_Freeform::setParamsPtr(diffRayFieldParams *paramsPtr)
//{
//	this->rayParamsPtr=paramsPtr;
//	this->update=true;
//};
//
///**
// * \detail getParamsPtr 
// *
// * \param[in] void
// * 
// * \return diffRayFieldParams*
// * \sa 
// * \remarks 
// * \author Mauch
// */
//diffRayFieldParams* DiffRayField_Freeform::getParamsPtr(void)
//{
//	return this->rayParamsPtr;
//};
//
//
///* functions for GPU usage */
//
////void DiffRayField_Freeform::setPathToPtx(char* path)
////{
////	memcpy(this->path_to_ptx_rayGeneration, path, sizeof(this->path_to_ptx_rayGeneration));
////};
////
////const char* DiffRayField_Freeform::getPathToPtx(void)
////{
////	return this->path_to_ptx_rayGeneration;
////};

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
fieldError DiffRayField_Freeform::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	RTvariable output_buffer;

	RTvariable freeForm_buffer;
	RTvariable freeForm_y2a_buffer;
	RTvariable freeForm_x1a_buffer;
	RTvariable freeForm_x2a_buffer;

	RTbuffer freeForm_ya_t_buffer_obj;
	RTbuffer freeForm_y2a_t_buffer_obj;
	RTbuffer freeForm_u_buffer_obj;
	RTbuffer freeForm_yytmp_buffer_obj;
	RTbuffer freeForm_ytmp_buffer_obj;

	RTvariable freeForm_ya_t_buffer;
	RTvariable freeForm_y2a_t_buffer;
	RTvariable freeForm_u_buffer;
	RTvariable freeForm_yytmp_buffer;
	RTvariable freeForm_ytmp_buffer;

	RTvariable params;

	if (FIELD_NO_ERR != RayField::createOptixInstance(context, output_buffer_obj, seed_buffer_obj))
	{
		std::cout <<"error in DiffRayField_Freeform.createOptixInstance(): RayField.creatOptiXInstance() returned an error." << "...\n";
		return FIELD_ERR;
	}

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

	/* declare freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "freeForm_buffer", &freeForm_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &freeForm_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_buffer_obj, 100*100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_buffer, freeForm_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare freeForm_y2a buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "y2a", &freeForm_y2a_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &freeForm_y2a_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_y2a_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_y2a_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_y2a_buffer_obj, 100*100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_y2a_buffer, freeForm_y2a_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare freeForm_x1a buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "x1a", &freeForm_x1a_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &freeForm_x1a_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_x1a_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_x1a_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_x1a_buffer_obj, 100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_x1a_buffer, freeForm_x1a_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare freeForm_x1a buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "x2a", &freeForm_x2a_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT, &freeForm_x2a_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_x2a_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_x2a_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_x2a_buffer_obj, 100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_x2a_buffer, freeForm_x2a_buffer_obj ), context ))
		return FIELD_ERR;

	/* fill freeForm buffer */
	char freeFormFile[128];
	sprintf( freeFormFile, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "freeform.txt" );
	std::ifstream inFile(freeFormFile);

	// make sure the filestream is good
	if (!inFile)
	{
		std::cout << "error in DiffRayField_Freeform.createOptiXInstance(): failed to open freeform file " << "...\n";
		return FIELD_ERR;
	}

	void *data;
	// read the seed buffer from the GPU
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(freeForm_buffer_obj, &data), context ))
		return FIELD_ERR;
	freeForm_buffer_CPU = reinterpret_cast<double*>( data );
	//freeForm_buffer_CPU=(double*)malloc(100*100*sizeof(double));

	char test;
	RTsize buffer_width;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferGetSize1D(freeForm_buffer_obj, &buffer_width), context ))
		return FIELD_ERR;
	for ( unsigned int i = 0; i < (unsigned int)buffer_width; ++i )
	{
		if (inFile.eof())
		{
			std::cout << "error in DiffRayField_Freeform.createOptiXInstance(): end of file of freeform file before all points were read " << "...\n";
			return FIELD_ERR;
		}
		inFile >> freeForm_buffer_CPU[i];
		//inFile >> test;
		freeForm_buffer_CPU[i]=freeForm_buffer_CPU[i]*1e3;
	}
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( freeForm_buffer_obj ), context ))
		return FIELD_ERR;

	void *data_y2a;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(freeForm_y2a_buffer_obj, &data_y2a), context ))
		return FIELD_ERR;
	freeForm_y2a_buffer_CPU = reinterpret_cast<double*>( data_y2a );

	// create array for 2nd derivative of freeform data along x-dimension
	void *data_x1a;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(freeForm_x1a_buffer_obj, &data_x1a), context ))
		return FIELD_ERR;
	freeForm_x1a_buffer_CPU = reinterpret_cast<double*>( data_x1a );
	double x10=-2.5;
	double deltaW= 5.0/100;
	void *data_x2a;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferMap(freeForm_x2a_buffer_obj, &data_x2a), context ))
		return FIELD_ERR;
	freeForm_x2a_buffer_CPU = reinterpret_cast<double*>( data_x2a );

	double x20=-2.5;
	double deltaH= 5.0/100;
	for (unsigned int i=0; i<100;i++)
	{
		this->freeForm_x1a_buffer_CPU[i]=x10+deltaW/2+i*deltaW;
		this->freeForm_x2a_buffer_CPU[i]=x20+deltaH/2+i*deltaH;
	}
	this->splie2(this->freeForm_x1a_buffer_CPU, this->freeForm_x2a_buffer_CPU, this->freeForm_buffer_CPU, freeForm_y2a_buffer_CPU, 100, 100);

	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( freeForm_y2a_buffer_obj ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( freeForm_x1a_buffer_obj ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferUnmap( freeForm_x2a_buffer_obj ), context ))
		return FIELD_ERR;

	//RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "diff_epsilon", &diff_epsilon ) );
	if (!RT_CHECK_ERROR_NOEXIT( rtProgramDeclareVariable( this->ray_gen_program, "params", &params ), context ))
		return FIELD_ERR;

	/* declare ya_t buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "ya_t", &freeForm_ya_t_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT_OUTPUT, &freeForm_ya_t_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_ya_t_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_ya_t_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_ya_t_buffer_obj, 100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_ya_t_buffer, freeForm_ya_t_buffer_obj ), context ))

	/* declare y2a_t buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "y2a_t", &freeForm_y2a_t_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT_OUTPUT, &freeForm_y2a_t_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_y2a_t_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_y2a_t_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_y2a_t_buffer_obj, 100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_y2a_t_buffer, freeForm_y2a_t_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare u buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "u", &freeForm_u_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT_OUTPUT, &freeForm_u_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_u_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_u_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_u_buffer_obj, 100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_u_buffer, freeForm_u_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare ytmp buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "ytmp", &freeForm_ytmp_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT_OUTPUT, &freeForm_ytmp_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_ytmp_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_ytmp_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_ytmp_buffer_obj, 100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_ytmp_buffer, freeForm_ytmp_buffer_obj ), context ))
		return FIELD_ERR;

	/* declare yytmp buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtContextDeclareVariable( context, "yytmp", &freeForm_yytmp_buffer ), context ))
		return FIELD_ERR;
    /* Render freeForm buffer */
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferCreate( context, RT_BUFFER_INPUT_OUTPUT, &freeForm_yytmp_buffer_obj ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetFormat( freeForm_yytmp_buffer_obj, RT_FORMAT_USER ), context ))
		return FIELD_ERR;
	if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetElementSize( freeForm_yytmp_buffer_obj, sizeof(double) ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtBufferSetSize1D( freeForm_yytmp_buffer_obj, 100 ), context ))
		return FIELD_ERR;
    if (!RT_CHECK_ERROR_NOEXIT( rtVariableSetObject( freeForm_yytmp_buffer, freeForm_yytmp_buffer_obj ), context ))
		return FIELD_ERR;

	if (FIELD_NO_ERR!=this->createCPUSimInstance())
	{
		std::cout <<"error in DiffRayField_Freeform.createOptixInstance(): create CPUSimInstance() returned an error." << "...\n";
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

	if ( !RT_CHECK_ERROR_NOEXIT( rtVariableSetUserData(params, sizeof(diffRayFieldParams), this->rayParamsPtr), context) )
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
fieldError DiffRayField_Freeform::initCPUSubset()
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

			// start timing
			start=clock();

			std::cout << "initalizing random seed" << "...\n";

			int seed = (int)time(0);            // random seed
			RandomInit(seed, x);


			// create random seeds for all the rays
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
		std::cout << "initializing rays on " << numCPU << " cores of CPU." << "...\n";
		omp_set_num_threads(numCPU);

#pragma omp parallel default(shared)
{
			#pragma omp for schedule(dynamic, 50)//schedule(static)//

			for(signed long long jx=0;jx<this->rayParamsPtr->GPUSubset_width;jx++)
			{
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
				double3 dirImpAreaCentre, tmpPos, impAreaRoot, rayAngleCentre;
				double3 impAreaAxisX, impAreaAxisY;
				double impAreaX, impAreaY, theta;

				double3 alpha=make_double3(0,0,0); // rotation angle of raydirection around x and y
				unsigned long long nrGes=0;

				diffRayStruct rayData;
				rayData.flux=this->rayParamsPtr->flux;
				rayData.depth=0;	
				rayData.position.z=this->rayParamsPtr->rayPosStart.z;
				rayData.running=true;
				rayData.currentGeometryID=0;
				rayData.lambda=this->rayParamsPtr->lambda;
				rayData.nImmersed=1;//this->materialList[0]->calcSourceImmersion(this->rayParamsPtr->lambda);
				double epsilon=this->rayParamsPtr->epsilon;//DIFF_EPSILON; // small distance. The ray is moved out of the caustic by this distance
				rayData.opl=epsilon; // the opl is initialized with the small value the ray will be moved out of the caustic later
				rayData.wavefrontRad=make_double2(-epsilon,-epsilon); // init wavefron radius according to small distance
				rayData.mainDirX=make_double3(1,0,0);
				rayData.mainDirY=make_double3(0,1,0);

				uint32_t x_l[5];
				RandomInit(this->rayList[jx].currentSeed, x_l); // seed random generator

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
						R=(deltaRx/2+deltaRx*iPosY)*(deltaRy/2+deltaRy*iPosY)/sqrt(pow((deltaRy/2+deltaRy*iPosY)*cos(deltaPhi/2+deltaPhi*iPosX),2)+pow((deltaRx/2+deltaRx*iPosY)*sin(deltaPhi/2+deltaPhi*iPosX),2));							
						if (deltaRy==0)
							R=0;
						// now calc rectangular coordinates from polar coordinates
						rayData.position.x=cos(deltaPhi/2+deltaPhi*iPosX)*R;
						rayData.position.y=sin(deltaPhi/2+deltaPhi*iPosX)*R;

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
						//	rayData.position.x=(Random(x)-0.5)*physWidth;
						//	rayData.position.y=(Random(x)-0.5)*physHeight;
						//	r=rayData.position.x*rayData.position.x/(physWidth*physWidth/4)+rayData.position.y*rayData.position.y/(physHeight*physHeight/4);
						//	index++;
						//	if (index>1000000)
						//		break;
						//} while ( (r >= 1.0) );
						break;
					default:
						rayData.position=make_double3(0,0,0);
						std::cout << "error in DiffRayField_Freeform.createCPUInstance: unknown distribution of rayposition" << "...\n";
						// report error
						break;
				}
				// interpolate z-position from data table
				//this->splin2(this->freeForm_x1a_buffer_CPU, this->freeForm_x2a_buffer_CPU, 100, 100, this->freeForm_buffer_CPU, this->freeForm_y2a_buffer_CPU, rayData.position.x, rayData.position.y, rayData.position.z);
				this->oInterpPtr->doInterpolation(this->freeForm_x1a_buffer_CPU, this->freeForm_x2a_buffer_CPU, 100, 100, this->freeForm_buffer_CPU, rayData.position.x, rayData.position.y, &(rayData.position.z));
				
				/*******************************************************************************
				* this is only to check wether the freeform file is given with wrong sign
				********************************************************************************/
				rayData.position.z=-rayData.position.z;
				rayData.opl=rayData.opl+100-rayData.nImmersed*rayData.position.z; // As this freeform is an reflector and the created rays are assumed to have travelled from infinity to the freeform, we need to give the ray another phase offset according to the height profile of the frreform

				// transform rayposition into global coordinate system
				rayData.position=this->rayParamsPtr->Mrot*rayData.position+this->rayParamsPtr->translation;

				switch (this->rayParamsPtr->dirDistrType)
				{
					case RAYDIR_UNIFORM:
						rayData.direction=this->rayParamsPtr->rayDirection;
						// transform raydirection into global coordinate system
						rayData.direction=this->rayParamsPtr->Mrot*rayData.direction;
						break;
					case RAYDIR_RAND_RECT:
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
						aimRayTowardsImpArea(rayData.direction, rayData.position, this->rayParamsPtr->importanceAreaRoot, this->rayParamsPtr->importanceAreaHalfWidth, this->rayParamsPtr->importanceAreaTilt, this->rayParamsPtr->importanceAreaApertureType, rayData.currentSeed);
//						if (this->rayParamsPtr->importanceAreaApertureType==AT_RECT)
//						{
//							// place temporal point uniformingly randomly inside the importance area
//							impAreaX=(Random(x_l)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.x;
//							impAreaY=(Random(x_l)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.y; 
//						}
//						else
//						{
//							if (this->rayParamsPtr->importanceAreaApertureType==AT_ELLIPT)
//							{
//								theta=2*PI*Random(x_l);
//								r=sqrt(Random(x_l));
//								impAreaX=this->rayParamsPtr->importanceAreaHalfWidth.x*r*cos(theta);
//								impAreaY=this->rayParamsPtr->importanceAreaHalfWidth.y*r*sin(theta);
//							}
//							else
//							{
//								std::cout << "error in DiffRayField_Freeform.createCPUInstance: importance area for defining ray directions of source is only allowed with objects that have rectangular or elliptical apertures" << "...\n";
////								return FIELD_ERR;
//							}
//						}
//
//						impAreaAxisX=make_double3(1,0,0);
//						impAreaAxisY=make_double3(0,1,0);
//						rotateRay(&impAreaAxisX,this->rayParamsPtr->importanceAreaTilt);
//						rotateRay(&impAreaAxisY,this->rayParamsPtr->importanceAreaTilt);
//
//						tmpPos=this->rayParamsPtr->importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
//						rayData.direction=normalize(tmpPos-rayData.position);
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
					//	impAreaX=(Random(x)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.x;
					//	impAreaY=(Random(x)-0.5)*2*this->rayParamsPtr->importanceAreaHalfWidth.y; 
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
					//	theta=2*PI*Random(x);
					//	r=sqrt(Random(x));
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
						std::cout << "error in GeometricRayField.updateCPUInstance: unknown raydirection distribution" << "...\n";
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
				//theta=2*PI*Random(x);
				//r=sqrt(Random(x));
				//impAreaX=impAreaHalfWidth.x*r*cos(theta);
				//impAreaY=impAreaHalfWidth.y*r*sin(theta);
				//tmpPos=impAreaRoot+impAreaX*make_double3(1,0,0)+impAreaY*make_double3(0,1,0);
				//rayData.direction=normalize(tmpPos-make_double3(0,0,0));

				// move ray out of caustic
				rayData.position=rayData.position+epsilon*rayData.direction;
				rayData.currentSeed=(uint)BRandom(x);
				// adjust flux
				rayData.flux=1/(epsilon*epsilon)*rayData.flux;
				//further adjust flux??
				//rayData.flux=rayData.flux*abs(dot(rayData.direction,make_double3(0,0,1)));
				// create main directions
				// calc angles with respect to global x- and y-axis
				double2 phi=calcAnglesFromVector(rayData.direction,this->rayParamsPtr->tilt);
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
			} // end for
} // end omp
			end=clock();
			msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
			std::cout << " " << msecs <<" ms to initialize " << this->rayParamsPtr->GPUSubset_width << " rays." << "...\n";

		}
		else if(this->rayParamsPtr->width*this->rayParamsPtr->height<1)
		{
			//not Possible. Report error or set n=-n
			std::cout << "error in DiffRayField_Freeform.initCPUInstance: negative raynumber" << "...\n";
		}
		this->update=false;
	}	// end if GPUsubsetwidth*height<rayListLength
	else
	{
		std::cout << "error in DiffRayField_Freeform.initCPUInstance: rayList is smaller than simulation subset" << "...\n";
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
};
//
///**
// * \detail calcSubsetDim 
// *
// * \param[in] void
// * 
// * \return void
// * \sa 
// * \remarks 
// * \author Mauch
// */
//long2 DiffRayField_Freeform::calcSubsetDim()
//{
//	unsigned long long width=this->rayParamsPtr->width*this->rayParamsPtr->height*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;
//
//	long2 l_GPUSubsetDim;
//
//	// calc launch_width of current launch
//	long long restWidth=width-this->rayParamsPtr->launchOffsetX-this->rayParamsPtr->launchOffsetY*this->rayParamsPtr->width*this->rayParamsPtr->nrRayDirections.x*this->rayParamsPtr->nrRayDirections.y;
//	// if the restWidth is smaller than the maximum subset-width. Take restWidth
//	if (restWidth < this->rayParamsPtr->GPUSubset_width)
//	{
//		l_GPUSubsetDim.x=restWidth;
//	}
//	else
//	{
//		l_GPUSubsetDim.x=this->rayParamsPtr->GPUSubset_width;
//	}
//	// we need to set to one
//	l_GPUSubsetDim.y=1;
//	this->rayParamsPtr->GPUSubset_height=l_GPUSubsetDim.y;
//	this->rayParamsPtr->GPUSubset_width=l_GPUSubsetDim.x;
//	return l_GPUSubsetDim;
//};

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
fieldError DiffRayField_Freeform::createCPUSimInstance()
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
		std::cout << "error in DiffRayField_Freeform.createLayoutInstance(): memory for rayList could not be allocated. try to reduce ray tiling size" << "...\n";
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

	/* fill freeForm buffer */
	char freeFormFile[128];
	sprintf( freeFormFile, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "freeform.txt" );
	std::ifstream inFile(freeFormFile);

	// make sure the filestream is good
	if (!inFile)
	{
		std::cout << "error in DiffRayFreeform.createOptiXInstance(): failed to open freeform file " << "...\n";
		//return FIELD_ERR;
	}

	// create array for freeform data
	freeForm_buffer_CPU=(double*)malloc(100*100*sizeof(double));
	char test;
	for ( unsigned int i = 0; i < 100*100; ++i )
	{
		if (inFile.eof())
		{
			std::cout << "error in DiffRayFreeform.createOptiXInstance(): end of file of freeform file before all points were read " << "...\n";
			//return FIELD_ERR;
		}
		inFile >> freeForm_buffer_CPU[i];
//		inFile >> test;
		freeForm_buffer_CPU[i]=freeForm_buffer_CPU[i]*1e3;
	}

	//// create array for 2nd derivative of freeform data along x-dimension
	//freeForm_y2a_buffer_CPU=(double*)malloc(100*100*sizeof(double));
	this->freeForm_x1a_buffer_CPU=(double*)malloc(100*sizeof(double));
	double x10=-2.5;
	double deltaW= 5.0/100;
	this->freeForm_x2a_buffer_CPU=(double*)malloc(100*sizeof(double));
	double x20=-2.5;
	double deltaH= 5.0/100;
	for (unsigned int i=0; i<100;i++)
	{
		this->freeForm_x1a_buffer_CPU[i]=x10+deltaW/2+i*deltaW;
		this->freeForm_x2a_buffer_CPU[i]=x20+deltaH/2+i*deltaH;
	}
	//this->splie2(this->freeForm_x1a_buffer_CPU, this->freeForm_x2a_buffer_CPU, this->freeForm_buffer_CPU, freeForm_y2a_buffer_CPU, 100, 100);

	this->oInterpPtr->initInterpolation(this->freeForm_x1a_buffer_CPU,this->freeForm_x2a_buffer_CPU, freeForm_buffer_CPU, 100, 100);

	return FIELD_NO_ERR;
};


// compute second derivative y2 of tabulated function y(x), given values for the first derivative on the borders yp1 and ypn. A value of 1e30 or greater sets boundary conditions for a natural spline, i.e. for the second derivative on the borders to be zero.
// see Numerical recipes in C++ second edition pp.118 for reference
void DiffRayField_Freeform::spline(double *x, double *y, const unsigned int width, const double yp1,  const double ypn, double *y2)
{
	int i,k;
	double p,qn,sig,un;

	double *u;
	u=(double*)malloc(width*sizeof(double));
	if (yp1 > 0.99e30)
		y2[0]=u[0]=0.0;
	else
	{
		y2[0]=-0.5;
		u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
	}
	for (i=1;i<width-1;i++)
	{
		sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]);
		u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
	}
	if (ypn > 0.99e30)
		qn=un=0.0;
	else
	{
		qn=0.5;
		un=(3.0/(x[width-1]-x[width-2]))*(ypn-(y[width-1]-y[width-2])/(x[width-1]-x[width-2]));
	}
	y2[width-1]=(un-qn*u[width-2])/(qn*y2[width-2]+1.0);
	for (k=width-2;k>=0;k--)
		y2[k]=y2[k]*y2[k+1]+u[k];

	delete u;
	u=NULL;
}

// compute the value y(x), of a tabulated function y(x) given the tables ya, xa, and the tabulatedt values of the second derivative y2a
// see Numerical recipes in C++ second edition pp.119 for reference
void DiffRayField_Freeform::splint(double *xa, double *ya, double *y2a, const unsigned int width, const double x,  double &y)
{
	int k;
	double h,b,a;

	int klo=0;
	int khi=width-1;
	while (khi-klo > 1) // find bucket of x in xa
	{
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	if (h==0.0) 
	{
		std::cout << "error in DiffRayField_Freeform.splint(): the tabulated x-values must be distinct" << "...\n";
	}
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

// see Numerical recipes in C++ second edition pp.131 for reference
void DiffRayField_Freeform::splie2(double *x1a, double *x2a, double *ya, double *y2a, const unsigned int width, const unsigned int height)
{
	unsigned int j, k;

	double *ya_t, *y2a_t;
	ya_t=(double*)malloc(100*sizeof(double));
	y2a_t=(double*)malloc(100*sizeof(double));

	for (j=0;j<height;j++)
	{
		for (k=0;k<width;k++)
			ya_t[k]=ya[k+j*width];
		spline(x2a,ya_t,width,1.0e30,1.0e30,y2a_t);
		for (k=0;k<width;k++)
			y2a[k+j*width]=y2a_t[k];
	}
	delete ya_t;
	ya_t=NULL;
	delete y2a_t;
	y2a_t=NULL;
}

// see Numerical recipes in C++ second edition pp.131 for reference
void DiffRayField_Freeform::splin2(double *x1a, double *x2a, const unsigned int width, const unsigned int height, double *ya, double *y2a, const double x1, const double x2, double &y)
{
	int j,k;

	double *ya_t, *y2a_t, *yytmp, *ytmp;
	ya_t=(double*)malloc(width*sizeof(double));
	y2a_t=(double*)malloc(width*sizeof(double));
	yytmp=(double*)malloc(height*sizeof(double));
	ytmp=(double*)malloc(height*sizeof(double));

	for (j=0;j<height;j++)
	{
		for (k=0;k<width;k++)
		{
			ya_t[k]=ya[k+j*width];
			y2a_t[k]=y2a[k+j*width];
		}
		this->splint(x2a,ya_t,y2a_t,height, x2,yytmp[j]);
	}
	spline(x1a,yytmp,width,1.0e30,1.0e30,ytmp);
	splint(x1a,yytmp,ytmp,width,x1,y);

	delete ya_t;
	ya_t=NULL;
	delete y2a_t;
	y2a_t=NULL;
	delete yytmp;
	yytmp=NULL;
	delete ytmp;
	ytmp=NULL;
}

