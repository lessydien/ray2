/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix.h>
#include <optix_math.h>
//#include "helpers.h"
#include "rayData.h"
#include "wavefrontIn.h"
#include "randomGenerator.h"
#include "time.h"

#ifndef PI
	#define PI 3.14159265358979323846
#endif

rtDeclareVariable(double3,        origin_min, , );
rtDeclareVariable(double3,        origin_max, , );
rtDeclareVariable(double,        lambda, , );
rtDeclareVariable(double,        flux, , );
rtDeclareVariable(double,        nImmersed, , );
rtDeclareVariable(unsigned int,        launch_width, , );
rtDeclareVariable(unsigned int,        launch_height, , );
rtDeclareVariable(unsigned int,        nrDirs, , );
rtDeclareVariable(rayPosDistrType,        posDistrType, , );
rtDeclareVariable(double3x3,        Mrot, , );
rtDeclareVariable(double3,        translation, , );
rtDeclareVariable(double2,        alphaMax, , );
rtDeclareVariable(double2,        alphaMin, , );

rtDeclareVariable(double3,        rayDir, , );

rtDeclareVariable(long,        launch_offsetX, , );
rtDeclareVariable(long,        launch_offsetY, , );

//rtDeclareVariable(unsigned long,	listlength, , );
//rtDeclareVariable(rayStruct*,	list, , );

rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float,         diff_epsilon, , );
rtBuffer<diffRayStruct, 2>              output_buffer;
rtBuffer<double, 1>              xGrad_buffer;
rtDeclareVariable(unsigned int,        size_xGrad, , );
rtBuffer<double, 1>              yGrad_buffer;
rtDeclareVariable(unsigned int,        size_yGrad, , );
rtBuffer<uint, 2>              seed_buffer;
rtDeclareVariable(rtObject,      top_object, , );
//rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

//#define TIME_VIEW

RT_PROGRAM void rayGeneration()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  diffRayStruct prd;
  prd.flux = flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

  // width of ray field in physical dimension
  double physWidth = origin_max.x-origin_min.x;    

  // height of ray field in physical dimension
  double physHeight = origin_max.y-origin_min.y;
  
  double r; // variable for generating random variables inside an ellipse
  long index=0; // loop counter for random rejection method
 
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
  double2 rayFieldCentre=make_double2(origin_min.x+physWidth/2,origin_min.y+physHeight/2);

	uint32_t x1[5]; // variable for random generator			
	//int seed = (int)clock();            // random seed
	RandomInit(prd.currentSeed, x1);
	//for ( unsigned int i = 0; i < launch_index.x+(launch_index.y*launch_dim.x); ++i )
	//	prd.currentSeed = (uint)BRandom(x1);	


	// calc centre angle of opening cone
	double3 rayAngleCentre=make_double3((alphaMax.x+alphaMin.x)/2,(alphaMax.y+alphaMin.y)/2,0);
	double angleWidth=(alphaMax.x-alphaMin.x);
	double angleHeight=(alphaMax.y-alphaMin.y);

	double3 alpha=make_double3(0,0,0); // rotation angle of raydirection around x and y

					RandomInit(prd.currentSeed, x1); // init random variable
					// create random angle until it is inside the ellipse
					do
					{
						alpha.x=(Random(x1)-0.5)*angleWidth;
						alpha.y=(Random(x1)-0.5)*angleHeight;
						r=alpha.x*alpha.x/(angleWidth*angleWidth/4)+alpha.y*alpha.y/(angleHeight*angleHeight/4);
						index++;
						if (index>1000000)
							break;
					} while ( (r >= 1.0) );
					alpha=alpha+rayAngleCentre;
					prd.direction=make_double3(0,0,1); // init direction along z-direction
					rotateRay(&prd.direction, alpha); // rotate ray according to the angle

			// transform raydirection into global coordinate system
			prd.direction=Mrot*prd.direction;

			prd.position.z=0;
			switch (posDistrType)
			{
				case RAYPOS_GRID_RECT:
					// calc increment along x- and y-direction
					if (launch_width>1)
						deltaW= (physWidth)/double(launch_width-1);
					if (launch_height>1)
						deltaH= (physHeight)/double(launch_height-1);
					prd.position.x=origin_min.x+(launch_index.x+launch_offsetX)*deltaW;
					prd.position.y=origin_min.y+(launch_index.y+launch_offsetY)*deltaH;
					break;
				case RAYPOS_RAND_RECT:	
					RandomInit(prd.currentSeed, x1); // init random variable
					prd.position.x=(Random(x1)-0.5)*physWidth+rayFieldCentre.x;
					prd.position.y=(Random(x1)-0.5)*physHeight+rayFieldCentre.y;
					break;
				case RAYPOS_GRID_RAD:
					// calc increment along radial and angular direction
					if (launch_width>1)
					{
						deltaRx= (physWidth/2)/double(launch_width-1);
						deltaRy= (physHeight/2)/double(launch_width-1);
					}
					if (launch_height>1)
						deltaPhi= (2*PI)/double(launch_height-1);
					if ((launch_index.x+launch_offsetX)==0 || deltaRx==0 || deltaRy==0)
						prd.position=make_double3(0,0,0);
					else
					{
						// calc r(phi) for given phi and radii of ellipse. see http://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center for reference
						R=deltaRx*(launch_index.x+launch_offsetX)*deltaRy*(launch_index.x+launch_offsetX)/sqrt(pow(deltaRy*(launch_index.x+launch_offsetX)*cos(deltaPhi*(launch_index.y+launch_offsetY)),2)+pow(deltaRx*(launch_index.x+launch_offsetX)*sin(deltaPhi*(launch_index.y+launch_offsetY)),2));
						// now calc rectangular coordinates from polar coordinates
						prd.position.x=cos(deltaPhi*(launch_index.y+launch_offsetY))*R;
						prd.position.y=sin(deltaPhi*(launch_index.y+launch_offsetY))*R;
					}
					break;
				case RAYPOS_RAND_RAD:
					RandomInit(prd.currentSeed, x1); // init random variable
					// create random position until it is inside the ellipse
					do
					{
						prd.position.x=(Random(x1)-0.5)*physWidth;
						prd.position.y=(Random(x1)-0.5)*physHeight;
						r=prd.position.x*prd.position.x/(physWidth*physWidth/4)+prd.position.y*prd.position.y/(physHeight*physHeight/4);
						index++;
						if (index>1000000)
							break;
					} while ( (r >= 1.0) );

					break;
				default:
					rtPrintf("RAYPOS_DEFAULT");
					prd.position=make_double3(0,0,0);
					// report error
					break;
			}
			// transform rayposition into global coordinate system
			prd.position=Mrot*prd.position+translation;


  prd.currentGeometryID = 0;
  prd.lambda=lambda;
  prd.nImmersed=nImmersed;
  prd.running=true;
  prd.opl=0; 
  prd.currentSeed=x1[0];

#ifdef TIME_VIEW
  t1 = clock(); 
  time = (double)(t1-t0);
  rtPrintf("time elapsed while creating ray: %lf ms\n", time);
#endif 
 
  for(;;) 
  {
    //optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(top_object, ray, prd);
    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  diffRayStruct prd;
  prd.flux = flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

  // width of ray field in physical dimension
  double physWidth = origin_max.x-origin_min.x;    

  // height of ray field in physical dimension
  double physHeight = origin_max.y-origin_min.y;
  
  double r; // variable for generating random variables inside an ellipse
  long index=0; // loop counter for random rejection method
 
  // calc centre of ray field 
  double2 rayFieldCentre=make_double2(origin_min.x+physWidth/2,origin_min.y+physHeight/2);

	// calc centre angle of opening cone
	double3 rayAngleCentre=make_double3((alphaMax.x+alphaMin.x)/2,(alphaMax.y+alphaMin.y)/2,0);
	double angleWidth=(alphaMax.x-alphaMin.x);
	double angleHeight=(alphaMax.y-alphaMin.y);

	double3 alpha=make_double3(0,0,0); // rotation angle of raydirection around x and y

//	case RAYDIR_RAND:
	uint32_t x1[5]; // variable for random generator			
	//int seed = (int)clock();            // random seed
	RandomInit(prd.currentSeed, x1);
	//for ( unsigned int i = 0; i < launch_index.x+(launch_index.y*launch_dim.x); ++i )
	//	prd.currentSeed = (uint)BRandom(x1);	

	// create random angle until it is inside the ellipse
	do
	{
		alpha.x=(Random(x1)-0.5)*angleWidth;
		alpha.y=(Random(x1)-0.5)*angleHeight;
		r=alpha.x*alpha.x/(angleWidth*angleWidth/4)+alpha.y*alpha.y/(angleHeight*angleHeight/4);
		index++;
		if (index>1000000)
			break;
	} while ( (r >= 1.0) );
	alpha=alpha+rayAngleCentre;
	prd.direction=make_double3(0,0,1); // init direction along z-direction
	rotateRay(&prd.direction, alpha); // rotate ray according to the angle

	// transform raydirection into global coordinate system
	prd.direction=Mrot*prd.direction;

	prd.position.z=0;
//	case RAYPOS_RAND_RAD:
		
	// create random position until it is inside the ellipse
	do
	{
		prd.position.x=(Random(x1)-0.5)*physWidth;
		prd.position.y=(Random(x1)-0.5)*physHeight;
		r=prd.position.x*prd.position.x/(physWidth*physWidth/4)+prd.position.y*prd.position.y/(physHeight*physHeight/4);
		index++;
		if (index>1000000)
			break;
	} while ( (r >= 1.0) );
	// transform rayposition into global coordinate system
	prd.position=Mrot*prd.position+translation;


  prd.currentGeometryID = 0;
  prd.lambda=lambda;
  prd.nImmersed=nImmersed;
  prd.running=true;
  prd.opl=0; 
  prd.currentSeed=x1[0];

#ifdef TIME_VIEW
  t1 = clock(); 
  time = (double)(t1-t0);
  rtPrintf("time elapsed while creating ray: %lf ms\n", time);
#endif 
 
  for(;;) 
  {
    //optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(top_object, ray, prd);
    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_PosGridRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  diffRayStruct prd;
  prd.flux = flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

  // width of ray field in physical dimension
  double physWidth = origin_max.x-origin_min.x;    

  // height of ray field in physical dimension
  double physHeight = origin_max.y-origin_min.y;
  
  double r; // variable for generating random variables inside an ellipse
  long index=0; // loop counter for random rejection method
 
  // increment radial ( along x- and y ) and angular direction in GridRad definition
  double deltaRx=0;
  double deltaRy=0;
  double deltaPhi=0;
  // radius in dependence of phi when calculating GRID_RAD
  double R=0;
  // calc centre of ray field 
  double2 rayFieldCentre=make_double2(origin_min.x+physWidth/2,origin_min.y+physHeight/2);

	// calc centre angle of opening cone
	double3 rayAngleCentre=make_double3((alphaMax.x+alphaMin.x)/2,(alphaMax.y+alphaMin.y)/2,0);
	double angleWidth=(alphaMax.x-alphaMin.x);
	double angleHeight=(alphaMax.y-alphaMin.y);

	double3 alpha=make_double3(0,0,0); // rotation angle of raydirection around x and y

//	case RAYDIR_RAND:
	uint32_t x1[5]; // variable for random generator			
	//int seed = (int)clock();            // random seed
	RandomInit(prd.currentSeed, x1);
	//for ( unsigned int i = 0; i < launch_index.x+(launch_index.y*launch_dim.x); ++i )
	//	prd.currentSeed = (uint)BRandom(x1);	

	for ( unsigned int i = 0; i < launch_index.x+(launch_index.y*launch_dim.x); ++i )
		prd.currentSeed = (uint)BRandom(x1);	// create random angle until it is inside the ellipse
		
	do
	{
		alpha.x=(Random(x1)-0.5)*angleWidth;
		alpha.y=(Random(x1)-0.5)*angleHeight;
		r=alpha.x*alpha.x/(angleWidth*angleWidth/4)+alpha.y*alpha.y/(angleHeight*angleHeight/4);
		index++;
		if (index>1000000)
			break;
	} while ( (r >= 1.0) );
	alpha=alpha+rayAngleCentre;
	prd.direction=make_double3(0,0,1); // init direction along z-direction
	rotateRay(&prd.direction, alpha); // rotate ray according to the angle
	// transform raydirection into global coordinate system
	prd.direction=Mrot*prd.direction;

	prd.position.z=0;
//	case RAYPOS_GRID_RAD:
	// calc increment along radial and angular direction
	if (launch_width>1)
	{
		deltaRx= (physWidth/2)/double(launch_width-1);
		deltaRy= (physHeight/2)/double(launch_width-1);
	}
	if (launch_height>1)
		deltaPhi= (2*PI)/double(launch_height/nrDirs-1);
	if ((launch_index.x+launch_offsetX)==0 || deltaRx==0 || deltaRy==0)
		prd.position=make_double3(0,0,0);
	else
	{
		// calc r(phi) for given phi and radii of ellipse. see http://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center for reference
		R=deltaRx*(launch_index.x+launch_offsetX)*deltaRy*(launch_index.x+launch_offsetX)/sqrt(pow(deltaRy*(launch_index.x+launch_offsetX)*cos(deltaPhi*(launch_index.y+launch_offsetY)),2)+pow(deltaRx*(launch_index.x+launch_offsetX)*sin(deltaPhi*(launch_index.y+launch_offsetY)),2));
		// now calc rectangular coordinates from polar coordinates
		prd.position.x=cos(deltaPhi*floorf((launch_index.y+launch_offsetY)/nrDirs))*R;
		prd.position.y=sin(deltaPhi*floorf((launch_index.y+launch_offsetY)/nrDirs))*R;
	}
	// transform rayposition into global coordinate system
	prd.position=Mrot*prd.position+translation;

	// move ray a short distance out of the caustic
	prd.wavefrontRad=make_double2(-diff_epsilon,-diff_epsilon);
	prd.mainDirX=make_double3(1,0,0);
	prd.mainDirY=make_double3(0,1,0);
	prd.opl=diff_epsilon;
	prd.position=prd.position+diff_epsilon*prd.direction;

  prd.currentGeometryID = 0;
  prd.lambda=lambda;
  prd.nImmersed=nImmersed;
  prd.running=true;
  prd.currentSeed=x1[0];

#ifdef TIME_VIEW
  t1 = clock(); 
  time = (double)(t1-t0);
  rtPrintf("time elapsed while creating ray: %lf ms\n", time);
#endif 
 
  for(;;) 
  {
    //optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(top_object, ray, prd);
    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}


RT_PROGRAM void rayGeneration_PosRandRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  float3 ray_origin = make_float3(0.0f);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  diffRayStruct prd;
  prd.flux = flux;
  prd.depth = 0;
  
  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

  // width of ray field in physical dimension
  double physWidth = origin_max.x-origin_min.x;    

  // height of ray field in physical dimension
  double physHeight = origin_max.y-origin_min.y;
  
  double r; // variable for generating random variables inside an ellipse
  long index=0; // loop counter for random rejection method
 
  // calc centre of ray field 
  double2 rayFieldCentre=make_double2(origin_min.x+physWidth/2,origin_min.y+physHeight/2);

	// calc centre angle of opening cone
	double3 rayAngleCentre=make_double3((alphaMax.x+alphaMin.x)/2,(alphaMax.y+alphaMin.y)/2,0);
	double angleWidth=(alphaMax.x-alphaMin.x);
	double angleHeight=(alphaMax.y-alphaMin.y);

	double3 alpha=make_double3(0,0,0); // rotation angle of raydirection around x and y

//	case RAYDIR_RAND:
	uint32_t x1[5]; // variable for random generator			
	//int seed = (int)clock();            // random seed
	RandomInit(prd.currentSeed, x1);
	//for ( unsigned int i = 0; i < launch_index.x+(launch_index.y*launch_dim.x); ++i )
	//	prd.currentSeed = (uint)BRandom(x1);	
		
	// create random angle until it is inside the ellipse
	do
	{
		alpha.x=(Random(x1)-0.5)*angleWidth;
		alpha.y=(Random(x1)-0.5)*angleHeight;
		r=alpha.x*alpha.x/(angleWidth*angleWidth/4)+alpha.y*alpha.y/(angleHeight*angleHeight/4);
		index++;
		if (index>1000000)
			break;
	} while ( (r >= 1.0) );
	alpha=alpha+rayAngleCentre;
	prd.direction=make_double3(0,0,1); // init direction along z-direction
	rotateRay(&prd.direction, alpha); // rotate ray according to the angle

	// transform raydirection into global coordinate system
	prd.direction=Mrot*prd.direction;

	prd.position.z=0;
//	case RAYPOS_RAND_RECT:	
	prd.position.x=(Random(x1)-0.5)*physWidth+rayFieldCentre.x;
	prd.position.y=(Random(x1)-0.5)*physHeight+rayFieldCentre.y;
	// transform rayposition into global coordinate system
	prd.position=Mrot*prd.position+translation;
	
	prd.currentSeed=x1[0];


  prd.currentGeometryID = 0;
  prd.lambda=lambda;
  prd.nImmersed=nImmersed;
  prd.running=true;
  prd.opl=0; 
  prd.currentSeed=x1[0];
  

#ifdef TIME_VIEW
  t1 = clock(); 
  time = (double)(t1-t0);
  rtPrintf("time elapsed while creating ray: %lf ms\n", time);
#endif 
 
  for(;;) 
  {
    rtTrace(top_object, ray, prd);
    if(!prd.running) 
    {
       break;
    }
  }
  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_PosGridRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  diffRayStruct prd;
  prd.flux = flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

  // width of ray field in physical dimension
  double physWidth = origin_max.x-origin_min.x;    

  // height of ray field in physical dimension
  double physHeight = origin_max.y-origin_min.y;
  
  double r; // variable for generating random variables inside an ellipse
  long index=0; // loop counter for random rejection method
 
  // increment of rayposition in x and y in case of GridRect definition 
  double deltaW=0;
  double deltaH=0;
  // calc centre of ray field 
  double2 rayFieldCentre=make_double2(origin_min.x+physWidth/2,origin_min.y+physHeight/2);

	// calc centre angle of opening cone
	double3 rayAngleCentre=make_double3((alphaMax.x+alphaMin.x)/2,(alphaMax.y+alphaMin.y)/2,0);
	double angleWidth=(alphaMax.x-alphaMin.x);
	double angleHeight=(alphaMax.y-alphaMin.y);

	double3 alpha=make_double3(0,0,0); // rotation angle of raydirection around x and y

	uint32_t x1[5]; // variable for random generator			
	//int seed = (int)clock();            // random seed
	RandomInit(prd.currentSeed, x1);
	//for ( unsigned int i = 0; i < launch_index.x+(launch_index.y*launch_dim.x); ++i )
	//	prd.currentSeed = (uint)BRandom(x1);	
		
	// create random angle until it is inside the ellipse
	do
	{
		alpha.x=(Random(x1)-0.5)*angleWidth;
		alpha.y=(Random(x1)-0.5)*angleHeight;
		r=alpha.x*alpha.x/(angleWidth*angleWidth/4)+alpha.y*alpha.y/(angleHeight*angleHeight/4);
		index++;
		if (index>1000000)
			break;
	} while ( (r >= 1.0) );
	alpha=alpha+rayAngleCentre;
	prd.direction=make_double3(0,0,1); // init direction along z-direction
	rotateRay(&prd.direction, alpha); // rotate ray according to the angle
	// transform raydirection into global coordinate system
	prd.direction=Mrot*prd.direction;
	
	prd.position.z=0;
	// calc increment along x- and y-direction
	if (launch_width>1)
		deltaW= (physWidth)/double(launch_width-1);
	if (launch_height>1)
		deltaH= (physHeight)/double(launch_height/nrDirs-1);
	prd.position.x=origin_min.x+(launch_index.x+launch_offsetX)*deltaW;
	prd.position.y=origin_min.y+floorf((launch_index.y+launch_offsetY)/nrDirs)*deltaH;

	// transform rayposition into global coordinate system
	prd.position=Mrot*prd.position+translation;

	// move ray a short distance out of the caustic
	prd.wavefrontRad=make_double2(-diff_epsilon,-diff_epsilon);
	prd.mainDirX=make_double3(1,0,0);
	prd.mainDirY=make_double3(0,1,0);
	prd.opl=diff_epsilon;
	prd.position=prd.position+diff_epsilon*prd.direction;


  prd.currentGeometryID = 0;
  prd.lambda=lambda;
  prd.nImmersed=nImmersed;
  prd.running=true;
  prd.currentSeed=x1[0];

#ifdef TIME_VIEW
  t1 = clock(); 
  time = (double)(t1-t0);
  rtPrintf("time elapsed while creating ray: %lf ms\n", time);
#endif 
 
  for(;;) 
  {
    //optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(top_object, ray, prd);
    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }
  //prd.position=make_double3(0,0,100);
  output_buffer[launch_index] = prd;//.direction;//prd.position;
}


RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, (launch_index.x+launch_offsetX), (launch_index.y+launch_offsetY) );
//  output_buffer[launch_index] = prd.position;
}
