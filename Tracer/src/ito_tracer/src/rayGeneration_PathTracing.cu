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


#include <optix.h>
#include <optix_math.h>
#include "rayData.h"
#include "RayField.h"
#include "rayTracingMath.h"
#include "randomGenerator.h"
#include "time.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

rtDeclareVariable(rayFieldParams, params, , );

//rtDeclareVariable(double3,        origin_min, , );
//rtDeclareVariable(double3,        origin_max, , );
//rtDeclareVariable(double,        lambda, , );
//rtDeclareVariable(double,        flux, , );
//rtDeclareVariable(double,        nImmersed, , );
rtDeclareVariable(unsigned int,        launch_width, , );
rtDeclareVariable(unsigned int,        launch_height, , );
//rtDeclareVariable(unsigned int,        nrDirs, , );
//rtDeclareVariable(rayPosDistrType,        posDistrType, , );
//rtDeclareVariable(double3x3,        Mrot, , );
//rtDeclareVariable(double3,        translation, , );
//rtDeclareVariable(double2,        alphaMax, , );
//rtDeclareVariable(double2,        alphaMin, , );

//rtDeclareVariable(double3,        rayDir, , );

rtDeclareVariable(long,        launch_offsetX, , );
rtDeclareVariable(long,        launch_offsetY, , );

//rtDeclareVariable(unsigned long,	listlength, , );
//rtDeclareVariable(rayStruct*,	list, , );

rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float,         diff_epsilon, , );
rtBuffer<rayStruct_PathTracing, 1>              output_buffer;
//rtBuffer<double, 1>              xGrad_buffer;
//rtDeclareVariable(unsigned int,        size_xGrad, , );
//rtBuffer<double, 1>              yGrad_buffer;
//rtDeclareVariable(unsigned int,        size_yGrad, , );
rtBuffer<uint, 1>              seed_buffer;
rtDeclareVariable(rtObject,      top_object, , );
//rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim,   rtLaunchDim, );

//#define TIME_VIEW

/**********************************************************/
// device functions for distributing ray positions
/**********************************************************/

__forceinline__ __device__ void init_PathTracingRay_device(rayStruct_PathTracing &prd)
{
	prd.result=0;
	prd.secondary=false;
	prd.secondary_nr=1;
}

__forceinline__ __device__ void posDistr_RandRect_device(rayStruct &prd, rayFieldParams &params)
{
	// calc index
	unsigned long long iGes=launch_index+launch_offsetX+launch_offsetY*params.width*params.nrRayDirections.x*params.nrRayDirections.y;

	// calc position indices from 1D index
	unsigned long long iPosX=floorf(iGes/(params.nrRayDirections.x*params.nrRayDirections.y));
	unsigned long long iPosY=floorf(iPosX/params.width);
	iPosX=iPosX % params.width;

	uint32_t x1[5]; // variable for random generator
	unsigned int index=(iPosX+iPosY*params.width) % launch_dim;
	// init random variable
	RandomInit(seed_buffer[index], x1); // rays with same position index use same seed to create their position

	// width of ray field in physical dimension
	double physWidth=params.rayPosEnd.x-params.rayPosStart.x;
	// height of ray field in physical dimension
	double physHeight=params.rayPosEnd.y-params.rayPosStart.y;
	// calc centre of ray field 
	double2 rayFieldCentre=make_double2(params.rayPosStart.x+physWidth/2,params.rayPosStart.y+physHeight/2);

	prd.position.z=0;
	prd.position.x=(Random(x1)-0.5)*physWidth+rayFieldCentre.x;
	prd.position.y=(Random(x1)-0.5)*physHeight+rayFieldCentre.y;
	
	//prd.currentSeed=x1[4]; // don't set current seed as we used the same seed for rays that origin from the same position here...
};

__forceinline__ __device__ void posDistr_GridRect_device(rayStruct &prd, rayFieldParams &params)
{
	// calc index
	unsigned long long iGes=launch_index+launch_offsetX+launch_offsetY*params.width*params.nrRayDirections.x*params.nrRayDirections.y;

	// calc position indices from 1D index
	unsigned long long iPosX=floorf(iGes/(params.nrRayDirections.x*params.nrRayDirections.y));
	unsigned long long iPosY=floorf(iPosX/params.width);
	iPosX=iPosX % params.width;

	// width of ray field in physical dimension
	double physWidth=params.rayPosEnd.x-params.rayPosStart.x;
	// height of ray field in physical dimension
	double physHeight=params.rayPosEnd.y-params.rayPosStart.y;

	double deltaW=0;
	double deltaH=0;
	// calc increment along x- and y-direction
	if (params.width>0)
		deltaW= (physWidth)/(params.width);
	if (params.height>0)
		// multiple directions per point are listed in y-direction. Therefore the physical height of the rayfield is different from the height of the ray list. This has to be considered here...
		deltaH= (physHeight)/(params.height);
	prd.position.z=0;
	prd.position.x=params.rayPosStart.x+deltaW/2+iPosX*deltaW;
	prd.position.y=params.rayPosStart.y+deltaH/2+iPosY*deltaH;
};

__forceinline__ __device__ void posDistr_RandRad_device(rayStruct &prd, rayFieldParams &params)
{
	// calc index
	unsigned long long iGes=launch_index+launch_offsetX+launch_offsetY*params.width*params.nrRayDirections.x*params.nrRayDirections.y;

	// calc position indices from 1D index
	unsigned long long iPosX=floorf(iGes/(params.nrRayDirections.x*params.nrRayDirections.y));
	unsigned long long iPosY=floorf(iPosX/params.width);
	iPosX=iPosX % params.width;

	uint32_t x1[5]; // variable for random generator
	unsigned int index=(iPosX+iPosY*params.width) % launch_dim;
	// init random variable
	RandomInit(seed_buffer[index], x1); // rays with same position index use same seed to create their position

	// width of ray field in physical dimension
	double physWidth=params.rayPosEnd.x-params.rayPosStart.x;
	// height of ray field in physical dimension
	double physHeight=params.rayPosEnd.y-params.rayPosStart.y;

	// place a point uniformingly randomly inside the importance area
	double theta=2*PI*Random(x1);
	double r=sqrt(Random(x1));
	double ellipseX=physWidth/2*r*cos(theta);
	double ellipseY=physHeight/2*r*sin(theta);
	double3 exApt=make_double3(1,0,0);
	double3 eyApt=make_double3(0,1,0);
	prd.position=make_double3(0,0,0)+ellipseX*exApt+ellipseY*eyApt;

	//prd.currentSeed=x1[4]; // don't set current seed as we used the same seed for rays that origin from the same position here...
};

__forceinline__ __device__ void posDistr_GridRad_device(rayStruct &prd, rayFieldParams &params)
{
	// calc index
	unsigned long long iGes=launch_index+launch_offsetX+launch_offsetY*params.width*params.nrRayDirections.x*params.nrRayDirections.y;

	// calc position indices from 1D index
	unsigned long long iPosX=floorf(iGes/(params.nrRayDirections.x*params.nrRayDirections.y));
	unsigned long long iPosY=floorf(iPosX/params.width);
	iPosX=iPosX % params.width;

    // width of ray field in physical dimension
    double physWidth = params.rayPosEnd.x-params.rayPosStart.x;    
    // height of ray field in physical dimension
    double physHeight = params.rayPosEnd.y-params.rayPosStart.y;

	double deltaRx=0;
	double deltaRy=0;
	if (params.width>0)
	{
		deltaRx= (physWidth/2)/double(params.width);
		deltaRy= (physHeight/2)/double(params.width);
	}
	double deltaPhi=0;
	if (params.height>0)
		deltaPhi= (2*PI)/double(params.height);
	// calc r(phi) for given phi and radii of ellipse. see http://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center for reference
	double R=(deltaRx/2+deltaRx*iPosY)*(deltaRy/2+deltaRy*iPosY)/sqrt(pow((deltaRy/2+deltaRy*iPosY)*cos(deltaPhi/2+deltaPhi*iPosX),2)+pow((deltaRx/2+deltaRx*iPosY)*sin((deltaPhi/2+deltaPhi*iPosX)),2));
	// now calc rectangular coordinates from polar coordinates
	prd.position.z=0;
	prd.position.x=cos(deltaPhi/2+deltaPhi*iPosX)*R;
	prd.position.y=sin(deltaPhi/2+deltaPhi*iPosX)*R;
};

/**********************************************************/
// device functions for distributing ray directions
/**********************************************************/

__forceinline__ __device__ void dirDistr_Rand_device(rayStruct &prd, rayFieldParams &params)
{
	uint32_t x1[5]; // variable for random generator			
	
	RandomInit(prd.currentSeed, x1); // init random variable

	double3 rayAngleCentre=make_double3((params.alphaMax.x+params.alphaMin.x)/2,(params.alphaMax.y+params.alphaMin.y)/2,0);
	double2 rayAngleHalfWidth=make_double2((params.alphaMax.x-params.alphaMin.x)/2,(params.alphaMax.y-params.alphaMin.y)/2);
	// create random angles inside the given range
	double2 phi=make_double2(2*(Random(x1)-0.5)*rayAngleHalfWidth.x+rayAngleCentre.x,2*(Random(x1)-0.5)*rayAngleHalfWidth.y+rayAngleCentre.y);
	// create unit vector with the given angles
	prd.direction=createObliqueVec(phi);//normalize(make_double3(tan(phi.y),tan(phi.x),1));
	// transform raydirection into global coordinate system
	prd.direction=params.Mrot*prd.direction;

			
	// create points inside importance area to randomly distribute ray direction
//	double3 rayAngleCentre=make_double3((params.alphaMax.x+params.alphaMin.x)/2,(params.alphaMax.y+params.alphaMin.y)/2,0);
//	double2 impAreaHalfWidth;
//	impAreaHalfWidth.x=(tan(params.alphaMax.x)-tan(params.alphaMin.x))/2;
//	impAreaHalfWidth.y=(tan(params.alphaMax.y)-tan(params.alphaMin.y))/2;
//	double3 dirImpAreaCentre=make_double3(0,0,1);
//	rotateRay(&dirImpAreaCentre, rayAngleCentre);
	// the centre of the importance area is the root of the current geometry + the direction to the imp area centre normalized such that the importance area is 1mm away from the current geometry
//	double3 impAreaRoot=make_double3(0,0,0)+dirImpAreaCentre/dot(make_double3(0,0,1), dirImpAreaCentre);
	// now distribute points inside importance area
//	double theta=2*PI*Random(x1);
//	double r=sqrt(Random(x1));
//	double impAreaX=impAreaHalfWidth.x*r*cos(theta);
//	double impAreaY=impAreaHalfWidth.y*r*sin(theta);
//	double3 tmpPos=impAreaRoot+impAreaX*make_double3(1,0,0)+impAreaY*make_double3(0,1,0);
//	prd.direction=normalize(tmpPos-make_double3(0,0,0));
	// transform raydirection into global coordinate system
//	prd.direction=params.Mrot*prd.direction;

	// save seed for next randomization
	prd.currentSeed=x1[4];
};

__forceinline__ __device__ void dirDistr_RandImpArea_device(rayStruct &prd, rayFieldParams &params)
{
	uint32_t x1[5]; // variable for random generator			
	
	// declar variables for randomly distributing ray directions via an importance area
	double3 dirImpAreaCentre, tmpPos, impAreaRoot;
	
	RandomInit(prd.currentSeed, x1); // init random variable

	double impAreaX;
	double impAreaY;
			
	// now distribute points inside importance area

	if (params.importanceAreaApertureType==AT_RECT)
	{
		// place temporal point uniformingly randomly inside the importance area
		impAreaX=(Random(x1)-0.5)*2*params.importanceAreaHalfWidth.x;
		impAreaY=(Random(x1)-0.5)*2*params.importanceAreaHalfWidth.y; 
	}
	else 
	{
		if (params.importanceAreaApertureType==AT_ELLIPT)
		{
			double theta=2*PI*Random(x1);
			double r=sqrt(Random(x1));
			impAreaX=params.importanceAreaHalfWidth.x*r*cos(theta);
			impAreaY=params.importanceAreaHalfWidth.y*r*sin(theta);
		}
	}
	
	double3 impAreaAxisX=make_double3(1,0,0);
	double3 impAreaAxisY=make_double3(0,1,0);
		
	rotateRay(&impAreaAxisX,params.importanceAreaTilt);
	rotateRay(&impAreaAxisY,params.importanceAreaTilt);

	tmpPos=params.importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
	prd.direction=normalize(tmpPos-prd.position);
	// save seed for next randomization
	prd.currentSeed=x1[4];
};

__forceinline__ __device__ void dirDistr_Uniform_device(rayStruct &prd, rayFieldParams &params)
{
	prd.direction=params.rayDirection;
};

__forceinline__ __device__ void dirDistr_GridRect_device(rayStruct &prd, rayFieldParams &params)
{
	// calc index
	unsigned long long iGes=launch_index+launch_offsetX+launch_offsetY*params.width*params.nrRayDirections.x*params.nrRayDirections.y;

	// calc position indices from 1D index
	unsigned long long iPosX=floorf(iGes/(params.nrRayDirections.x*params.nrRayDirections.y));
	unsigned long long iPosY=floorf(iPosX/params.width);
	iPosX=iPosX % params.width;

	// calc direction indices from 1D index
	unsigned long long iDirX=(iGes-iPosX*params.nrRayDirections.x*params.nrRayDirections.y-iPosY*params.nrRayDirections.x*params.nrRayDirections.y*params.width) % params.nrRayDirections.x;
	unsigned long long iDirY=floorf((iGes-iPosX*params.nrRayDirections.x*params.nrRayDirections.y-iPosY*params.nrRayDirections.x*params.nrRayDirections.y*params.width)/params.nrRayDirections.x);

    double r; // variable for generating random variables inside an ellipse
	// declar variables for randomly distributing ray directions via an importance area
	double2 impAreaHalfWidth;
	double3 dirImpAreaCentre, tmpPos, impAreaRoot, impAreaAxisX, impAreaAxisY;
	double impAreaX, impAreaY, theta;
	// increment of temporary raypos in x and y 
	double deltaW=0;
	double deltaH=0;

	// calc increment along x- and y-direction
	if (params.nrRayDirections.x>0)
		deltaW= (2*params.importanceAreaHalfWidth.x)/(params.nrRayDirections.x);
	if (params.nrRayDirections.y>0)
		deltaH= (2*params.importanceAreaHalfWidth.y)/(params.nrRayDirections.y);
	impAreaX=-params.importanceAreaHalfWidth.x+deltaW/2+iDirX*deltaW; 
	impAreaY=-params.importanceAreaHalfWidth.y+deltaH/2+iDirY*deltaH; 
	impAreaAxisX=make_double3(1,0,0);
	impAreaAxisY=make_double3(0,1,0);
	rotateRay(&impAreaAxisX,params.importanceAreaTilt);
	rotateRay(&impAreaAxisY,params.importanceAreaTilt);

	tmpPos=params.importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
	prd.direction=normalize(tmpPos-prd.position);
};

__forceinline__ __device__ void dirDistr_GridRad_device(rayStruct &prd, rayFieldParams &params)
{
	// calc index
	unsigned long long iGes=launch_index+launch_offsetX+launch_offsetY*params.width*params.nrRayDirections.x*params.nrRayDirections.y;

	// calc position indices from 1D index
	unsigned long long iPosX=floorf(iGes/(params.nrRayDirections.x*params.nrRayDirections.y));
	unsigned long long iPosY=floorf(iPosX/params.width);
	iPosX=iPosX % params.width;

	// calc direction indices from 1D index
	unsigned long long iDirX=(iGes-iPosX*params.nrRayDirections.x*params.nrRayDirections.y-iPosY*params.nrRayDirections.x*params.nrRayDirections.y*params.width) % params.nrRayDirections.x;
	unsigned long long iDirY=floorf((iGes-iPosX*params.nrRayDirections.x*params.nrRayDirections.y-iPosY*params.nrRayDirections.x*params.nrRayDirections.y*params.width)/params.nrRayDirections.x);

    double r; // variable for generating random variables inside an ellipse
	// declar variables for randomly distributing ray directions via an importance area
	double2 impAreaHalfWidth;
	double3 dirImpAreaCentre, tmpPos, impAreaRoot, impAreaAxisX, impAreaAxisY;
	double impAreaX, impAreaY, theta, deltaPhi;
	// increment of temporary raypos in x and y 
	double deltaRx=0;
	double deltaRy=0;
	
	// calc increment along radial and angular direction
	if (params.nrRayDirections.x>0)
	{
		deltaRx= (params.importanceAreaHalfWidth.x)/double(params.nrRayDirections.x);
		deltaRy= (params.importanceAreaHalfWidth.y)/double(params.nrRayDirections.x);
	}
	if (params.nrRayDirections.y>0)
		deltaPhi= (2*PI)/params.nrRayDirections.y;
	// calc r(phi) for given phi and radii of ellipse. see http://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center for reference
	double R=(deltaRx/2+deltaRx*iDirY)*(deltaRy/2+deltaRy*iDirY)/sqrt(pow((deltaRy/2+deltaRy*iDirY)*cos(deltaPhi/2+deltaPhi*iDirX),2)+pow((deltaRx/2+deltaRx*iDirY)*sin(deltaPhi/2+deltaPhi*iDirX),2));
	if (deltaRy==0)
		R=0;
	// now calc rectangular coordinates from polar coordinates
	impAreaX=cos(deltaPhi*iDirX)*R;
	impAreaY=sin(deltaPhi*iDirX)*R;
	
	impAreaAxisX=make_double3(1,0,0);
	impAreaAxisY=make_double3(0,1,0);
	rotateRay(&impAreaAxisX,params.importanceAreaTilt);
	rotateRay(&impAreaAxisY,params.importanceAreaTilt);

	tmpPos=params.importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
	prd.direction=normalize(tmpPos-prd.position);
};

RT_PROGRAM void rayGeneration()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;
  prd.result=0;
  prd.secondary=false;
  prd.secondary_nr=1;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

 			switch (params.posDistrType)
			{
				case RAYPOS_GRID_RECT:
					posDistr_GridRect_device(prd, params);
					break;
				case RAYPOS_RAND_RECT:	
					posDistr_RandRect_device(prd, params);
					break;
				case RAYPOS_GRID_RAD:
					posDistr_GridRad_device(prd, params);
					break;
				case RAYPOS_RAND_RAD:
					posDistr_RandRad_device(prd, params);
					break;
				default:
					rtPrintf("RAYPOS_DEFAULT");
					prd.position=make_double3(0,0,0);
					// report error
					break;
			}
			// transform rayposition into global coordinate system
			prd.position=params.Mrot*prd.position+params.translation;

			switch (params.dirDistrType)
			{
				case RAYDIR_UNIFORM:
					dirDistr_Uniform_device(prd, params);
					break;
				case RAYDIR_RAND_RECT:
				    dirDistr_Rand_device(prd, params);
					break;
				case RAYDIR_GRID_RECT:
					dirDistr_GridRect_device(prd, params);
					break;
				case RAYDIR_GRID_RAD:
					dirDistr_GridRad_device(prd, params);
					break;
				default:
					prd.direction=make_double3(0,0,0);
					// report error
					break;
			}

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 

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

/*****************************************************************
/   DirRandImpArea
/*****************************************************************/

RT_PROGRAM void rayGeneration_DirRandImpArea_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRad_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_RandImpArea_device(prd, params);
	
	init_PathTracingRay_device(prd);


  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 

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

RT_PROGRAM void rayGeneration_DirRandImpArea_PosGridRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_GridRad_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_RandImpArea_device(prd, params);
	
	init_PathTracingRay_device(prd);
	
  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;

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


RT_PROGRAM void rayGeneration_DirRandImpArea_PosRandRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;
  
  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRect_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_RandImpArea_device(prd, params);
	
	init_PathTracingRay_device(prd);

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 
  

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

RT_PROGRAM void rayGeneration_DirRandImpArea_PosGridRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_GridRect_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_RandImpArea_device(prd, params);
	
	init_PathTracingRay_device(prd);

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;

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

/**********************************************************************
/   DirRand
/**********************************************************************/

RT_PROGRAM void rayGeneration_DirRand_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRad_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_Rand_device(prd, params);
	
	init_PathTracingRay_device(prd);


  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 

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

RT_PROGRAM void rayGeneration_DirRand_PosGridRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_GridRad_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_Rand_device(prd, params);
	
	init_PathTracingRay_device(prd);
	
  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;

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


RT_PROGRAM void rayGeneration_DirRand_PosRandRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;
  
  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRect_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_Rand_device(prd, params);
	
	init_PathTracingRay_device(prd);

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 
  

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

RT_PROGRAM void rayGeneration_DirRand_PosGridRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_GridRect_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_Rand_device(prd, params);
	
	init_PathTracingRay_device(prd);

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;

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

/********************************************************************************************/
//                 Dir_GridRad
/********************************************************************************************/

RT_PROGRAM void rayGeneration_DirGridRad_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRad_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_GridRad_device(prd, params);
	
	init_PathTracingRay_device(prd);


  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 

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

RT_PROGRAM void rayGeneration_DirGridRad_PosGridRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_GridRad_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_GridRad_device(prd, params);
	
	init_PathTracingRay_device(prd);
	
  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;

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


RT_PROGRAM void rayGeneration_DirGridRad_PosRandRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;
  
  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRect_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_GridRad_device(prd, params);
	
	init_PathTracingRay_device(prd);

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 
  

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

RT_PROGRAM void rayGeneration_DirGridRad_PosGridRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_GridRect_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_GridRad_device(prd, params);
	
	init_PathTracingRay_device(prd);

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;

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

/**********************************************************************************************/
//                    DirGridRect
/**********************************************************************************************/

RT_PROGRAM void rayGeneration_DirGridRect_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRad_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_GridRect_device(prd, params);
	
	init_PathTracingRay_device(prd);


  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 

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

RT_PROGRAM void rayGeneration_DirGridRect_PosGridRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_GridRad_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_GridRect_device(prd, params);
	
	init_PathTracingRay_device(prd);
	
  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;

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


RT_PROGRAM void rayGeneration_DirGridRect_PosRandRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;
  
  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRect_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_GridRect_device(prd, params);
	
	init_PathTracingRay_device(prd);

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;
  prd.opl=0; 
  

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

RT_PROGRAM void rayGeneration_DirGridRect_PosGridRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  float3 ray_origin = make_float3(0.0f);
  float3 ray_direction = make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
  
  rayStruct_PathTracing prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_GridRect_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_GridRect_device(prd, params);
	
	init_PathTracingRay_device(prd);

  prd.currentGeometryID = 0;
  prd.lambda=params.lambda;
  prd.nImmersed=params.nImmersed;
  prd.running=true;

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
       break;
    }
  }
  //prd.position=make_double3(0,0,100);
  output_buffer[launch_index] = prd;//.direction;//prd.position;
}


RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d)\n", code, (launch_index+launch_offsetX) );
//  output_buffer[launch_index] = prd.position;
}
