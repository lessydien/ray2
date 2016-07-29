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
#include "randomGenerator.h"
#include "rayTracingMath.h"
#include "time.h"
#include "RayField.h"

#ifndef PI
#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

rtDeclareVariable(rayFieldParams, params, , );

rtDeclareVariable(long long,        launch_offsetX, , );
rtDeclareVariable(long long,        launch_offsetY, , );

rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<rayStruct, 2>              output_buffer;
rtBuffer<uint, 2>              seed_buffer;
rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

//#define TIME_VIEW

/**********************************************************/
// device functions for distributing ray positions
/**********************************************************/

__forceinline__ __device__ void posDistr_RandRect_device(rayStruct &prd, rayFieldParams &params)
{
	uint32_t x1[5]; // variable for random generator			
	RandomInit(prd.currentSeed, x1); // init random variable
    // width of ray field in physical dimension
    double physWidth = params.rayPosEnd.x-params.rayPosStart.x;    
    // height of ray field in physical dimension
    double physHeight = params.rayPosEnd.y-params.rayPosStart.y;

    // calc centre of ray field 
    double2 rayFieldCentre=make_double2(params.rayPosStart.x+physWidth/2,params.rayPosStart.y+physHeight/2);

	prd.position.z=0;
	prd.position.x=(Random(x1)-0.5)*physWidth+rayFieldCentre.x;
	prd.position.y=(Random(x1)-0.5)*physHeight+rayFieldCentre.y;
	// save seed for next randomization
	prd.currentSeed=x1[4];	
};

__forceinline__ __device__ void posDistr_RandRectNorm_device(rayStruct &prd, rayFieldParams &params)
{
	uint32_t x1[5]; // variable for random generator			
	RandomInit(prd.currentSeed, x1); // init random variable
    // width of ray field in physical dimension
    double physWidth = params.rayPosEnd.x-params.rayPosStart.x;    
    // height of ray field in physical dimension
    double physHeight = params.rayPosEnd.y-params.rayPosStart.y;

    // calc centre of ray field 
    double2 rayFieldCentre=make_double2(params.rayPosStart.x+physWidth/2,params.rayPosStart.y+physHeight/2);

	prd.position.z=0;
	prd.position.x=RandomGauss(x1)*physWidth/2+rayFieldCentre.x;
	prd.position.y=RandomGauss(x1)*physHeight/2+rayFieldCentre.y;

	// save seed for next randomization
	prd.currentSeed=x1[4];	
};

__forceinline__ __device__ void posDistr_GridRect_device(rayStruct &prd, rayFieldParams &params)
{
    // width of ray field in physical dimension
    double physWidth = params.rayPosEnd.x-params.rayPosStart.x;    
    // height of ray field in physical dimension
    double physHeight = params.rayPosEnd.y-params.rayPosStart.y;

	// calc increment along x- and y-direction
	double deltaW=0;
	double deltaH=0;
	if (params.width>0)
		deltaW= (physWidth)/double(params.width);
	if (params.height>0)
		deltaH= (physHeight)/double(params.height);
	prd.position.z=0;		
	prd.position.x=params.rayPosStart.x+deltaW/2+(launch_index.x+launch_offsetX)*deltaW;
	prd.position.y=params.rayPosStart.y+deltaH/2+(launch_index.y+launch_offsetY)*deltaH;
};

__forceinline__ __device__ void posDistr_RandRad_device(rayStruct &prd, rayFieldParams &params)
{
	uint32_t x1[5]; // variable for random generator			
	RandomInit(prd.currentSeed, x1); // init random variable

	// place a point uniformingly randomly inside the importance area
	double theta=2*PI*Random(x1);
	double r=sqrt(Random(x1));

    // width of ray field in physical dimension
    double physWidth = params.rayPosEnd.x-params.rayPosStart.x;    
    // height of ray field in physical dimension
    double physHeight = params.rayPosEnd.y-params.rayPosStart.y;
	double ellipseX=physWidth/2*r*cos(theta);
	double ellipseY=physHeight/2*r*sin(theta);
	double3 exApt=make_double3(1,0,0);
	double3 eyApt=make_double3(0,1,0);
	prd.position=make_double3(0,0,0)+ellipseX*exApt+ellipseY*eyApt;
	// save seed for next randomization
	prd.currentSeed=x1[4];	
};

__forceinline__ __device__ void posDistr_GridRad_device(rayStruct &prd, rayFieldParams &params)
{
    // width of ray field in physical dimension
    double physWidth = params.rayPosEnd.x-params.rayPosStart.x;    
    // height of ray field in physical dimension
    double physHeight = params.rayPosEnd.y-params.rayPosStart.y;
    
    unsigned long long iPosX=launch_index.x+launch_offsetX;
    unsigned long long iPosY=launch_index.y+launch_offsetY;

	double deltaRx=0;
	double deltaRy=0;
	if (params.width>0)
	{
		deltaRx= (physWidth/2)/double(params.width);
		deltaRy= (physHeight/2)/double(params.width);
	}
	double deltaPhi=0;
	double R;
	if (params.height>0)
		deltaPhi= (2*PI)/double(params.height);
	// calc r(phi) for given phi and radii of ellipse. see http://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center for reference
	R=deltaRx/2+deltaRx*iPosY*(deltaRy/2+deltaRy*iPosY)/sqrt(pow((deltaRy/2+deltaRy*iPosY)*cos(deltaPhi/2+deltaPhi*iPosX),2)+pow((deltaRx/2+deltaRx*iPosY)*sin((deltaPhi/2+deltaPhi*iPosX)),2));
	// now calc rectangular coordinates from polar coordinates
	if (deltaRy == 0)
		R=0;
	prd.position.z=0;
	prd.position.x=cos(deltaPhi*iPosX)*R;
	prd.position.y=sin(deltaPhi*iPosX)*R;
	
};

/**********************************************************/
// device functions for distributing ray directions
/**********************************************************/

__forceinline__ __device__ void dirDistr_RandRect_device(rayStruct &prd, rayFieldParams &params)
{
	uint32_t x1[5]; // variable for random generator			
	
//    double r; // variable for generating random variables inside an ellipse
	// declar variables for randomly distributing ray directions via an importance area
//	double2 impAreaHalfWidth;
//	double3 dirImpAreaCentre, tmpPos, impAreaRoot;
//	double impAreaX, impAreaY, theta;

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
//	impAreaHalfWidth.x=(tan(params.alphaMax.x)-tan(params.alphaMin.x))/2;
//	impAreaHalfWidth.y=(tan(params.alphaMax.y)-tan(params.alphaMin.y))/2;
//	dirImpAreaCentre=make_double3(0,0,1);
//	rotateRay(&dirImpAreaCentre, rayAngleCentre);
	// the centre of the importance area is the root of the current geometry + the direction to the imp area centre normalized such that the importance area is 1mm away from the current geometry
//	impAreaRoot=make_double3(0,0,0)+dirImpAreaCentre/dot(make_double3(0,0,1), dirImpAreaCentre);
	// now distribute points inside importance area
//	theta=2*PI*Random(x1);
//	r=sqrt(Random(x1));
//	impAreaX=impAreaHalfWidth.x*r*cos(theta);
//	impAreaY=impAreaHalfWidth.y*r*sin(theta);
//	tmpPos=impAreaRoot+impAreaX*make_double3(1,0,0)+impAreaY*make_double3(0,1,0);
//	prd.direction=normalize(tmpPos-make_double3(0,0,0));
	// transform raydirection into global coordinate system
	//prd.direction=params.Mrot*prd.direction;

	// save seed for next randomization
	prd.currentSeed=x1[4];
};

__forceinline__ __device__ void dirDistr_RandRad_device(rayStruct &prd, rayFieldParams &params)
{
	// the strategy is to define an importance area that corresponds to the given emission cone. The ray directions are then distributed to aim in thi importance area
	double3 rayAngleCentre=make_double3((params.alphaMax.x+params.alphaMin.x)/2,(params.alphaMax.y+params.alphaMin.y)/2,0);
	double3 dirImpAreaCentre=params.rayDirection;
	rotateRay(&dirImpAreaCentre, rayAngleCentre+params.tilt);
	double3 impAreaRoot = prd.position+dirImpAreaCentre;
	double angleHeight=(params.alphaMax.y-params.alphaMin.y);
	double angleWidth=(params.alphaMax.x-params.alphaMin.x);
	double2 impAreaHalfWidth=make_double2(tan(angleHeight/2), tan(angleWidth/2));
	aimRayTowardsImpArea(prd.direction, prd.position, impAreaRoot, impAreaHalfWidth, params.tilt, AT_ELLIPT, prd.currentSeed);

};

__forceinline__ __device__ void dirDistr_RandNorm_Rect_device(rayStruct &prd, rayFieldParams &params)
{
	// the strategy is to define an importance area that corresponds to the given emission cone. The ray directions are then distributed to aim in thi importance area
	double3 rayAngleCentre;
	make_double3((params.alphaMax.x+params.alphaMin.x)/2,(params.alphaMax.y+params.alphaMin.y)/2,0);
	double angleHeight, angleWidth;
	angleHeight=params.alphaMax.y-params.alphaMin.y;
	angleWidth=params.alphaMax.x-params.alphaMin.x;
	double2 impAreaHalfWidth = make_double2(tan(angleHeight/2), tan(angleWidth/2));
	double3 dirImpAreaCentre=params.rayDirection;
	rotateRay(&dirImpAreaCentre,rayAngleCentre);
	double3 impAreaRoot = prd.position+dirImpAreaCentre;
	aimRayTowardsImpArea(prd.direction, prd.position, impAreaRoot, impAreaHalfWidth, params.importanceAreaTilt, AT_ELLIPT, prd.currentSeed);
};

__forceinline__ __device__ void dirDistr_RandImpArea_device(rayStruct &prd, rayFieldParams &params)
{

	aimRayTowardsImpArea(prd.direction, prd.position, params.importanceAreaRoot, params.importanceAreaHalfWidth, params.importanceAreaTilt, params.importanceAreaApertureType, prd.currentSeed);

//	uint32_t x1[5]; // variable for random generator			
	
//	// declar variables for randomly distributing ray directions via an importance area
//	double3 dirImpAreaCentre, tmpPos, impAreaRoot;
	
//	RandomInit(prd.currentSeed, x1); // init random variable

//	double impAreaX;
//	double impAreaY;
			
//	// now distribute points inside importance area

//	if (params.importanceAreaApertureType==AT_RECT)
//	{
		// place temporal point uniformingly randomly inside the importance area
//		impAreaX=(Random(x1)-0.5)*2*params.importanceAreaHalfWidth.x;
//		impAreaY=(Random(x1)-0.5)*2*params.importanceAreaHalfWidth.y; 
//	}
//	else 
//	{
//		if (params.importanceAreaApertureType==AT_ELLIPT)
//		{
//			double theta=2*PI*Random(x1);
//			double r=sqrt(Random(x1));
//			impAreaX=params.importanceAreaHalfWidth.x*r*cos(theta);
//			impAreaY=params.importanceAreaHalfWidth.y*r*sin(theta);
//		}
//	}
		
	
//	double3 impAreaAxisX=make_double3(1,0,0);
//	double3 impAreaAxisY=make_double3(0,1,0);
		
//	rotateRay(&impAreaAxisX,params.importanceAreaTilt);
//	rotateRay(&impAreaAxisY,params.importanceAreaTilt);

//	tmpPos=params.importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
//	prd.direction=normalize(tmpPos-prd.position);
	// save seed for next randomization
//	prd.currentSeed=x1[4];
};

__forceinline__ __device__ void dirDistr_Uniform_device(rayStruct &prd, rayFieldParams &params)
{
	prd.direction=params.rayDirection;
	// transform raydirection into global coordinate system
	prd.direction=params.Mrot*prd.direction;
};

__forceinline__ __device__ void dirDistr_GridRect_device(rayStruct &prd, rayFieldParams &params)
{
    double r; // variable for generating random variables inside an ellipse
	// declar variables for randomly distributing ray directions via an importance area
	double2 impAreaHalfWidth;
	double3 dirImpAreaCentre, tmpPos, impAreaRoot, impAreaAxisX, impAreaAxisY;
	double impAreaX, impAreaY, theta;
	// increment of temporary raypos in x and y 
	double deltaW=0;
	double deltaH=0;

	// calc increment along x- and y-direction
	if (params.nrRayDirections.x>1)
		deltaW= (2*params.importanceAreaHalfWidth.x)/double(params.nrRayDirections.x-1);
	if (params.nrRayDirections.y>1)
		deltaH= (2*params.importanceAreaHalfWidth.y)/double(params.nrRayDirections.y-1);
	impAreaX=-params.importanceAreaHalfWidth.x+deltaW/2+(launch_index.x+launch_offsetX)*deltaW; 
	impAreaY=-params.importanceAreaHalfWidth.y+deltaW/2+(launch_index.y+launch_offsetY)*deltaH; 
	impAreaAxisX=make_double3(1,0,0);
	impAreaAxisY=make_double3(0,1,0);
	rotateRay(&impAreaAxisX,params.importanceAreaTilt);
	rotateRay(&impAreaAxisY,params.importanceAreaTilt);

	tmpPos=params.importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
	prd.direction=normalize(tmpPos-prd.position);
};

__forceinline__ __device__ void dirDistr_GridRad_device(rayStruct &prd, rayFieldParams &params)
{
    double r; // variable for generating random variables inside an ellipse
	// declar variables for randomly distributing ray directions via an importance area
	double2 impAreaHalfWidth;
	double3 dirImpAreaCentre, tmpPos, impAreaRoot, impAreaAxisX, impAreaAxisY;
	double impAreaX, impAreaY, theta, deltaPhi;
	// increment of temporary raypos in x and y 
	double deltaRx=0;
	double deltaRy=0;
	
	unsigned long long iDirX=(launch_index.x+launch_offsetX);
	unsigned long long iDirY=(launch_index.y+launch_offsetY);

	// calc increment along radial and angular direction
	if (params.nrRayDirections.x>0)
	{
		deltaRx= (params.importanceAreaHalfWidth.x)/double(params.nrRayDirections.x);
		deltaRy= (params.importanceAreaHalfWidth.y)/double(params.nrRayDirections.x);
	}
	if (params.nrRayDirections.y>0)
		deltaPhi= (2*PI)/params.nrRayDirections.y;
	// calc r(phi) for given phi and radii of ellipse. see http://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center for reference
	double R=deltaRx/2+deltaRx*iDirY*(deltaRy/2+deltaRy*iDirY)/sqrt(pow((deltaRy/2+deltaRy*iDirY)*cos(deltaPhi/2+deltaPhi*iDirX),2)+pow((deltaRx/2+deltaRx*iDirY)*sin((deltaPhi/2+deltaPhi*iDirX)),2));
	// now calc rectangular coordinates from polar coordinates
	if ( deltaRy==0 )
		R=0;
	impAreaX=cos(deltaPhi*iDirX)*R;
	impAreaY=sin(deltaPhi*iDirX)*R;
	
	impAreaAxisX=make_double3(1,0,0);
	impAreaAxisY=make_double3(0,1,0);
	rotateRay(&impAreaAxisX,params.importanceAreaTilt);
	rotateRay(&impAreaAxisY,params.importanceAreaTilt);

	tmpPos=params.importanceAreaRoot+impAreaX*impAreaAxisX+impAreaY*impAreaAxisY;
	prd.direction=normalize(tmpPos-prd.position);
};

/**********************************************************/
//   ray generation programs
/**********************************************************/

RT_PROGRAM void rayGeneration()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
 
  rayStruct prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  //prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

//			switch (params.posDistrType)
//			{
//				case RAYPOS_GRID_RECT:
//					posDistr_GridRect_device(prd, params);
//					break;
//				case RAYPOS_RAND_RECT:	
//					posDistr_RandRect_device(prd, params);
//					break;
//				case RAYPOS_GRID_RAD:
//					posDistr_GridRad_device(prd, params);
//					break;
//				case RAYPOS_RAND_RAD:
//					posDistr_RandRad_device(prd, params);
//					break;
//				default:
//					rtPrintf("RAYPOS_DEFAULT");
//					prd.position=make_double3(0,0,0);
//					// report error
//					break;
//			}
//			// transform rayposition into global coordinate system
//			prd.position=params.Mrot*prd.position+params.translation;
			
//			switch (params.dirDistrType)
//			{
//				case RAYDIR_UNIFORM:
//					dirDistr_Uniform_device(prd, params);
//					break;
//				case RAYDIR_RAND_RECT:
//				    dirDistr_Rand_device(prd, params);
//					break;
//				case RAYDIR_GRID_RECT:
//					dirDistr_GridRect_device(prd, params);
//					break;
//				case RAYDIR_GRID_RAD:
//					dirDistr_GridRad_device(prd, params);
//					break;
//				default:
//					prd.direction=make_double3(0,0,0);
//					// report error
//					break;
//			}

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}


RT_PROGRAM void rayGeneration_DirRandRect_PosRandRectNorm()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
  prd.flux = params.flux;
  prd.depth = 0;

  // set seed for each ray
  prd.currentSeed=seed_buffer[launch_index];

#ifdef TIME_VIEW
  t0=clock();
#endif 

	// calc ray position
	posDistr_RandRectNorm_device(prd, params);	
	// transform rayposition into global coordinate system
	prd.position=params.Mrot*prd.position+params.translation;
	// calc ray direction
	dirDistr_RandRect_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

/*************************************************************************
*
*               PosRandRad
*
*************************************************************************/



RT_PROGRAM void rayGeneration_DirRandRect_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_RandRect_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_DirUniform_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_Uniform_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

/*********************************************************************************
*
*                       PosGridRad
*
*********************************************************************************/

RT_PROGRAM void rayGeneration_DirRandRad_PosGridRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_RandRad_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_DirRandRect_PosGridRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_RandRect_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

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
  
  rayStruct prd;
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

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_DirUniform_PosGridRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_Uniform_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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

	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

/**********************************************************************************
*
*                    PosRandRect
*
***********************************************************************************/

RT_PROGRAM void rayGeneration_DirRandRad_PosRandRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_RandRad_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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

	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       break;
    }
  }
  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_DirRandRect_PosRandRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_RandRect_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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

	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
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
  
  rayStruct prd;
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

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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

	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       break;
    }
  }
  output_buffer[launch_index] = prd;//.direction;//prd.position;
}


RT_PROGRAM void rayGeneration_DirUniform_PosRandRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_Uniform_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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

	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       break;
    }
  }
  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

/*********************************************************************************
*
*             PosGridRect
*
*********************************************************************************/
RT_PROGRAM void rayGeneration_DirRandRad_PosGridRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
  
  rayStruct prd;
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
	dirDistr_RandRad_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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

	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_DirRandRect_PosGridRect()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
 
  rayStruct prd;
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
	dirDistr_RandRect_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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

	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
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
  
  rayStruct prd;
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

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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

	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}


RT_PROGRAM void rayGeneration_DirUniform_PosGridRect()
{
    rtPrintf(" start tracing ray %i,%i\n", (launch_index.x + launch_offsetX), (launch_index.y + launch_offsetY));
#ifdef TIME_VIEW
    clock_t t0, t1;
    double time;
#endif

    rayStruct prd;
    prd.flux = params.flux;
    prd.depth = 0;

    // set seed for each ray
    prd.currentSeed = seed_buffer[launch_index];

#ifdef TIME_VIEW
    t0=clock();
#endif 

    // calc ray position
    posDistr_GridRect_device(prd, params);
    // transform rayposition into global coordinate system
    prd.position = params.Mrot*prd.position + params.translation;
    // calc ray direction
    dirDistr_Uniform_device(prd, params);

    float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
    float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);


#ifdef TIME_VIEW
    t1 = clock(); 
    time = (double)(t1-t0);
    rtPrintf("time elapsed while creating ray: %lf ms\n", time);
#endif 

    prd.currentGeometryID = 0;
    prd.lambda = params.lambda;
    prd.nImmersed = params.nImmersed;
    prd.running = true;
    prd.opl = 0;

    for (;;)
    {
        //optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
        rtPrintf("%i, pos before trace: %lf %lf %lf\n", (launch_index.x + launch_offsetX), ray.origin.x, ray.origin.y, ray.origin.z);
        rtTrace(top_object, ray, prd);
        rtPrintf("%i, pos after trace: %lf %lf %lf\n", (launch_index.x + launch_offsetX), ray.origin.x, ray.origin.y, ray.origin.z);
        // update ray
        ray.origin = make_float3(prd.position);
        ray.direction = make_float3(prd.direction);

        rtPrintf("%i, pos after trace (2): %lf %lf %lf\n", (launch_index.x + launch_offsetX), ray.origin.x, ray.origin.y, ray.origin.z);
        if (!prd.running)
        {
            //prd.result += prd.radiance * prd.attenuation;
            break;
        }
    }

    output_buffer[launch_index] = prd;//.direction;//prd.position;
}

/***********************************************************************************
*
*             PosRandRad
*
***********************************************************************************/
RT_PROGRAM void rayGeneration_DirRandRad_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  rayStruct prd;
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
	dirDistr_RandRad_device(prd, params);

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_DirGridRect_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif

  rayStruct prd;
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

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}

RT_PROGRAM void rayGeneration_DirRandImpArea_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
 
  rayStruct prd;
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

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}


RT_PROGRAM void rayGeneration_DirGridRad_PosRandRad()
{
#ifdef TIME_VIEW
  clock_t t0, t1;
  double time;
#endif
 
  rayStruct prd;
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

  float3 ray_origin = make_float3(prd.position);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
  float3 ray_direction = make_float3(prd.direction);//make_float3(0.0f);
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);

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
	// update ray
	ray.origin=make_float3(prd.position);
	ray.direction=make_float3(prd.direction);

    if(!prd.running) 
    {
       //prd.result += prd.radiance * prd.attenuation;
       break;
    }
  }

  output_buffer[launch_index] = prd;//.direction;//prd.position;
}


//RT_PROGRAM void rayGeneration_WaveIn()
//{
//  float3 ray_origin = make_float3(0.0f);//(float)(launch_index.x+launch_offsetX), (float)(launch_index.y+launch_offsetY), origin.z);
//  float3 ray_direction = make_float3(0.0f);
//  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
//
//  rayStruct prd;
//  prd.flux = 1.f;
//  prd.depth = 0;
//  prd.currentGeometryID = 0;
//  prd.lambda=params.lambda;
//  prd.running=true;
//  // set seed for each ray
//  //prd.currentSeed=seed_buffer[launch_index];
//
//  /* create rays in a rectangular grid evenly spaced between params.rayPosEnd and params.rayPosStart */
//  double width = params.rayPosEnd.x-params.rayPosStart.x;
//  double height = params.rayPosEnd.y-params.rayPosStart.y;
//  
//  double deltaw=width/(double)(params.width-1);
//  double deltah=height/(double)(params.height-1);
//
//  prd.position = make_double3(params.rayPosStart.x+deltaw*((launch_index.x+launch_offsetX)),params.rayPosStart.y+deltah*((launch_index.y+launch_offsetY)),params.rayPosStart.z);
//  prd.direction.x=wavfronIn_single_polyval_rowc(&(xGrad_buffer[0]),prd.position.x/(width/2),prd.position.y/(height/2), size_yGrad, size_xGrad);
//
//  prd.direction.y=wavfronIn_single_polyval_rowc(&(yGrad_buffer[0]),prd.position.x/(width/2),prd.position.y/(height/2), size_yGrad, size_xGrad);
//
//  prd.direction.z=sqrt(1-pow(prd.direction.y,2)-pow(prd.direction.x,2));
//
//    for(;;) 
//	{
//      optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, RT_DEFAULT_MAX);
//      rtTrace(top_object, ray, prd);
//      if(!prd.running) 
//	  {
//        //prd.result += prd.radiance * prd.attenuation;
//        break;
//      }
//	}
//  //rtTrace(top_object, ray, prd);
//
//	uint32_t x1[5]; // variable for random generator			
//	int seed = (int)clock();            // random seed
//	RandomInit(seed, x1);
//	for ( unsigned int i = 0; i < launch_index.x+(launch_index.y*launch_dim.x); ++i )
//		prd.currentSeed = (uint)BRandom(x1);
//
//  output_buffer[launch_index] = prd;//.direction;
//}

//RT_PROGRAM void rayListConvert()
//{
//	rayStruct prd;
//	//prd= list[(launch_index.y+launch_offsetY)*listlength+(launch_index.x+launch_offsetX)];
//	prd= list[(launch_index.y+launch_offsetY)*10+(launch_index.x+launch_offsetX)]; //hier muss statt 10 die berechnete breite stehen
//
//	optix::Ray ray = optix::make_Ray(make_float3(0,0,0), make_float3(0,0,0), 0, 0, RT_DEFAULT_MAX);
//	
//	rtTrace(top_object, ray, prd);
//	output_buffer[launch_index] = prd.position;
//}


RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, (launch_index.x+launch_offsetX), (launch_index.y+launch_offsetY) );
//  output_buffer[launch_index] = prd.position;
}
