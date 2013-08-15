
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

/*
 *  sample2.cpp -- generates test scene and traces rays through the scene
 */


#include <optix.h>
#include "vector_functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>
#include <sampleConfig.h>
#include <sutilCommonDefines.h>
#include "Group.h"
#include "Geometry.h"
#include "GeometryLib.h"
#include "PlaneSurface.h"
#include "SphericalSurface.h"
#include "AsphericalSurface.h"
#include "ConePipe.h"
#include "CylPipe.h"
#include "RayField.h"
#include <math.h>
#include "rayTracingMath.h"
#include <ctime>
#include "FlexZemax.h"
#include "MaterialLib.h"



#define NUM_BOXES 1

/* define the size of ray structure */
unsigned long width  = 101;
unsigned long height = 101;

/*definition of global constants*/
//constants for raytracing on CPU
const int MAX_DEPTH_CPU=10;
const float MIN_FLUX_CPU=0.001f;
const double MAX_TOLERANCE=1e-10;

typedef struct struct_BoxExtent{
    float min[3];
    float max[3];
} BoxExtent;

// declare scene group
Group oGroup;

void createContext( RTcontext* context, RTbuffer* buffer );
void createGeometrySphericalSurf( SphericalSurface &oSphericalSurface);
void createGeometryAsphericalSurf( AsphericalSurface &oAsphericalSurface);
void createGeometryPlaneSurf( PlaneSurface &oPlaneSurface);
void printUsageAndExit( const char* argv0 );
void doCPUTracing(Group &oGroup, RayField &oRayField);
void createSceneFromZemax(Group *oGroup, FILE *hfile);


int main(int argc, char* argv[])
{
    /* Primary RTAPI objects */
	RTcontext           context;
	FILE				*hFile; // handle for output file
 	RTsize				buffer_width, buffer_height; // get size of output buffer
    RTbuffer            output_buffer_obj;
	void				*data; // pointer to cast output buffer into
 	double3				*bufferData;
	unsigned int		j; // loop counter for printing the output buffer to output file
    

    ///* Process command line args */
    //RT_CHECK_ERROR_NO_CONTEXT( sutilInitGlut( &argc, argv ) );
    //for( i = 1; i < argc; ++i ) {
    //  if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 ) {
    //    printUsageAndExit( argv[0] );
    //  } else if ( strncmp( argv[i], "--dim=", 6 ) == 0 ) {
    //    const char *dims_arg = &argv[i][6];
    //    if ( sutilParseImageDimensions( dims_arg, &width1, &height1 ) != RT_SUCCESS ) {
    //      fprintf( stderr, "Invalid window dimensions: '%s'\n", dims_arg );
    //      printUsageAndExit( argv[0] );
    //    }
    //  } else {
    //    fprintf( stderr, "Unknown option '%s'\n", argv[i] );
    //    printUsageAndExit( argv[0] );
    //  }
    //}

	/* create the context */
	createContext( &context, &output_buffer_obj );

	/* create scene */
//	Group oGroup;

	/*******************************************************/
	/* create geometry from parsing Zemax description file */
	/*******************************************************/

	/* get handle to Zemax description file */
	FILE *hfile = fopen( "e:\\mauch\\prescription_nonsequential_aspheres_max.txt", "r" ) ;
	createSceneFromZemax(&oGroup, hfile);

	/*********************************************************************/
	/* create geometry manually                                           */
	/*********************************************************************/

	///* create instances */
	//oGroup.setGeometryGroupListLength(1);
	//GeometryGroup oGeometrygroup(2);
	///* create material */
	//MaterialAbsorbing *oMaterialAbsorbingPtr = new MaterialAbsorbing();
	//char path_to_ptx_MaterialAbsorbing[512];
	//sprintf( path_to_ptx_MaterialAbsorbing, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionAbsorbing.cu.ptx" );
	//oMaterialAbsorbingPtr->setPathToPtx(path_to_ptx_MaterialAbsorbing);

	//////Material oMaterialRefracting;
	//////char path_to_ptx_MaterialRefracting[512];
	//////sprintf( path_to_ptx_MaterialRefracting, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionRefracting.cu.ptx" );
	//////oMaterialRefracting.setPathToPtx(path_to_ptx_MaterialRefracting);
	////
	//MaterialReflecting *oMaterialReflectingPtr = new MaterialReflecting();
	//char path_to_ptx_MaterialReflecting[512];
	//sprintf( path_to_ptx_MaterialReflecting, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionReflecting.cu.ptx" );
	//oMaterialReflectingPtr->setPathToPtx(path_to_ptx_MaterialReflecting);


	///* init geometry */
	///* aspherical surface */
	//AsphericalSurface *oAsphericalSurfacePtr = new AsphericalSurface(1);
	//createGeometryAsphericalSurf(*oAsphericalSurfacePtr);
	//oAsphericalSurfacePtr->createMaterial(0);
	//oAsphericalSurfacePtr->setMaterial(oMaterialReflectingPtr,0);
	///* plane surface */
	//PlaneSurface *oPlaneSurfacePtr=new PlaneSurface(1);
	//createGeometryPlaneSurf(*oPlaneSurfacePtr);
	//oPlaneSurfacePtr->createMaterial(0);
	//oPlaneSurfacePtr->setMaterial(oMaterialAbsorbingPtr,0);

	///* fill scene graph */
	//oGeometrygroup.setGeometry(oAsphericalSurfacePtr,1);
	//oGeometrygroup.setGeometry(oPlaneSurfacePtr,0);
	//oGroup.setGeometryGroup(&oGeometrygroup, 0);

	/****************************************************************/
	/* init ray field                                               */
	/****************************************************************/

	RayField oRayField(width*height);
	/* init RayField */
	//char path_to_ptx_rayGeneration[512];
	//sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_rayGeneration.cu.ptx" );
	//oRayField.setPathToPtx(path_to_ptx_rayGeneration);
	//oRayField.createOptixInstance(context,width,make_double3(-5,-5,0),make_double3(5,5,0),make_double3(0,0,1));
	//

	//Set Values of Rays and Edges of the Rayfield
	double3 rayPosition;
	rayPosition.x=-4;
	rayPosition.y=-4;
	rayPosition.z=0;
	double3 rayPosStart,rayPosEnd;
	rayPosStart.x=-5;
	rayPosStart.y=-5;
	rayPosStart.z=0;
	rayPosEnd.x=5;
	rayPosEnd.y=5;
	rayPosEnd.z=0;
	double3 rayDirection;
	rayDirection.x=0;
	rayDirection.y=0;
	rayDirection.z=1;
	rayDirection=normalize(rayDirection);

	//asphere_params params;
	//params=oAsphericalSurface.getParams();
	
	//oRayField.createRayList(width,rayPosStart,rayPosEnd,rayDirection,1.0);
	char path_to_ptx_rayGeneration[512];
	sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_rayGeneration.cu.ptx" );
	oRayField.setPathToPtx(path_to_ptx_rayGeneration);
	oRayField.createRayList(width,rayPosStart,rayPosEnd,rayDirection, 1.0);
	doCPUTracing(oGroup,oRayField);



	/* convert to GPU code */
	//unsigned long rayNumber=11;
	//oRayField.createOptixInstance(context,rayNumber,rayPosStart,rayPosEnd,rayDirection);
	//oGroup.createOptixInstance(context);

	///* set an excpetion program */
	//RTprogram		exception_program;
	//rtContextSetExceptionEnabled(context, RT_EXCEPTION_ALL, 1);
	//char path_to_ptx_exception[512];
	//sprintf( path_to_ptx_exception, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_exception.cu.ptx" );
	//RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx_exception, "exception_program", &exception_program ) );
	//rtContextSetExceptionProgram( context,1, exception_program );


	///* Run */
	//RT_CHECK_ERROR( rtContextValidate( context ) );
 //   RT_CHECK_ERROR( rtContextCompile( context ) );
 //   RT_CHECK_ERROR( rtContextLaunch2D( context, 0, width, height ) );

	///* unmap output-buffer */
	//RT_CHECK_ERROR( rtBufferGetSize2D(output_buffer_obj, &buffer_width, &buffer_height) );
	//// recast from Optix RTsize to standard int
	//width = (int)(buffer_width);
	//height = (int)(buffer_height);//static_cast<int>(buffer_height);

	//RT_CHECK_ERROR( rtBufferMap(output_buffer_obj, &data) );
	//// cast data pointer to format of the output buffer
	//bufferData=(double3*)data;
	//RT_CHECK_ERROR( rtBufferUnmap( output_buffer_obj ) );
	//
	///* write GPU results to file */
	//hFile = fopen( "e:\\mauch\\test.txt", "w" ) ;
 //   if( hFile == NULL )
 //   {
 //      puts ( "cannot open file" ) ;
 //   }
	//else
	//{
	//	//double tHelp1=(double)405;
	//	//double tHelp2=(double)-0.99503;
	//	//double test=(double)tHelp1 / (double)tHelp2;
	//	//float tHelp1f=(float)405;
	//	//float tHelp2f=(float)-0.99503;
	//	//float testf=tHelp1f / tHelp2f;
	//	//bufferData[0].x=(double)405 / (double)-0.99503;// = prd.position;
	//	//bufferData[0].y=test;
	//	//bufferData[0].z=(double)testf;
	//	fprintf(hFile, "%.16lf ;%.16lf ;%.16lf ;", bufferData[0].x, bufferData[0].y, bufferData[0].z);
	//	for (j=1; j<width*height; j++)
	//	{
	//		// write the date in row major format, wher width is the size of one row and height is the size of one coloumn
	//		// if the end of a row is reached append a line feed 
	//		if ((j+1) % (width) == 0)
	//		{
	//			fprintf(hFile, "%.16lf ;%.16lf ;%.16lf \n", bufferData[j].x, bufferData[j].y, bufferData[j].z);
	//		}
	//		else
	//		{
	//			fprintf(hFile, "%.16lf ;%.16lf ;%.16lf ;", bufferData[j].x, bufferData[j].y, bufferData[j].z);
	//		}
	//	}
	//	fclose(hFile);
	//}

	///* Clean up */
	//RT_CHECK_ERROR( rtContextDestroy( context ) );



	/*************************************************************************/
	/* do time measurements                                                  */
	/*************************************************************************/
	//clock_t start, end;
	//double msecs;
	//const int runs=1;
	//double2 messdaten[runs];
	//unsigned long rayNumber=1;

	//bool RunOnCPU;
	//RunOnCPU=false;
	//for(i=0;i<runs;i++)
	//{
	//	
	//	rayNumber=rayNumber+0;
	//	RayField oRayField1(rayNumber*rayNumber);
	//	oRayField1.createRayList(rayNumber,rayPosStart,rayPosEnd,rayDirection);
	//	//oRayField1.createRayList(rayNumber,rayNumber,0.0,rayDirection,rayPosition);
	//	
	//	/* chose wether to run on CPU or GPU */
	//	if (RunOnCPU)
	//	{
	//		//time measuring
	//		start=clock();

	//		doCPUTracing(oGroup,oRayField1);

	//		end=clock();
	//	}
	//	else
	//	{

	//		RTcontext           context;
	//		RTprogram		exception_program;
	//		/* Setup state */
	//		createContext( &context, &output_buffer_obj );
	//		rtContextSetExceptionEnabled(context, RT_EXCEPTION_ALL, 1);
	//				
	//		char path_to_ptx_exception[512];
	//		sprintf( path_to_ptx_exception, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_exception.cu.ptx" );
	//		RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx_exception, "exception_program", &exception_program ) );
	//		rtContextSetExceptionProgram( context,1, exception_program );

	//		// convert to GPU code
	//		oGroup.createOptixInstance(context);
	//			
	//		// init GPU variables
	//		char path_to_ptx_rayGeneration[512];
	//		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_rayGeneration.cu.ptx" );
	//		oRayField1.setPathToPtx(path_to_ptx_rayGeneration);
	//		// convert to GPU code
	//		oRayField1.createOptixInstance(context,rayNumber,rayPosStart,rayPosEnd,rayDirection);
	//		//oRayField1.convertRayListToGPU(context);

	//		RT_CHECK_ERROR( rtContextValidate( context ) );
	//		RT_CHECK_ERROR( rtContextCompile( context ) );

	//		//time measuring
	//		start=clock();

	//		// start GPU tracing
	//		RT_CHECK_ERROR( rtContextLaunch2D( context, 0, rayNumber, rayNumber ) );
	//			
	//		/* unmap output-buffer */
	//		RT_CHECK_ERROR( rtBufferGetSize2D(output_buffer_obj, &buffer_width, &buffer_height) );
	//		// recast from Optix RTsize to standard int
	//		width = (unsigned long)(buffer_width);
	//		height = (unsigned long)(buffer_height);//static_cast<int>(buffer_height);

	//		RT_CHECK_ERROR( rtBufferMap(output_buffer_obj, &data) );
	//		// cast data pointer to format of the output buffer
	//		bufferData=(double3*)data;
	//		RT_CHECK_ERROR( rtBufferUnmap( output_buffer_obj ) );
	//		end=clock();

	//		/* Clean up */
	//		RT_CHECK_ERROR( rtContextDestroy( context ) );


	//	}


	//	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	//	messdaten[i].x=rayNumber*rayNumber;
	//	messdaten[i].y=msecs;

	//}

	////write time measurement results to file
	//	if (RunOnCPU)
	//	{
	//		hFile = fopen( "e:\\mauch\\CPUTime.txt", "w" ) ;
	//		if( hFile == NULL )
	//		{
	//			puts ( "cannot open file" ) ;
	//		}
	//		else
	//		{
	//			fprintf(hFile,"Number of rays;CPU-Time;\n");
	//			for (j=0; j<runs; j++)
	//		{
	//			fprintf(hFile, "%d;%d;\n",(int)messdaten[j].x,(int)messdaten[j].y);
	//		}
	//		}
	//		fclose(hFile);
	//			
	//	}
	//	else
	//	{
	//		hFile = fopen( "e:\\mauch\\GPUTime.txt", "w" ) ;
	//		if( hFile == NULL )
	//		{
	//			puts ( "cannot open file" ) ;
	//		}
	//		else
	//		{
	//			fprintf(hFile,"Number of rays;GPU-Time;\n");
	//			for (j=0; j<runs; j++)
	//		{
	//			fprintf(hFile, "%d;%d;\n",(int)messdaten[j].x,(int)messdaten[j].y);
	//		}
	//		}
	//		fclose(hFile);
	//	}
		

	/* write CPU results to file */
hFile = fopen( "e:\\mauch\\RayTracingResults\\testZMaxCPU.txt", "w" ) ;
    if( hFile == NULL )
    {
       puts ( "cannot open file" ) ;
    }
	else
	{

		fprintf(hFile, "%.16lf ;%.16lf ;%.16lf ;", oRayField.getRay(0)->position.x,oRayField.getRay(0)->position.y, oRayField.getRay(0)->position.z);
		for (j=1; j<width*height; j++)
		{
			// write the date in row major format, where width is the size of one row and height is the size of one coloumn
			// if the end of a row is reached append a line feed 
			if ((j+1) % (width) == 0)
			{
				fprintf(hFile, "%.16lf ;%.16lf ;%.16lf \n", oRayField.getRay(j)->position.x,oRayField.getRay(j)->position.y, oRayField.getRay(j)->position.z);
			}
			else
			{
				fprintf(hFile, "%.16lf ;%.16lf ;%.16lf ;", oRayField.getRay(j)->position.x,oRayField.getRay(j)->position.y, oRayField.getRay(j)->position.z);
			}
		}
		fclose(hFile);
	}



    /* Display image */
//    RT_CHECK_ERROR( sutilDisplayBufferInGlutWindow( argv[0], output_buffer_obj ) );



	return( 0 );
}

void createContext( RTcontext* context, RTbuffer* output_buffer_obj )
{
    RTprogram  miss_program;
    RTvariable output_buffer;


    /* variables for the miss program */

    /* Setup context */
    RT_CHECK_ERROR2( rtContextCreate( context ) );
    RT_CHECK_ERROR2( rtContextSetRayTypeCount( *context, 1 ) ); /* shadow and radiance */
    RT_CHECK_ERROR2( rtContextSetEntryPointCount( *context, 1 ) );

    RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "output_buffer", &output_buffer ) );

    /* Render result buffer */
    RT_CHECK_ERROR2( rtBufferCreate( *context, RT_BUFFER_OUTPUT, output_buffer_obj ) );
    RT_CHECK_ERROR2( rtBufferSetFormat( *output_buffer_obj, RT_FORMAT_USER ) );
	RT_CHECK_ERROR2( rtBufferSetElementSize( *output_buffer_obj, sizeof(double3) ) );
    RT_CHECK_ERROR2( rtBufferSetSize2D( *output_buffer_obj, width, height ) );
    RT_CHECK_ERROR2( rtVariableSetObject( output_buffer, *output_buffer_obj ) );


	char path_to_ptx[512];
    /* Miss program */
	sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_missFunction.cu.ptx" );
    RT_CHECK_ERROR2( rtProgramCreateFromPTXFile( *context, path_to_ptx, "miss", &miss_program ) );
    RT_CHECK_ERROR2( rtContextSetMissProgram( *context, 0, miss_program ) );

}

void createGeometryPlaneSurf( PlaneSurface &oPlaneSurface )
{
	/* set attributes of plane surface */
	PlaneSurface_Params params;
	float box_max[3];
	float box_min[3];
    box_min[0] = box_min[1] =-1000.0f;
	box_min[2] = -1000.0f;
    box_max[0] = box_max[1] =1000.0f;
	box_max[2] =  1000.1f;
	oPlaneSurface.setBoundingBox_max(box_max);
	oPlaneSurface.setBoundingBox_min(box_min);
	params.normal.x=(double)0;
	params.normal.y=(double)0;
	params.normal.z=(double)1;
	params.root.x=(double)0;
	params.root.y=(double)0;
	params.root.z=(double)55.83745;
	oPlaneSurface.setParams(&params);
	oPlaneSurface.setID(1);
	/* create material */
	char pathBoundingBox[512];
	sprintf( pathBoundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_boundingBox.cu.ptx" );
	oPlaneSurface.setPathToPtxBoundingBox(pathBoundingBox);
	char pathIntersect[512];
	sprintf( pathIntersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_PlaneSurface.cu.ptx" );
	oPlaneSurface.setPathToPtxIntersect(pathIntersect);
}

void createGeometrySphericalSurf( SphericalSurface &oSphericalSurface)
{
	float box_max[3];
	float box_min[3];
    box_min[0] = box_min[1] =-1000.0f;
	box_min[2] = -1000.0f;
    box_max[0] = box_max[1] =1000.0f;
	box_max[2] =  1000.0f;
	oSphericalSurface.setBoundingBox_max(box_max);
	oSphericalSurface.setBoundingBox_min(box_min);
	SphericalSurface_Params params;
	params.centre.x=(double)0;
	params.centre.y=(double)0;
	params.centre.z=(double)400;
	params.orientation.x=(double)0;
	params.orientation.y=(double)0;
	params.orientation.z=(double)1;
	params.curvatureRadius.x=200;
	params.curvatureRadius.y=200;
	params.apertureRadius.x=100;
	params.apertureRadius.y=100;
	oSphericalSurface.setParams(&params);
	oSphericalSurface.setID(2);
	oSphericalSurface.setType(GEOM_SPHERICALSURF);
	char pathBoundingBox[512];
	sprintf( pathBoundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_boundingBox.cu.ptx" );
	oSphericalSurface.setPathToPtxBoundingBox(pathBoundingBox);
	char pathIntersect[512];
	sprintf( pathIntersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_sphericalSurface.cu.ptx" );
	oSphericalSurface.setPathToPtxIntersect(pathIntersect);
	//oSphericalSurface.setMaterial(material);
}

void createGeometryAsphericalSurf( AsphericalSurface &oAsphericalSurface)
{
	float box_max[3];
	float box_min[3];
    box_min[0] = box_min[1] =-1000.0f;
	box_min[2] = -1000.0f;
    box_max[0] = box_max[1] =1000.0f;
	box_max[2] =  1000.0f;
	oAsphericalSurface.setBoundingBox_max(box_max);
	oAsphericalSurface.setBoundingBox_min(box_min);
	AsphericalSurface_Params params;
	params.c=1./34.322;
	params.c2=0;
	params.c4=-4.851391*pow(10.0,-6);
	params.c6=6.673651*pow(10.0,-10);
	params.c8=-5.226474*pow(10.0,-12);
	params.c10=2.580481*pow(10.0,-15);
	params.k=-9.2582*pow(10.0,-2);
	params.orientation=make_double3(0,0,1);
	params.vertex=make_double3(0,0,10);
	oAsphericalSurface.setParams(&params);
	oAsphericalSurface.setID(2);
	oAsphericalSurface.setType(GEOM_ASPHERICALSURF);
	char pathBoundingBox[512];
	sprintf( pathBoundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_boundingBox.cu.ptx" );
	oAsphericalSurface.setPathToPtxBoundingBox(pathBoundingBox);
	char pathIntersect[512];
	sprintf( pathIntersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_asphericalSurface.cu.ptx" );
	oAsphericalSurface.setPathToPtxIntersect(pathIntersect);
}




void printUsageAndExit( const char* argv0 )
{
  fprintf( stderr, "Usage  : %s [options]\n", argv0 );
  fprintf( stderr, "Options: --help | -h             Print this usage message\n" );
  fprintf( stderr, "         --dim=<width>x<height>  Set image dimensions; defaults to 512x384\n" );
  exit(0);
}

void doCPUTracing(Group &oGroup, RayField &oRayField)
{
	// loop over rayList
	unsigned long index;
	rayStruct* rayList;
	rayList=oRayField.getRayList();
	for (index=0; index < oRayField.getRayListLength(); index++)
	{
		oGroup.trace(rayList[index]);
	}

}

void createSceneFromZemax(Group *oGroupPtr, FILE *hfile)
{
	/* output the geometry-data for debugging purpose */
	/* get handle to parser-debug file */
	FILE *hfileDebug = fopen( "e:\\mauch\\geometry.txt", "w" ) ;
	// define structure to hold the parse results
	parseResultStruct parseResults;
	/* parse Zemax description file */
	parserError err=parseZemaxPrescr(&parseResults, hfile);
	if (err != PARSER_NO_ERR)
	{
		fprintf( hfileDebug, parseResults.errMsg);
		return;
	}

	/* set number of geometry groups */
	oGroupPtr->setGeometryGroupListLength(1);
	/* create a geometryGroup inside the group object */
	oGroupPtr->createGeometryGroup(0);
	/* set number of geometries */
	oGroupPtr->getGeometryGroup(0)->setGeometryListLength(parseResults.geomNumber);

	int k=0;
	PlaneSurface *oPlaneSurfacePtr;
	PlaneSurface_Params planeParams;
	CylPipe_Params cylParams;
	CylPipe *oCylPipePtr;
	ConePipe_Params coneParams;
	ConePipe *oConePipePtr;
	SphericalSurface *oSphericalSurfacePtr;
	SphericalSurface_Params sphereParams;
	AsphericalSurface *oAsphericalSurfacePtr;
	AsphericalSurface_Params asphereParams;
	Material *oMaterialPtr;
	double theta;
	//MaterialReflecting *oMaterialReflectingPtr;
	//MaterialRefracting *oMaterialRefractingPtr;
	//MaterialAbsorbing *oMaterialAbsorbingPtr;
	Geometry *oGeometryPtr;

	GeometryParamStruct test;
	for (k=0; k<parseResults.geomNumber;k++)
	{
		
		test=parseResults.geometryParams[k];

		switch(parseResults.geometryParams[k].type)
		{
			case (GEOM_PLANESURF):
				/* allocate memory for plane surface */
				oGeometryPtr = new PlaneSurface(1);
				oPlaneSurfacePtr=dynamic_cast<PlaneSurface*>(oGeometryPtr);
				/* set geometry params */
				planeParams.normal=parseResults.geometryParams[k].normal;
				planeParams.root=parseResults.geometryParams[k].root;
				planeParams.apertureType=parseResults.geometryParams[k].aperture;
				planeParams.apertureRadius=parseResults.geometryParams[k].apertureHalfWidth1;
				oPlaneSurfacePtr->setParams(&planeParams);

				break;
			case (GEOM_GRATING):
				/* allocate memory for plane surface */
				oGeometryPtr = new PlaneSurface(1);
				oPlaneSurfacePtr=dynamic_cast<PlaneSurface*>(oGeometryPtr);
				/* set geometry params */
				planeParams.normal=parseResults.geometryParams[k].normal;
				planeParams.root=parseResults.geometryParams[k].root;
				planeParams.apertureRadius=parseResults.geometryParams[k].apertureHalfWidth1;
				oPlaneSurfacePtr->setParams(&planeParams);

				break;
			case (GEOM_SPHERICALSURF):
				/* allocate memory for spherical surface */
				oGeometryPtr = new SphericalSurface(1);
				oSphericalSurfacePtr=dynamic_cast<SphericalSurface*>(oGeometryPtr);
				sphereParams.orientation=parseResults.geometryParams[k].normal;
				sphereParams.centre=parseResults.geometryParams[k].root;
				sphereParams.apertureRadius.x=parseResults.geometryParams[k].apertureHalfWidth1.x;
				sphereParams.apertureRadius.y=parseResults.geometryParams[k].apertureHalfWidth1.y;
				sphereParams.curvatureRadius.x=parseResults.geometryParams[k].radius1.x;
				sphereParams.curvatureRadius.y=parseResults.geometryParams[k].radius1.y;
				oSphericalSurfacePtr->setParams(&sphereParams);

				break;
			case (GEOM_ASPHERICALSURF):
				/* allocate memory for aspherical surface */
				oGeometryPtr = new AsphericalSurface(1);
				oAsphericalSurfacePtr=dynamic_cast<AsphericalSurface*>(oGeometryPtr);
				asphereParams.orientation=parseResults.geometryParams[k].normal;
				asphereParams.vertex =parseResults.geometryParams[k].root;
				asphereParams.apertureRadius.x=parseResults.geometryParams[k].apertureHalfWidth1.x;
				asphereParams.apertureRadius.y=parseResults.geometryParams[k].apertureHalfWidth1.y;
				asphereParams.apertureType=parseResults.geometryParams[k].aperture;
				asphereParams.k=-9.2582*pow(10.0,-2);;//parseResults.geometryParams[k].params[0];
				asphereParams.c=1/parseResults.geometryParams[k].radius1.x;
				//asphereParams.k.y=parseResults.geometryParams[k].radius1.y;
				asphereParams.c2=parseResults.geometryParams->params[1];
				asphereParams.c4=parseResults.geometryParams->params[2];
				asphereParams.c6=parseResults.geometryParams->params[3];
				asphereParams.c8=parseResults.geometryParams->params[4];
				asphereParams.c10=parseResults.geometryParams->params[5];
				oAsphericalSurfacePtr->setParams(&asphereParams);
				break;
			case (GEOM_CYLPIPE):
				/* allocate memory for cylindrcial pipe */
				oGeometryPtr = new CylPipe(1);
				oCylPipePtr=dynamic_cast<CylPipe*>(oGeometryPtr);
				cylParams.orientation=parseResults.geometryParams[k].normal;
				cylParams.root=parseResults.geometryParams[k].root;
				cylParams.thickness=parseResults.geometryParams[k].thickness;
				cylParams.radius=parseResults.geometryParams[k].apertureHalfWidth1;
				oCylPipePtr->setParams(&cylParams);

				break;
			case (GEOM_CONEPIPE):
				/* allocate memory for cone pipe */
				oGeometryPtr = new ConePipe(1);
				oConePipePtr=dynamic_cast<ConePipe*>(oGeometryPtr);
				coneParams.orientation=parseResults.geometryParams[k].normal;
				coneParams.root=parseResults.geometryParams[k].root;
				coneParams.thickness=parseResults.geometryParams[k].thickness;
				theta=atan((parseResults.geometryParams[k].apertureHalfWidth2.x-parseResults.geometryParams[k].apertureHalfWidth1.x)/coneParams.thickness);
				// so far only rotationally symmetric cones are implemented. Therefore cosTheta.y is not calculated !!
				coneParams.cosTheta.x=cos(theta);
				coneParams.coneEnd=coneParams.root-coneParams.orientation*parseResults.geometryParams[k].apertureHalfWidth1.x/tan(theta);
				oConePipePtr->setParams(&coneParams);

				break;

			default:
				/* allocate memory for plane surface */
				oGeometryPtr = new PlaneSurface(1);
				oPlaneSurfacePtr=dynamic_cast<PlaneSurface*>(oGeometryPtr);
				/* set geometry params */
				planeParams.normal=parseResults.geometryParams[k].normal;
				planeParams.root=parseResults.geometryParams[k].root;
				oPlaneSurfacePtr->setParams(&planeParams);

				break;
		} // end switch

		switch (parseResults.geometryParams[k].glass)
		{
			case MT_MIRROR:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialReflecting();
				break;
			case MT_NBK7:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialRefracting();
				break;
			default:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialAbsorbing();
				break;
		} // end switch glass
		/* copy the pointer to the material. Note that we do not release the allocated memory */
		/* here. This will be taken care of in the destructor of the geometry!!          */
		oGeometryPtr->setMaterial(oMaterialPtr,0);
		oGeometryPtr->setID(k);
		oGeometryPtr->setComment(parseResults.geometryParams[k].comment);
		/* copy the pointer to the geometry. Note that we do not release the allocated memory */
		/* here. This will be taken care of in the destructor of the geometryGroup!!          */
		oGroupPtr->getGeometryGroup(0)->setGeometry(oGeometryPtr, k);


	}


//	int k=0;
	Geometry* geometryPtrDebug;
	PlaneSurface *ptrPlane;
	SphericalSurface *ptrSphere;
	CylPipe *ptrCyl;
	ConePipe *ptrCone;
	//char testString[20];
	//sprintf(testString, "%s", "this is a test");
	//test=oGroupPtr->getGeometryGroup(0)->getGeometry(0);
	//test->setComment(testString);
	for (k=0; k<oGroupPtr->getGeometryGroup(0)->getGeometryListLength();k++)
	{
	//Material oMaterialAbsorbing;
	//char path_to_ptx_MaterialAbsorbing[512];
	//sprintf( path_to_ptx_MaterialAbsorbing, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionAbsorbing.cu.ptx" );
	//oMaterialAbsorbing.setPathToPtx(path_to_ptx_MaterialAbsorbing);
		geometryPtrDebug=oGroupPtr->getGeometryGroup(0)->getGeometry(k);
		//test->createMaterial(0);
		//test->setMaterial(&oMaterialAbsorbing,0);
		//test->getMaterial(0)->setPathToPtx(path_to_ptx_MaterialAbsorbing);

		switch(geometryPtrDebug->getType())
		{
			case (GEOM_PLANESURF):
				/* cast pointer to plane surface */
				ptrPlane=dynamic_cast<PlaneSurface*>(geometryPtrDebug);
				/* output params */
				fprintf( hfileDebug, " geometry %i plane surf '%s': root: x,y,z: %f, %f, %f; \n", k, ptrPlane->getComment(), ptrPlane->getParams()->root.x, ptrPlane->getParams()->root.y, ptrPlane->getParams()->root.z);
				break;
			case (GEOM_SPHERICALSURF):
				/* cast pointer to spherical surface */
				ptrSphere=dynamic_cast<SphericalSurface*>(geometryPtrDebug);
				fprintf( hfileDebug, " geometry %i sphere '%s': root: x,y,z: %f, %f, %f; radius: %f; aperture: %f \n", k, ptrSphere->getComment(), ptrSphere->getParams()->centre.x, ptrSphere->getParams()->centre.y, ptrSphere->getParams()->centre.z, ptrSphere->getParams()->curvatureRadius.x, ptrSphere->getParams()->apertureRadius.x);
				break;
			case (GEOM_ASPHERICALSURF):
				/* cast pointer to aspherical surface */
				break;
			case (GEOM_CYLPIPE):
				ptrCyl=dynamic_cast<CylPipe*>(geometryPtrDebug);
				fprintf( hfileDebug, " geometry %i cylinder pipe '%s': root: x,y,z: %f, %f, %f; radius x, y: %f, %f \n", k, ptrCyl->getComment(), ptrCyl->getParams()->root.x, ptrCyl->getParams()->root.y, ptrCyl->getParams()->root.z, ptrCyl->getParams()->radius.x, ptrCyl->getParams()->radius.y);
				break;
			case (GEOM_CONEPIPE):
				ptrCone=dynamic_cast<ConePipe*>(geometryPtrDebug);
				fprintf( hfileDebug, " geometry %i cone pipe '%s': root: x,y,z: %f, %f, %f; cosTheta: %f; coneEnd x,y,z: %f, %f, %f \n", k, ptrCone->getComment(), ptrCone->getParams()->root.x, ptrCone->getParams()->root.y, ptrCone->getParams()->root.z, ptrCone->getParams()->cosTheta.x, ptrCone->getParams()->coneEnd.x, ptrCone->getParams()->coneEnd.y, ptrCone->getParams()->coneEnd.z);
				break;
			default:
				break;
		}
	}
	fclose(hfileDebug);
		
}

