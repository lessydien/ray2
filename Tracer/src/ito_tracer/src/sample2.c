
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
//#include <optix_math_new.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>
#include <sampleConfig.h>
#include <sutilCommonDefines.h>

char path_to_ptx[512];


#define NUM_BOXES 1
unsigned int width  = 400u;
unsigned int height = 400u;


typedef struct struct_BoxExtent{
    float min[3];
    float max[3];
} BoxExtent;

typedef struct
{
  double x, y, z;
}  double3Struct;



void createContext( RTcontext* context, RTbuffer* buffer );
void createMaterialAbsorbing( RTcontext context, RTmaterial* material );
void createMaterialRefracting( RTcontext context, RTmaterial* material );
void createGeometrySphericalSurf( RTcontext context, RTgeometry* geometry );
void createGeometryPlaneSurf( RTcontext context, RTgeometry* geometry );
void createInstancesSphericalSurf( RTcontext context, RTgeometry geometry, RTmaterial material, RTgeometrygroup geometrygroup );
void createInstancesPlaneSurf( RTcontext context, RTgeometry geometry, RTmaterial material, RTgeometrygroup geometrygroup );
void printUsageAndExit( const char* argv0 );

int main(int argc, char* argv[])
{
    /* Primary RTAPI objects */
    RTcontext           context;
    RTbuffer            output_buffer_obj;
    RTgroup				top_level_group;
	RTgeometrygroup		geometrygroup;
    RTvariable			top_object;
	RTacceleration		top_level_acceleration;
	RTacceleration		acceleration;
    RTgeometry          geometry_sphericalSurf;
	RTgeometry          geometry_planeSurf;
    RTmaterial          material_absorbing;
	RTmaterial          material_refracting;
    RTtransform			transforms;
	float				m[16]; // transformation matrix in homogenous coordinates
    int					i;
	FILE				*hFile; // handle for output file
	RTsize				buffer_width, buffer_height; // get size of output buffer
	void				*data; // pointer to cast output buffer into
	double3				*bufferData;
	unsigned int		j; // loop counter for printing the output buffer to output file
    


	
    /* Process command line args */
    RT_CHECK_ERROR_NO_CONTEXT( sutilInitGlut( &argc, argv ) );
    for( i = 1; i < argc; ++i ) {
      if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 ) {
        printUsageAndExit( argv[0] );
      } else if ( strncmp( argv[i], "--dim=", 6 ) == 0 ) {
        const char *dims_arg = &argv[i][6];
        if ( sutilParseImageDimensions( dims_arg, &width, &height ) != RT_SUCCESS ) {
          fprintf( stderr, "Invalid window dimensions: '%s'\n", dims_arg );
          printUsageAndExit( argv[0] );
        }
      } else {
        fprintf( stderr, "Unknown option '%s'\n", argv[i] );
        printUsageAndExit( argv[0] );
      }
    }

    /* Setup state */
    createContext( &context, &output_buffer_obj );
    createGeometrySphericalSurf( context, &geometry_sphericalSurf );
	createGeometryPlaneSurf( context, &geometry_planeSurf );
    createMaterialAbsorbing( context, &material_absorbing);
	createMaterialRefracting( context, &material_refracting);

    /* create top level group in context */
    RT_CHECK_ERROR( rtGroupCreate( context, &top_level_group ) );
    RT_CHECK_ERROR( rtGroupSetChildCount( top_level_group, 1 ) );

    RT_CHECK_ERROR( rtContextDeclareVariable( context, "top_object", &top_object ) );
    RT_CHECK_ERROR( rtVariableSetObject( top_object, top_level_group ) );

    RT_CHECK_ERROR( rtAccelerationCreate( context, &top_level_acceleration ) );
    RT_CHECK_ERROR( rtAccelerationSetBuilder(top_level_acceleration,"NoAccel") );
    RT_CHECK_ERROR( rtAccelerationSetTraverser(top_level_acceleration,"NoAccel") );
    RT_CHECK_ERROR( rtGroupSetAcceleration( top_level_group, top_level_acceleration) );

	RT_CHECK_ERROR( rtAccelerationMarkDirty( top_level_acceleration ) );

    /* create geometry group group to hold instance transform */
    RT_CHECK_ERROR( rtGeometryGroupCreate( context, &geometrygroup ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChildCount( geometrygroup, 2 ) );

    createInstancesPlaneSurf( context, geometry_planeSurf, material_absorbing, geometrygroup );
	createInstancesSphericalSurf( context, geometry_sphericalSurf, material_refracting, geometrygroup );

    /* create acceleration object for geometrygroup and specify some build hints*/
    RT_CHECK_ERROR( rtAccelerationCreate(context,&acceleration) );
    RT_CHECK_ERROR( rtAccelerationSetBuilder(acceleration,"NoAccel") );
    RT_CHECK_ERROR( rtAccelerationSetTraverser(acceleration,"NoAccel") );
    RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( geometrygroup, acceleration) );

    /* mark acceleration as dirty */
    RT_CHECK_ERROR( rtAccelerationMarkDirty( acceleration ) );

	RT_CHECK_ERROR( rtGroupSetChild( top_level_group, 0, geometrygroup ) );

	/* add a transform node */
    m[ 0] = 1.0f;  m[ 1] = 0.0f;  m[ 2] = 0.0f;  m[ 3] = 0.0f;
    m[ 4] = 0.0f;  m[ 5] = 1.0f;  m[ 6] = 0.0f;  m[ 7] = 0.0f;
    m[ 8] = 0.0f;  m[ 9] = 0.0f;  m[10] = 1.0f;  m[11] = 0.0f;
    m[12] = 0.0f;  m[13] = 0.0f;  m[14] = 0.0f;  m[15] = 1.0f;

    RT_CHECK_ERROR( rtTransformCreate( context, &transforms ) );
    RT_CHECK_ERROR( rtTransformSetChild( transforms, geometrygroup ) );
    m[11] = 0.0f;//i*1.0f - (NUM_BOXES-1)*0.5f;
	m[7] = 0.0f;//i*0.1f;
    RT_CHECK_ERROR( rtTransformSetMatrix( transforms, 0, m, 0 ) );

    /* Run */
    RT_CHECK_ERROR( rtContextValidate( context ) );
    RT_CHECK_ERROR( rtContextCompile( context ) );
    RT_CHECK_ERROR( rtContextLaunch2D( context, 0, width, height ) );

	/* unmap output-buffer */
	RT_CHECK_ERROR( rtBufferGetSize2D(output_buffer_obj, &buffer_width, &buffer_height) );
	// recast from Optix RTsize to standard int
	width = (int)(buffer_width);
	height = (int)(buffer_height);//static_cast<int>(buffer_height);

	RT_CHECK_ERROR( rtBufferMap(output_buffer_obj, &data) );
	// cast data pointer to format of the output buffer
	bufferData=(double3*)data;
	RT_CHECK_ERROR( rtBufferUnmap( output_buffer_obj ) );


	
	/* write results to file */
	hFile = fopen( "e:\\mauch\\test.txt", "w" ) ;
    if( hFile == NULL )
    {
       puts ( "cannot open file" ) ;
    }
	else
	{
		fprintf(hFile, "%.12lf ;%.12lf ;%.12lf ;", bufferData[0].x, bufferData[0].y, bufferData[0].z);
		for (j=1; j<width*height; j++)
		{
			// write the date in row major format, wher width is the size of one row and height is the size of one coloumn
			// if the end of a row is reached append a line feed 
			if ((j+1) % (width) == 0)
			{
				fprintf(hFile, "%.12lf ;%.12lf ;%.12lf \n", bufferData[j].x, bufferData[j].y, bufferData[j].z);
			}
			else
			{
				fprintf(hFile, "%.12lf ;%.12lf ;%.12lf ;", bufferData[j].x, bufferData[j].y, bufferData[j].z);
			}
		}
		fclose(hFile);
	}


    ///* Display image */
    //RT_CHECK_ERROR( sutilDisplayBufferInGlutWindow( argv[0], output_buffer_obj ) );

    /* Clean up */
    RT_CHECK_ERROR( rtContextDestroy( context ) );



	return( 0 );
}


void createContext( RTcontext* context, RTbuffer* output_buffer_obj )
{
    RTprogram  ray_gen_program;
    RTprogram  miss_program;
    RTvariable output_buffer;
    RTvariable radiance_ray_type;
//    RTvariable shadow_ray_type;
    RTvariable epsilon;


    /* variables for ray gen program */
    RTvariable origin;
    RTvariable U;
    RTvariable V;
    RTvariable W;

    /* variables for the miss program */

    /* Setup context */
    RT_CHECK_ERROR2( rtContextCreate( context ) );
    RT_CHECK_ERROR2( rtContextSetRayTypeCount( *context, 1 ) ); /* shadow and radiance */
    RT_CHECK_ERROR2( rtContextSetEntryPointCount( *context, 1 ) );

    RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "output_buffer", &output_buffer ) );
    RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "radiance_ray_type", &radiance_ray_type ) );
    RT_CHECK_ERROR2( rtContextDeclareVariable( *context, "scene_epsilon", &epsilon ) );

    RT_CHECK_ERROR2( rtVariableSet1ui( radiance_ray_type, 0u ) );
    RT_CHECK_ERROR2( rtVariableSet1f( epsilon, 1.e-4f ) );

    /* Render result buffer */
    RT_CHECK_ERROR2( rtBufferCreate( *context, RT_BUFFER_OUTPUT, output_buffer_obj ) );
    RT_CHECK_ERROR2( rtBufferSetFormat( *output_buffer_obj, RT_FORMAT_USER ) );
	RT_CHECK_ERROR2( rtBufferSetElementSize( *output_buffer_obj, sizeof(double3) ) );
    RT_CHECK_ERROR2( rtBufferSetSize2D( *output_buffer_obj, width, height ) );
    RT_CHECK_ERROR2( rtVariableSetObject( output_buffer, *output_buffer_obj ) );

    /* Ray generation program */
    sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_rayGeneration.cu.ptx" );
    RT_CHECK_ERROR2( rtProgramCreateFromPTXFile( *context, path_to_ptx, "rayGeneration", &ray_gen_program ) );
    RT_CHECK_ERROR2( rtProgramDeclareVariable( ray_gen_program, "origin", &origin ) );
    RT_CHECK_ERROR2( rtProgramDeclareVariable( ray_gen_program, "U", &U ) );
    RT_CHECK_ERROR2( rtProgramDeclareVariable( ray_gen_program, "V", &V ) );
    RT_CHECK_ERROR2( rtProgramDeclareVariable( ray_gen_program, "W", &W ) );
    RT_CHECK_ERROR2( rtVariableSet3f( origin, 0.0f, 0.0f, 5.0f ) );
    RT_CHECK_ERROR2( rtVariableSet3f( U, 1.0f, 0.0f, 0.0f ) );
    RT_CHECK_ERROR2( rtVariableSet3f( V, 0.0f, 1.0f, 0.0f ) );
    RT_CHECK_ERROR2( rtVariableSet3f( W, 0.0f, 0.0f, -1.0f ) );
    RT_CHECK_ERROR2( rtContextSetRayGenerationProgram( *context, 0, ray_gen_program ) );

    /* Miss program */
	sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_missFunction.cu.ptx" );
    RT_CHECK_ERROR2( rtProgramCreateFromPTXFile( *context, path_to_ptx, "miss", &miss_program ) );
    RT_CHECK_ERROR2( rtContextSetMissProgram( *context, 0, miss_program ) );
}

void createGeometryPlaneSurf( RTcontext context, RTgeometry* geometry )
{
    RTprogram  geometry_intersection_program;
    RTprogram  geometry_bounding_box_program;
    RTvariable box_min_var;
    RTvariable box_max_var;

    float     box_min[3];
    float     box_max[3];

    RT_CHECK_ERROR( rtGeometryCreate( context, geometry ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( *geometry, 1u ) );

    sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_planeSurface.cu.ptx" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "bounds", &geometry_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( *geometry, geometry_bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "intersect", &geometry_intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( *geometry, geometry_intersection_program ) );


    box_min[0] = box_min[1] =0.0f;
	box_min[2] = -500.0f;
    box_max[0] = box_max[1] =100.0f;
	box_max[2] =  0.1f;

    RT_CHECK_ERROR( rtGeometryDeclareVariable( *geometry, "boxmin", &box_min_var ) );
    RT_CHECK_ERROR( rtGeometryDeclareVariable( *geometry, "boxmax", &box_max_var ) );
    RT_CHECK_ERROR( rtVariableSet3fv( box_min_var, box_min ) );
    RT_CHECK_ERROR( rtVariableSet3fv( box_max_var, box_max ) );
}

void createGeometrySphericalSurf( RTcontext context, RTgeometry* geometry )
{
    RTprogram  geometry_intersection_program;
    RTprogram  geometry_bounding_box_program;
    RTvariable box_min_var;
    RTvariable box_max_var;

    float     box_min[3];
    float     box_max[3];

    RT_CHECK_ERROR( rtGeometryCreate( context, geometry ) );
    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( *geometry, 1u ) );

    sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_sphericalSurface.cu.ptx" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "bounds", &geometry_bounding_box_program ) );
    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( *geometry, geometry_bounding_box_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "intersect", &geometry_intersection_program ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( *geometry, geometry_intersection_program ) );


    box_min[0] = box_min[1] =0.0f;
	box_min[2] = -500.0f;
    box_max[0] = box_max[1] =100.0f;
	box_max[2] =  0.1f;

    RT_CHECK_ERROR( rtGeometryDeclareVariable( *geometry, "boxmin", &box_min_var ) );
    RT_CHECK_ERROR( rtGeometryDeclareVariable( *geometry, "boxmax", &box_max_var ) );
    RT_CHECK_ERROR( rtVariableSet3fv( box_min_var, box_min ) );
    RT_CHECK_ERROR( rtVariableSet3fv( box_max_var, box_max ) );
}

void createMaterialAbsorbing( RTcontext context, RTmaterial* material )
{
    RTprogram closest_hit_program;
    RTprogram any_hit_program;

    /* Create our hit programs to be shared among all materials */
    sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionAbsorbing.cu.ptx" );
	RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closestHit", &closest_hit_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "anyHit", &any_hit_program ) );

    RT_CHECK_ERROR( rtMaterialCreate( context, material ) );

    /* Note that we are leaving anyHitProgram[0] and closestHitProgram[1] as NULL.
     * This is because our radiance rays only need closest_hit and shadow rays only
     * need any_hit */
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( *material, 0, closest_hit_program ) );
    RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( *material, 0, any_hit_program ) );
}

void createMaterialRefracting( RTcontext context, RTmaterial* material )
{
    RTprogram closest_hit_program;
    RTprogram any_hit_program;

    /* Create our hit programs to be shared among all materials */
	sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "sample2_generated_hitFunctionRefracting.cu.ptx" );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "closestHit", &closest_hit_program ) );
    RT_CHECK_ERROR( rtProgramCreateFromPTXFile( context, path_to_ptx, "anyHit", &any_hit_program ) );

    RT_CHECK_ERROR( rtMaterialCreate( context, material ) );

    /* Note that we are leaving anyHitProgram[0] and closestHitProgram[1] as NULL.
     * This is because our radiance rays only need closest_hit and shadow rays only
     * need any_hit */
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( *material, 0, closest_hit_program ) );
    RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( *material, 0, any_hit_program ) );
}

void createInstancesPlaneSurf( RTcontext context, RTgeometry geometry, RTmaterial material, RTgeometrygroup geometrygroup )
{
    RTvariable		planeNormal;
	RTvariable		planeRoot;
	RTvariable		geometryID;
	double3Struct planeNormalVar;
	double3Struct planeRootVar;
    
        RTgeometryinstance instance;

        /* Create this geometry instance */
        RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
        RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, geometry ) );
        RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( instance, 1 ) );
        RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, material ) );

		/* add this geometry instance to geometry group */
        RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 0, instance ) );

		/* set the variables of the geometry */
		RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "planeNormal", &planeNormal ) );
		RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "planeRoot", &planeRoot ) );
		RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "geometryID", &geometryID ) );

		planeNormalVar.x=(double)0;
		planeNormalVar.y=(double)0;
		planeNormalVar.z=(double)1;
		planeRootVar.x=(double)0;
		planeRootVar.y=(double)0;
		planeRootVar.z=(double)-600;
		RT_CHECK_ERROR( rtVariableSetUserData(planeNormal, sizeof(double3Struct), &planeNormalVar) );
		RT_CHECK_ERROR( rtVariableSetUserData(planeRoot, sizeof(double3Struct), &planeRootVar) );
		RT_CHECK_ERROR( rtVariableSet1i(geometryID, 1) );

}

void createInstancesSphericalSurf( RTcontext context, RTgeometry geometry, RTmaterial material, RTgeometrygroup geometrygroup )
{
	RTvariable		sphereRadius;
	RTvariable		sphereCentre;
	RTvariable		sphereOrientation;
	RTvariable		sphereApertureRadius;
    RTgeometryinstance instance;
	RTvariable		geometryID;
	double3Struct	sphereCentreVar;
	double3Struct	sphereOrientationVar;
	double			sphereRadiusVar;
	double			sphereApertureRadiusVar;

        /* Create this geometry instance */
        RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
        RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, geometry ) );
        RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( instance, 1 ) );
        RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, material ) );

		/* add this geometry instance to geometry group */
        RT_CHECK_ERROR( rtGeometryGroupSetChild( geometrygroup, 1, instance ) );

        /* set geometry variables */
		RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "sphereCentre", &sphereCentre ) );
		RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "sphereRadius", &sphereRadius ) );
		RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "sphereOrientation", &sphereOrientation ) );
		RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "sphereApertureRadius", &sphereApertureRadius ) );
		RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "geometryID", &geometryID ) );

		sphereCentreVar.x=(double)200;
		sphereCentreVar.y=(double)200;
		sphereCentreVar.z=(double)-250;
		sphereOrientationVar.x=(double)0;
		sphereOrientationVar.y=(double)0;
		sphereOrientationVar.z=(double)1;
		sphereRadiusVar=(double)200;
		sphereApertureRadiusVar=(double)100;
		RT_CHECK_ERROR( rtVariableSetUserData(sphereCentre, sizeof(double3Struct), &sphereCentreVar) );
		RT_CHECK_ERROR( rtVariableSetUserData(sphereOrientation, sizeof(double3Struct), &sphereOrientationVar) );
		RT_CHECK_ERROR( rtVariableSetUserData(sphereRadius, sizeof(double), &sphereRadiusVar) );
		RT_CHECK_ERROR( rtVariableSetUserData(sphereApertureRadius, sizeof(double), &sphereApertureRadiusVar) );

		RT_CHECK_ERROR( rtVariableSet1i(geometryID, 2) );

}


void printUsageAndExit( const char* argv0 )
{
  fprintf( stderr, "Usage  : %s [options]\n", argv0 );
  fprintf( stderr, "Options: --help | -h             Print this usage message\n" );
  fprintf( stderr, "         --dim=<width>x<height>  Set image dimensions; defaults to 512x384\n" );
  exit(0);
}


