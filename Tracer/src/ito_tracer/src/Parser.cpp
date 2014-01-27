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

/**\file Parser.cpp
* \brief collection of functions that create the internal scene graph from the result of parsing the various input files
* 
*           
* \author Mauch
*/

#include "Parser.h"
#include "FlexZemax.h"
#include <iostream>

/**
 * \detail createSceneFromZemax 
 *
 * parses the prescription files and creates an OptiX scene 
 *
 * \param[in] Group *oGroupPtr, FILE *hfile, RayField ***sourceListPtr, long long *sourceNumberPtr, Detector ***detListPtr, long long *detNumberPtr, TraceMode mode
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool createSceneFromZemax(Group *oGroupPtr, FILE *hfile, RayField ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, SimMode mode)
{

	std::cout <<"********************************************" << std::endl;
	std::cout <<"starting to parse prescritpion files..." << std::endl;
	/* output the geometry-data for debugging purpose */
	/* get handle to parser-debug file */
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "geometry.TXT");
	FILE *hfileDebug = fopen( filepath, "w" ) ;
	if (!hfileDebug)
	{
		std::cout <<"error in Parser.createSceneFromZemax(): cannot open description file: " << filepath << std::endl;
		return false;
	}
	// define structure to hold the parse results
	//parseResultStruct parseResults;
	//map<string,variant> *parseResults;
	parseResultStruct *parseResults;
	/* parse Zemax description file */
	parserError err=parseZemaxPrescr(&parseResults, hfile, mode);
	if (err != PARSER_NO_ERR)
	{
		std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxPrescr() returned an error" << std::endl;
		//fprintf( hfileDebug, parseResults->errMsg);
		return false;
	}

	switch (parseResults->simType)
	{
	case SIMTYPE_GEOM_RAYS:
		if (!createGeometricSceneFromZemax(oGroupPtr, parseResults, sourceListPtr, sourceNumberPtr, detListPtr, detNumberPtr, mode) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): createGeometricSceneFromZemax() returned an error" << std::endl;
			return false;
		}
		break;
	case SIMTYPE_DIFF_RAYS:
		if (!createDifferentialSceneFromZemax(oGroupPtr, parseResults, sourceListPtr, sourceNumberPtr, detListPtr, detNumberPtr, mode) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): createDifferentialSceneFromZemax() returned an error" << std::endl;
			return false;
		}
		break;
	case SIMTYPE_PATHTRACING:
		if (!createPathTracingSceneFromZemax(oGroupPtr, parseResults, sourceListPtr, sourceNumberPtr, detListPtr, detNumberPtr, mode) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): createPathTracingSceneFromZemax() returned an error" << std::endl;
			return false;
		}
		break;		
	default:
		std::cout <<"error in Parser.createSceneFromZemax(): unknown simulation type" << std::endl;
		return false;
		break;
	}

	// print the results of the scene creation for debugging purposes
	Geometry* geometryPtrDebug;
	PlaneSurface *ptrPlane;
	ApertureStop *ptrAptStop;
	PlaneSurface_Params *ptrPlaneParams;
	ApertureStop_Params *ptrAptStopParams;
	IdealLense *ptrIdealLense;
	IdealLense_Params *ptrIdealLenseParams;
	SphericalSurface *ptrSphere;
	SphericalSurface_Params *ptrSphereParams;
	SinusNormalSurface *ptrCosNorm;
	SinusNormalSurface_Params *ptrCosNormParams;
	CylPipe *ptrCyl;
	CylPipe_Params *ptrCylPipeParams;
	ConePipe *ptrCone;
	ConePipe_Params *ptrConePipeParams;
	//for (int k=0; k<oGroupPtr->getGeometryGroup(0)->getGeometryListLength();k++)
	//{
	//	geometryPtrDebug=oGroupPtr->getGeometryGroup(0)->getGeometry(k);

	//	switch(geometryPtrDebug->getType())
	//	{
	//		case (GEOM_PLANESURF):
	//			/* cast pointer to plane surface */
	//			ptrPlane=dynamic_cast<PlaneSurface*>(geometryPtrDebug);
	//			ptrPlaneParams=dynamic_cast<PlaneSurface_Params*>(ptrPlane->getParamsPtr());
	//			/* output params */
	//			fprintf( hfileDebug, " geometry %i plane surf '%s': root: x,y,z: %f, %f, %f; \n", ptrPlaneParams->geometryID, ptrPlane->getComment(), ptrPlaneParams->root.x, ptrPlaneParams->root.y, ptrPlaneParams->root.z);
	//			break;
	//		case (GEOM_APERTURESTOP):
	//			/* cast pointer to plane surface */
	//			ptrAptStop=dynamic_cast<ApertureStop*>(geometryPtrDebug);
	//			ptrAptStopParams=dynamic_cast<ApertureStop_Params*>(ptrAptStop->getParamsPtr());
	//			/* output params */
	//			fprintf( hfileDebug, " geometry %i aperture stop '%s': root: x,y,z: %f, %f, %f; apertureStopRadius x,y: %f, %f\n", ptrAptStopParams->geometryID, ptrAptStop->getComment(), ptrAptStopParams->root.x, ptrAptStopParams->root.y, ptrAptStopParams->root.z, ptrAptStopParams->apertureStopRadius.x, ptrAptStopParams->apertureStopRadius.y);
	//			break;
	//		case (GEOM_SPHERICALSURF):
	//			/* cast pointer to spherical surface */
	//			ptrSphere=dynamic_cast<SphericalSurface*>(geometryPtrDebug);
	//			ptrSphereParams=dynamic_cast<SphericalSurface_Params*>(ptrSphere->getParamsPtr());
	//			fprintf( hfileDebug, " geometry %i sphere '%s': root: x,y,z: %f, %f, %f; radius: %f; aperture: %f \n", ptrSphereParams->geometryID, ptrSphere->getComment(), ptrSphereParams->centre.x, ptrSphereParams->centre.y, ptrSphereParams->centre.z, ptrSphereParams->curvatureRadius.x, ptrSphereParams->apertureRadius.x);
	//			break;
	//		case (GEOM_ASPHERICALSURF):
	//			/* cast pointer to aspherical surface */
	//			break;
	//		case (GEOM_CYLPIPE):
	//			ptrCyl=dynamic_cast<CylPipe*>(geometryPtrDebug);
	//			ptrCylPipeParams=dynamic_cast<CylPipe_Params*>(ptrCyl->getParamsPtr());
	//			fprintf( hfileDebug, " geometry %i cylinder pipe '%s': root: x,y,z: %f, %f, %f; radius x, y: %f, %f \n", ptrCylPipeParams->geometryID, ptrCyl->getComment(), ptrCylPipeParams->root.x, ptrCylPipeParams->root.y, ptrCylPipeParams->root.z, ptrCylPipeParams->radius.x, ptrCylPipeParams->radius.y);
	//			break;
	//		case (GEOM_CONEPIPE):
	//			ptrCone=dynamic_cast<ConePipe*>(geometryPtrDebug);
	//			ptrConePipeParams=dynamic_cast<ConePipe_Params*>(ptrCone->getParamsPtr());
	//			fprintf( hfileDebug, " geometry %i cone pipe '%s': root: x,y,z: %f, %f, %f; cosTheta: %f; coneEnd x,y,z: %f, %f, %f \n", ptrConePipeParams->geometryID, ptrCone->getComment(), ptrConePipeParams->root.x, ptrConePipeParams->root.y, ptrConePipeParams->root.z, ptrConePipeParams->cosTheta.x, ptrConePipeParams->coneEnd.x, ptrConePipeParams->coneEnd.y, ptrConePipeParams->coneEnd.z);
	//			break;
	//		case (GEOM_IDEALLENSE):
	//			/* cast pointer to plane surface */
	//			ptrIdealLense=dynamic_cast<IdealLense*>(geometryPtrDebug);
	//			ptrIdealLenseParams=dynamic_cast<IdealLense_Params*>(ptrIdealLense->getParamsPtr());
	//			/* output params */
	//			fprintf( hfileDebug, " geometry %i ideal lense '%s': root: x,y,z: %f, %f, %f; \n", ptrIdealLenseParams->geometryID, ptrIdealLense->getComment(), ptrIdealLenseParams->root.x, ptrIdealLenseParams->root.y, ptrIdealLenseParams->root.z);
	//			break;
	//		case (GEOM_COSINENORMAL):
	//			/* cast pointer to plane surface */
	//			ptrCosNorm=dynamic_cast<SinusNormalSurface*>(geometryPtrDebug);
	//			ptrCosNormParams=dynamic_cast<SinusNormalSurface_Params*>(ptrCosNorm->getParamsPtr());
	//			/* output params */
	//			fprintf( hfileDebug, " geometry %i cosine normal '%s': root: x,y,z: %f, %f, %f; \n", ptrCosNormParams->geometryID, ptrCosNorm->getComment(), ptrCosNormParams->root.x, ptrCosNormParams->root.y, ptrCosNormParams->root.z);
	//			break;

	//		default:
	//			break;
	//	}
	//}
	fclose(hfileDebug); // close debug file

	return true;
}

/**
 * \detail createGeometricSceneFromZemax 
 *
 * parses the prescription files and creates an OptiX scene for geometric raytracing
 *
 * \param[in] Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long long *sourceNumberPtr, Detector ***detListPtr, long long *detNumberPtr, TraceMode mode
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool createGeometricSceneFromZemax(Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, SimMode mode)
{
	/* output the geometry-data for debugging purpose */
	/* get handle to parser-debug file */
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "geometry.TXT");
	FILE *hfileDebug = fopen( filepath, "w" ) ;
	if (!hfileDebug)
	{
		std::cout <<"error in Parser.createSceneFromZemax(): cannot open description file: " << filepath << std::endl;
		return false;
	}
	/* set number of geometry groups */
	if (GROUP_NO_ERR != oGroupPtr->setGeometryGroupListLength(1) )
	{
		std::cout <<"error in Parser.createSceneFromZemax(): group.setGeometryGroupListLength(1) returned an error" << std::endl;
		return false;
	}
	/* create a geometryGroup inside the group object at index 0 */
	if (GROUP_NO_ERR != oGroupPtr->createGeometryGroup(0) )
	{
		std::cout <<"error in Parser.createSceneFromZemax(): group.createGeometryGroup(0) returned an error" << std::endl;
		return false;
	}
	/* set number of geometries */
	if (GEOMGROUP_NO_ERR != oGroupPtr->getGeometryGroup(0)->setGeometryListLength(parseResults->geomNumber) )
	{
		std::cout <<"error in Parser.createSceneFromZemax(): group.getGeometryGroup(0)->setGeometryListLength(number) returned an error with number = " << parseResults->geomNumber << std::endl;
		return false;
	}

	/* create array for the sources of our simulation */
	*sourceListPtr=new RayField*[parseResults->sourceNumber];
	*sourceNumberPtr=parseResults->sourceNumber;
	/* create array for the detectors of our simulation */
	*detListPtr=new Detector*[parseResults->detectorNumber];
	*detNumberPtr=parseResults->detectorNumber;

	int k=0;
	double theta;
	// define a pointer to every object that can be parsed
	Detector *DetectorPtr;
	DetectorRaydata *DetectorRaydataPtr;
	DetectorIntensity *DetectorIntensityPtr;
	DetectorPhaseSpace *DetectorPhaseSpacePtr;
	DetectorField *DetectorFieldPtr;
	Geometry *oGeometryPtr;
	PlaneSurface *oPlaneSurfacePtr;
//	PlaneSurface_Params planeParams;
	ApertureStop *oApertureStopPtr;
//	ApertureStop_Params apertureStopParams;
	IdealLense *oIdealLensePtr;
//	IdealLense_Params idealLenseParams;
	CylLenseSurface *oCylLenseSurfPtr;
//	CylLenseSurface_Params cylLenseSurfParams;
	SinusNormalSurface *oSinusNormalPtr;
//	SinusNormalSurface_Params sinusNormalParams;
//	rayFieldParams *rayParamsPtr;
//	detParams *detParamsPtr;
//	detRaydataParams *detRaydataParamsPtr;
//	detIntensityParams *detIntensityParamsPtr;
//	detFieldParams *detFieldParamsPtr;
//	CylPipe_Params cylParams;
	CylPipe *oCylPipePtr;
//	ConePipe_Params coneParams;
	ConePipe *oConePipePtr;
	SphericalSurface *oSphericalSurfacePtr;
//	SphericalSurface_Params sphereParams;
	AsphericalSurface *oAsphericalSurfacePtr;
//	AsphericalSurface_Params asphereParams;
	Material *oMaterialPtr;
	MaterialDiffracting *oMaterialDiffractingPtr;
	MaterialReflecting *oMaterialReflectingPtr;
	MaterialReflecting_CovGlass *oMaterialReflecting_CovGlassPtr;
	MaterialRefracting *oMaterialRefractingPtr;
	MaterialRefracting_DiffRays *oMaterialRefracting_DiffRaysPtr;
	MaterialIdealLense *oMaterialIdealLensePtr;
	MaterialFilter *oMaterialFilterPtr;
	MaterialAbsorbing *oMaterialAbsorbingPtr;
//	Scatter_TorranceSparrow1D *oScatterTorrSparrPtr;
	MaterialLinearGrating1D *oMaterialLinearGrating1DPtr;
//	MatRefracting_params refrParams;
//	MatDiffracting_params diffParams;
//	MatIdealLense_DispersionParams idealLenseMatParams;
//	MatLinearGrating1D_params gratParams;
//	ScatTorranceSparrow1D_params torrSparr1DParams;
//	ScatTorranceSparrow2D_params torrSparr2DParams;
//	ScatLambert2D_params lambert2DParams;
	Scatter *oScatterPtr;
	Scatter_TorranceSparrow1D *oScatterTorrSparr1DPtr;
	Scatter_TorranceSparrow2D *oScatterTorrSparr2DPtr;
	Scatter_DoubleCauchy1D *oScatterDoubleCauchy1DPtr;
	Scatter_DispersiveDoubleCauchy1D *oScatterDispDoubleCauchy1DPtr;
	Scatter_Lambert2D *oScatterLambert2DPtr;
	Scatter_Params *scatterParamsPtr;
//	ScatTorranceSparrow1D_scatParams* scatTorrSparr1DParamsPtr;
//	ScatDoubleCauchy1D_scatParams* scatDoubleCauchy1DParamsPtr;
//	ScatLambert2D_scatParams* scatLambert2DParamsPtr;
	Coating *oCoatingPtr;
	Coating_DispersiveNumCoeffs *oCoatingDispNumCoeffPtr;
//	Coating_DispersiveNumCoeffs_FullParams *coatingDispNumCoeffsParamsPtr;
	Coating_NumCoeffs *oCoatingNumCoeffPtr;
//	Coating_NumCoeffs_FullParams *coatingNumCoeffsParamsPtr;
	Coating_FresnelCoeffs *oCoatingFresnelCoeffPtr;
//	Coating_FresnelCoeffs_FullParams *coatingFresnelCoeffsParamsPtr;
	GeometricRayField* GeomRayFieldPtr;

	int materialListLength;
	int coatingIndex;
	int scatterIndex;

	// define structures to hold the parse results
	parseGlassResultStruct* parseResultsGlassPtr;
	parseGlassResultStruct* parseResultsGlassImmPtr;
	ParseGratingResultStruct* parseResultsGratingPtr;
	MatLinearGrating1D_DiffractionParams* gratDiffractionParamsPtr;
	MatRefracting_DispersionParams* glassDispersionParamsPtr;
	MatRefracting_DispersionParams* immersionDispersionParamsPtr;
	MatDiffracting_DispersionParams* matDiffractParamsPtr;
	MatDiffracting_DispersionParams* matDiffImmersionDispersionParamsPtr;
	Coating_FullParams* coatingParamsPtr;//=new Coating_FullParams();

	/* define handle to the grating file */
	FILE *hfileGrating;
	/* get handle to glass catalog */
	FILE *hfileGlass;
	double3 tilt;

	// loop through detector definitions
	for (k=0; k<parseResults->detectorNumber;k++)
	{
		switch (parseResults->detectorParams[k].detectorType)
		{
		case DET_RAYDATA:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_GLOBAL:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_RED:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_RED_GLOBAL:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_INTENSITY:
			DetectorPtr=new DetectorIntensity();
			DetectorIntensityPtr=dynamic_cast<DetectorIntensity*>(DetectorPtr);

			DetectorIntensityPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorIntensityPtr;
			break;

		case DET_PHASESPACE:
			DetectorPtr=new DetectorPhaseSpace();
			DetectorPhaseSpacePtr=dynamic_cast<DetectorPhaseSpace*>(DetectorPtr);

			DetectorPhaseSpacePtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorPhaseSpacePtr;
			break;

		case DET_FIELD:
			DetectorPtr=new DetectorField();
			DetectorFieldPtr=dynamic_cast<DetectorField*>(DetectorPtr);

			DetectorFieldPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorFieldPtr;
			break;
			
		default:
			// some error mechanism
			std::cout << "error in Parser.createSceneFromZemax(): unknown detector type" << std::endl;
			return false;
			break;
		} // end switch detector type
	} // end loop detector definitions


	// loop through source definitions
	for (k=0; k<parseResults->sourceNumber;k++)
	{
		// create geometric rayField with rayListLength according to the parse raynumber
		//*sourceListPtr[k] = new GeometricRayField(parseResults->sourceParams->height*parseResults->sourceParams->width);
		// create geometric rayField with rayListLength according to the GPUSubset dimensions
		//*sourceListPtr[k] = new GeometricRayField(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX);
		*sourceListPtr[k] = new GeometricRayField();
		GeomRayFieldPtr=dynamic_cast<GeometricRayField*>(*sourceListPtr[k]);

		// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
		if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
		{
			/* get handle to the glass catalog file */
			sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
			hfileGlass = fopen( filepath, "r" );
			if (!hfileGlass)
			{
				std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
				fprintf( hfileDebug, "could not open glass file");
				return false;
			}
			/* parse Zemax glass catalog */
			if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
			{
				std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
				return false;
			}
		}
		else
		{
			parseResultsGlassPtr = NULL;
		}
		// parse importance area
		parseImpArea_Source(parseResults, k);

		if (FIELD_NO_ERR != GeomRayFieldPtr->processParseResults(parseResults->sourceParams[k], parseResultsGlassPtr) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): GeomRayField.processParseResults() returned an error in geometry number: " << k << std::endl;
			return false;
		}

	} // end loop source definitions

	// loop through geometry definitions
	GeometryParseParamStruct test;
	for (k=0; k<parseResults->geomNumber;k++)
	{
		test=parseResults->geometryParams[k];
		// set indices for scatter and coating to zero. When scatter or coating is present, this value will be overwritten and can be used to flag presence of the respective material
		coatingIndex=0;
		scatterIndex=0;

		/* create geometry */
		switch(parseResults->geometryParams[k].type)
		{
			case (GEOM_COSINENORMAL):
				/* allocate memory for plane surface */
				oGeometryPtr = new SinusNormalSurface();
				oSinusNormalPtr=dynamic_cast<SinusNormalSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oSinusNormalPtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oSinusNormalPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_IDEALLENSE):
				/* allocate memory for plane surface */
				oGeometryPtr = new IdealLense();
				oIdealLensePtr=dynamic_cast<IdealLense*>(oGeometryPtr);

				if (GEOM_NO_ERR != oIdealLensePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oIdealLensePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_PLANESURF):
				/* allocate memory for plane surface */
				oGeometryPtr = new PlaneSurface();
				oPlaneSurfacePtr=dynamic_cast<PlaneSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oPlaneSurfacePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oPlaneSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_APERTURESTOP):
				/* allocate memory for plane surface */
				oGeometryPtr = new ApertureStop();
				oApertureStopPtr=dynamic_cast<ApertureStop*>(oGeometryPtr);

				if (GEOM_NO_ERR != oApertureStopPtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oApertureStopPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_SPHERICALSURF):
				/* allocate memory for spherical surface */
				oGeometryPtr = new SphericalSurface();
				oSphericalSurfacePtr=dynamic_cast<SphericalSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oSphericalSurfacePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oSphericalSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_CYLLENSESURF):
				/* allocate memory for spherical surface */
				oGeometryPtr = new CylLenseSurface();
				oCylLenseSurfPtr=dynamic_cast<CylLenseSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oCylLenseSurfPtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCylLenseSurfPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_ASPHERICALSURF):
				/* allocate memory for aspherical surface */
				oGeometryPtr = new AsphericalSurface();
				oAsphericalSurfacePtr=dynamic_cast<AsphericalSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oAsphericalSurfacePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oAsphericalSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;
			case (GEOM_CYLPIPE):
				/* allocate memory for cylindrcial pipe */
				oGeometryPtr = new CylPipe();
				oCylPipePtr=dynamic_cast<CylPipe*>(oGeometryPtr);

				if (GEOM_NO_ERR != oCylPipePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCylPipePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;
			case (GEOM_CONEPIPE):
				/* allocate memory for cone pipe */
				oGeometryPtr = new ConePipe();
				oConePipePtr=dynamic_cast<ConePipe*>(oGeometryPtr);

				if (GEOM_NO_ERR != oConePipePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oConePipePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			default:
				std::cout << "error in createSceneFromZemax(): unknown geometry in geometry number: " << k << std::endl;
				return false;

				break;
		} // end switch

		// set material list length. so far we set it const to one!!
		if (oGeometryPtr->setMaterialListLength(1)!=GEOM_NO_ERR)
		{
			std::cout << "error in createSceneFromZemax(): setMaterialListLength(1) returned an error" << std::endl;
			return false;
		}
		/* create scatter */
		switch (parseResults->geometryParams[k].materialParams.scatterType)
		{
			case ST_TORRSPARR1D:
				oScatterTorrSparr1DPtr=new Scatter_TorranceSparrow1D(); 
//				parseImpArea_Material(parseResults, k); //importance area not implemented yet
				if (SCAT_NO_ERROR != oScatterTorrSparr1DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): scatTorrSparr1DParamsPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterTorrSparr1DPtr;

				break;
			case ST_TORRSPARR2D:
				oScatterTorrSparr2DPtr=new Scatter_TorranceSparrow2D(); 
				parseImpArea_Material(parseResults,k);
				if (SCAT_NO_ERROR != oScatterTorrSparr2DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): scatTorrSparr2DParamsPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterTorrSparr2DPtr;

				break;
			case ST_DOUBLECAUCHY1D:
				oScatterDoubleCauchy1DPtr=new Scatter_DoubleCauchy1D(); 
//				parseImpArea_Material(parseResults, k); //importance area not implemented yet
				if (SCAT_NO_ERROR != oScatterDoubleCauchy1DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oScatterDoubleCauchy1DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterDoubleCauchy1DPtr;
				break;

			case ST_DISPDOUBLECAUCHY1D:
				oScatterDispDoubleCauchy1DPtr=new Scatter_DispersiveDoubleCauchy1D(); 
//				parseImpArea_Material(parseResults, k); //importance area not implemented yet
				if (SCAT_NO_ERROR != oScatterDispDoubleCauchy1DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oScatterDispDoubleCauchy1DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterDispDoubleCauchy1DPtr;
				break;

			case ST_LAMBERT2D:
				oScatterLambert2DPtr=new Scatter_Lambert2D(); 
				parseImpArea_Material(parseResults, k); 
				if (SCAT_NO_ERROR != oScatterLambert2DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oScatterLambert2DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterLambert2DPtr;
				break;

			case ST_NOSCATTER:
				oScatterPtr=new Scatter(); 
				scatterParamsPtr=new Scatter_Params();
				scatterParamsPtr->type=ST_NOSCATTER;
				if ( SCAT_NO_ERROR != oScatterPtr->setFullParams(scatterParamsPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): Scatter.setFullParams() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				break;

			default:
				oScatterPtr=new Scatter(); 
				scatterParamsPtr=new Scatter_Params();
				scatterParamsPtr->type=ST_NOSCATTER;
				oScatterPtr->setFullParams(scatterParamsPtr);
				std::cout << "warning in createSceneFromZemax(): unknown scatter in geometry number: " << k << ". No scatter assumed..." << std::endl;
				break;
		} // end switch materialParams.scatterType
		

		/* create coating */
		switch (parseResults->geometryParams[k].materialParams.coatingType)
		{
			case CT_NOCOATING:
				oCoatingPtr=new Coating(); 
				coatingParamsPtr=new Coating_FullParams();
				coatingParamsPtr->type=CT_NOCOATING;
				if ( COAT_NO_ERROR != oCoatingPtr->setFullParams(coatingParamsPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): Coating.setFullParams() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				break;

			case CT_NUMCOEFFS:
				oCoatingNumCoeffPtr=new Coating_NumCoeffs(); 

				if (SCAT_NO_ERROR != oCoatingNumCoeffPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCoatingNumCoeffPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oCoatingPtr=oCoatingNumCoeffPtr;
				break;

			case CT_DISPNUMCOEFFS:
				oCoatingDispNumCoeffPtr=new Coating_DispersiveNumCoeffs(); 

				if (SCAT_NO_ERROR != oCoatingDispNumCoeffPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCoatingDispNumCoeffPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oCoatingPtr=oCoatingDispNumCoeffPtr;
				break;

			case CT_FRESNELCOEFFS:
				oCoatingFresnelCoeffPtr=new Coating_FresnelCoeffs(); 
				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassImmPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassImmPtr = NULL;
				}
				// if we neither have a user defined nor a standard glass we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.glassName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.glassName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.glassName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				if (SCAT_NO_ERROR != oCoatingFresnelCoeffPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr, parseResultsGlassPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCoatingFresnelCoeffPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oCoatingPtr=oCoatingFresnelCoeffPtr;
				break;


			default:
				oCoatingPtr=new Coating(); 
				coatingParamsPtr=new Coating_FullParams();
				coatingParamsPtr->type=CT_NOCOATING;
				oCoatingPtr->setFullParams(coatingParamsPtr);
				std::cout << "warning in createSceneFromZemax(): unknown coating in geometry number: " << k << ". No coating assumed..." << std::endl;
				break;
		} // end switch coating type

		/* create glass material */
		switch (parseResults->geometryParams[k].materialParams.matType)
		{
			case MT_DIFFRACT:
				oMaterialPtr = new MaterialDiffracting();
				oMaterialDiffractingPtr=dynamic_cast<MaterialDiffracting*>(oMaterialPtr);

				parseImpArea_Material( parseResults, k);

				// if we have an importance object defined, we read its aperture data and use it for our importance area
				// we need to do this here and not in the materials processParseResulst() function as we need to have acces to the aperture data of the other objects in the scene
				//if (parseResults->geometryParams[k].materialParams.importanceObjNr > 0 )
				//{
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].materialParams.apertureHalfWidth;
				//	parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].root;
				//	parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].tilt;
				//	parseResults->geometryParams[k].materialParams.importanceAreaApertureType=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].aperture;
				//}
				//else //if we have a cone angle defined, we calculate our importance area accordingly
				//{
				//	// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.x=(tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.x)-tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.x))/2;
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.y=(tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.y)-tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.y))/2;
				//	parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[k].root+parseResults->geometryParams[k].normal;
				//	parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[k].tilt;
				//}

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				if (MAT_NO_ERR != oMaterialDiffractingPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialDiffractingPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case MT_MIRROR:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialReflecting();
				break;

			case MT_IDEALLENSE:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialIdealLense();
				oMaterialIdealLensePtr=dynamic_cast<MaterialIdealLense*>(oMaterialPtr);

				if (MAT_NO_ERR != oMaterialIdealLensePtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialIdealLensePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case MT_FILTER:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialFilter();
				oMaterialFilterPtr=dynamic_cast<MaterialFilter*>(oMaterialPtr);

				if (MAT_NO_ERR != oMaterialFilterPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialFilterPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case MT_REFRMATERIAL:
				/* allocate memory for refracting surface */
				oMaterialPtr = new MaterialRefracting();
				oMaterialRefractingPtr=dynamic_cast<MaterialRefracting*>(oMaterialPtr);

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassImmPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassImmPtr = NULL;
				}
				// if we neither have a user defined nor a standard glass we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.glassName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.glassName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.glassName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				if (MAT_NO_ERR != oMaterialRefractingPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr, parseResultsGlassImmPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialRefractingPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				
				break;

			case MT_ABSORB:
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialAbsorbing();
				break;

			case MT_COVGLASS:
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialReflecting_CovGlass();
				oMaterialReflecting_CovGlassPtr=dynamic_cast<MaterialReflecting_CovGlass*>(oMaterialPtr);
				if (MAT_NO_ERR != oMaterialReflecting_CovGlassPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResults->detectorParams[0]) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialReflecting_CovGlassPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				else
				{
					// as we handle reflection coefficient directly in the material we delete the previous defined coating and set it to nocoating
					free(oCoatingPtr);
					oCoatingPtr=new Coating(); 
					coatingParamsPtr=new Coating_FullParams();
					coatingParamsPtr->type=CT_NOCOATING;
					if ( COAT_NO_ERROR != oCoatingPtr->setFullParams(coatingParamsPtr) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): Coating.setFullParams() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}

				break;

			case MT_LINGRAT1D:
				oMaterialPtr = new MaterialLinearGrating1D();
//				oMaterialPtr->setCoatingParams(coatingParamsPtr); // set coating parameters
				oMaterialLinearGrating1DPtr=dynamic_cast<MaterialLinearGrating1D*>(oMaterialPtr);

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassImmPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassImmPtr = NULL;
				}
				// if we neither have a user defined nor a standard glass we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.glassName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.glassName,"MIRROR") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.glassName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				// check wether we need to parse the grating file
				if (parseResults->geometryParams[k].materialParams.gratingEffsFromFile || parseResults->geometryParams[k].materialParams.gratingLinesFromFile || parseResults->geometryParams[k].materialParams.gratingOrdersFromFile)
				{
					// get handle to grating file
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "grating.TXT");
					hfileGrating = fopen( filepath, "r" ) ;
					if (!hfileGrating)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open grating file: " << filepath << ".  in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open grating file");
						return false;
					}
					/* parse MicroSim Grating Data */
					if (PARSER_NO_ERR != parseMicroSimGratingData(&parseResultsGratingPtr, hfileGrating) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseMicroSimGratingData() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}

				if (MAT_NO_ERR != oMaterialLinearGrating1DPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr, parseResultsGlassImmPtr, parseResultsGratingPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialLinearGrating1DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}				
				break;
			default:
				std::cout <<"warning: no material found in geometry number: " << k << " absorbing material is assumed." << std::endl;
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialAbsorbing();
				break;
		} // end switch materialParams.matType

		// if we have the detector at hands, we set its material fix to absorbing. The material that is assigned to the detector in prescription file is only for the coating...
		if (k==parseResults->geomNumber-1)
		{
			// discard the material we created in the parsing
			free(oMaterialPtr);
			oMaterialPtr=new MaterialAbsorbing();
		}
		// fuse coating, scatter, material and geometry into on object
		if (MAT_NO_ERR != oMaterialPtr->setCoating(oCoatingPtr) ) // set coating parameters
		{
			std::cout <<"error in Parser.createSceneFromZemax(): material.setCoating() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		if (MAT_NO_ERR != oMaterialPtr->setScatter(oScatterPtr) ) // set coating parameters
		{
			std::cout <<"error in Parser.createSceneFromZemax(): material.setScatter() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		/* copy the pointer to the material. Note that we do not release the allocated memory */
		/* here. This will be taken care of in the destructor of the geometry!!          */
		if (GEOM_NO_ERR != oGeometryPtr->setMaterial(oMaterialPtr,0) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): Geometry.set;aterial() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		//oGeometryPtr->setID(k);
		oGeometryPtr->setComment(parseResults->geometryParams[k].comment);

		/* copy the pointer to the geometrygroup. Note that we do not release the allocated memory */
		/* here. This will be taken care of in the destructor of the geometryGroup!!          */
		if (GEOMGROUP_NO_ERR != oGroupPtr->getGeometryGroup(0)->setGeometry(oGeometryPtr, k) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): geometryGroup.setGeometry() returned an error in geometry number: " << k << std::endl;
			return false;
		}
	} // end: for (k=0; k<parseResults->geomNumber;k++)
	return true;
};

/**
 * \detail createDifferentialSceneFromZemax 
 *
 * parses the prescription files and creates an OptiX scene for differential raytracing
 *
 * \param[in] Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long long *sourceNumberPtr, Detector ***detListPtr, long long *detNumberPtr, TraceMode mode
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool createDifferentialSceneFromZemax(Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, SimMode mode)
{
	/* output the geometry-data for debugging purpose */
	/* get handle to parser-debug file */
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "geometry.TXT");
	FILE *hfileDebug = fopen( filepath, "w" ) ;
	if (!hfileDebug)
	{
		std::cout <<"error in Parser.createDifferentialSceneFromZemax(): cannot open log file: " << filepath << std::endl;
		return false;
	}
	/* set number of geometry groups */
	if (GROUP_NO_ERR != oGroupPtr->setGeometryGroupListLength(1) )
	{
		std::cout <<"error in Parser.createDifferentialSceneFromZemax(): group.setGeometryGroupListLength(1) returned an error" << std::endl;
		return false;
	}
	/* create a geometryGroup inside the group object at index 0 */
	if (GROUP_NO_ERR != oGroupPtr->createGeometryGroup(0) )
	{
		std::cout <<"error in Parser.createDifferentialSceneFromZemax(): group.createGeometryGroup(0) returned an error" << std::endl;
		return false;
	}
	/* set number of geometries */
	if (GEOMGROUP_NO_ERR != oGroupPtr->getGeometryGroup(0)->setGeometryListLength(parseResults->geomNumber) )
	{
		std::cout <<"error in Parser.createDifferentialSceneFromZemax(): group.getGeometryGroup(0)->setGeometryListLength(number) returned an error with number = " << parseResults->geomNumber << std::endl;
		return false;
	}

	/* create array for the sources of our simulation */
	*sourceListPtr=new RayField*[parseResults->sourceNumber];
	*sourceNumberPtr=parseResults->sourceNumber;
	/* create array for the detectors of our simulation */
	*detListPtr=new Detector*[parseResults->detectorNumber];
	*detNumberPtr=parseResults->detectorNumber;

	int k=0;
	double theta;
	// define a pointer to every object that can be parsed
	Detector *DetectorPtr;
	DetectorRaydata *DetectorRaydataPtr;
	DetectorIntensity *DetectorIntensityPtr;
	DetectorField *DetectorFieldPtr;
	DetectorPhaseSpace *DetectorPhaseSpacePtr;
	Geometry *oGeometryPtr;
	PlaneSurface_DiffRays *oPlaneSurfacePtr;
	PlaneSurface_Params planeParams;
	ApertureStop_DiffRays *oApertureStopPtr;
	ApertureStop_Params apertureStopParams;
	IdealLense_DiffRays *oIdealLensePtr;
	IdealLense_Params idealLenseParams;
	SinusNormalSurface_DiffRays *oSinusNormalPtr;
	SinusNormalSurface_Params sinusNormalParams;
	rayFieldParams *rayParamsPtr;
	detParams *detParamsPtr;
	detRaydataParams *detRaydataParamsPtr;
	detIntensityParams *detIntensityParamsPtr;
	detFieldParams *detFieldParamsPtr;
	CylPipe_Params cylParams;
	CylPipe_DiffRays *oCylPipePtr;
	ConePipe_Params coneParams;
	ConePipe_DiffRays *oConePipePtr;
	SphericalSurface_DiffRays *oSphericalSurfacePtr;
	SphericalSurface_Params sphereParams;
	AsphericalSurface_DiffRays *oAsphericalSurfacePtr;
	AsphericalSurface_Params asphereParams;
	Material *oMaterialPtr;
	MaterialDiffracting_DiffRays *oMaterialDiffractingPtr;
	MaterialReflecting_DiffRays *oMaterialReflectingPtr;
	MaterialRefracting_DiffRays *oMaterialRefractingPtr;
	MaterialIdealLense_DiffRays *oMaterialIdealLensePtr;
	MaterialAbsorbing_DiffRays *oMaterialAbsorbingPtr;
	Scatter_TorranceSparrow1D_DiffRays *oScatterTorrSparrPtr;
	MaterialLinearGrating1D_DiffRays *oMaterialLinearGrating1DPtr;
	MatRefracting_params refrParams;
	MatDiffracting_params diffParams;
	MatIdealLense_DispersionParams idealLenseMatParams;
	MatLinearGrating1D_params gratParams;
	ScatTorranceSparrow1D_params torrSparr1DParams;
	ScatLambert2D_params lambert2DParams;
	Scatter *oScatterPtr;
	Scatter_DiffRays *oScatter_DiffRaysPtr;
	Scatter_TorranceSparrow1D_DiffRays *oScatterTorrSparr1DPtr;
	Scatter_DoubleCauchy1D_DiffRays *oScatterDoubleCauchy1DPtr;
	Scatter_Lambert2D_DiffRays *oScatterLambert2DPtr;
	Scatter_DiffRays_Params *scatterParamsPtr;
	ScatTorranceSparrow1D_scatParams* scatTorrSparr1DParamsPtr;
	ScatDoubleCauchy1D_scatParams* scatDoubleCauchy1DParamsPtr;
	ScatLambert2D_scatParams* scatLambert2DParamsPtr;
	Coating *oCoatingPtr;
	Coating_DiffRays *oCoating_DiffRaysPtr;
	Coating_NumCoeffs_DiffRays *oCoatingNumCoeffPtr;
	Coating_NumCoeffs_FullParams *coatingNumCoeffsParamsPtr;
	DiffRayField* DiffRayFieldPtr;
	DiffRayField_Freeform* DiffRayField_FreeformPtr;
	DiffRayField_RayAiming* DiffRayField_RayAimingPtr;
	DiffRayField_RayAiming_Holo* DiffRayField_RayAiming_HoloPtr;

	int materialListLength;
	int coatingIndex;
	int scatterIndex;

	// define structures to hold the parse results
	parseGlassResultStruct* parseResultsGlassPtr;
	parseGlassResultStruct* parseResultsGlassImmPtr;
	ParseGratingResultStruct* parseResultsGratingPtr;
	MatLinearGrating1D_DiffractionParams* gratDiffractionParamsPtr;
	MatRefracting_DispersionParams* glassDispersionParamsPtr;
	MatRefracting_DispersionParams* immersionDispersionParamsPtr;
	MatDiffracting_DispersionParams* matDiffractParamsPtr;
	MatDiffracting_DispersionParams* matDiffImmersionDispersionParamsPtr;
	Coating_DiffRays_FullParams* coatingParamsPtr;//=new Coating_FullParams();

	/* define handle to the grating file */
	FILE *hfileGrating;
	/* get handle to glass catalog */
	FILE *hfileGlass;
	double3 tilt;

	// loop through detector definitions
	for (k=0; k<parseResults->detectorNumber;k++)
	{
		switch (parseResults->detectorParams[k].detectorType)
		{
		case DET_RAYDATA:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_GLOBAL:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_RED:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_RED_GLOBAL:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_INTENSITY:
			DetectorPtr=new DetectorIntensity();
			DetectorIntensityPtr=dynamic_cast<DetectorIntensity*>(DetectorPtr);

			DetectorIntensityPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorIntensityPtr;
			break;

		case DET_FIELD:
			DetectorPtr=new DetectorField();
			DetectorFieldPtr=dynamic_cast<DetectorField*>(DetectorPtr);

			DetectorFieldPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorFieldPtr;
			break;

		case DET_PHASESPACE:
			DetectorPtr=new DetectorPhaseSpace();
			DetectorPhaseSpacePtr=dynamic_cast<DetectorPhaseSpace*>(DetectorPtr);

			DetectorPhaseSpacePtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorPhaseSpacePtr;
			break;

		default:
			// some error mechanism
			std::cout << "error in Parser.createSceneFromZemax(): unknown detector type" << std::endl;
			return false;
			break;
		} // end switch detector type
	} // end loop detector definitions


	// loop through source definitions
	for (k=0; k<parseResults->sourceNumber;k++)
	{
		// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
		if ( strcmp(parseResults->sourceParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->sourceParams[k].materialParams.immersionName,"STANDARD") )
		{
			/* get handle to the glass catalog file */
			sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
			hfileGlass = fopen( filepath, "r" );
			if (!hfileGlass)
			{
				std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
				fprintf( hfileDebug, "could not open glass file");
				return false;
			}
			/* parse Zemax glass catalog */
			if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
			{
				std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
				return false;
			}
		}
		else
		{
			parseResultsGlassPtr = NULL;
		}

		// parse importance area
		parseImpArea_Source(parseResults, k);

		// create geometric rayField with rayListLength according to the parse raynumber
		//*sourceListPtr[k] = new GeometricRayField(parseResults->sourceParams->height*parseResults->sourceParams->width);
		// create geometric rayField with rayListLength according to the GPUSubset dimensions
		switch (parseResults->sourceParams[k].type)
		{
		case DIFFSRC:
			//*sourceListPtr[k] = new DiffRayField(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX);
			*sourceListPtr[k] = new DiffRayField();
			DiffRayFieldPtr=dynamic_cast<DiffRayField*>(*sourceListPtr[k]);
			if (FIELD_NO_ERR != DiffRayFieldPtr->processParseResults(parseResults->sourceParams[k], parseResultsGlassPtr) )
			{
				std::cout <<"error in Parser.createSceneFromZemax(): DiffRayField.processParseResults() returned an error in geometry number: " << k << std::endl;
				return false;
			}
			break;
		case DIFFSRC_FREEFORM:
			//*sourceListPtr[k] = new DiffRayField_Freeform(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX);
			*sourceListPtr[k] = new DiffRayField_Freeform();
			DiffRayField_FreeformPtr=dynamic_cast<DiffRayField_Freeform*>(*sourceListPtr[k]);
			if (FIELD_NO_ERR != DiffRayField_FreeformPtr->processParseResults(parseResults->sourceParams[k], parseResultsGlassPtr) )
			{
				std::cout <<"error in Parser.createSceneFromZemax(): DiffRayField_Freeform.processParseResults() returned an error in geometry number: " << k << std::endl;
				return false;
			}
			break;
		case DIFFSRC_HOLO:
			//*sourceListPtr[k] = new DiffRayField_RayAiming_Holo(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX);
			*sourceListPtr[k] = new DiffRayField_RayAiming_Holo();
			DiffRayField_RayAiming_HoloPtr=dynamic_cast<DiffRayField_RayAiming_Holo*>(*sourceListPtr[k]);
			if (FIELD_NO_ERR != DiffRayField_RayAiming_HoloPtr->processParseResults(parseResults->sourceParams[k], parseResultsGlassPtr, parseResults->detectorParams[0]) )
			{
				std::cout <<"error in Parser.createSceneFromZemax(): DiffRayField_RayAiming_Holo.processParseResults() returned an error in geometry number: " << k << std::endl;
				return false;
			}
			break;
		case DIFFSRC_RAYAIM:
			//*sourceListPtr[k] = new DiffRayField_RayAiming(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX);
			*sourceListPtr[k] = new DiffRayField_RayAiming();
			DiffRayField_RayAimingPtr=dynamic_cast<DiffRayField_RayAiming*>(*sourceListPtr[k]);
			if (FIELD_NO_ERR != DiffRayField_RayAimingPtr->processParseResults(parseResults->sourceParams[k], parseResultsGlassPtr, parseResults->detectorParams[0]) )
			{
				std::cout <<"error in Parser.createSceneFromZemax(): DiffRayField_RayAimingPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
				return false;
			}
			break;
		default:
			std::cout <<"error in Parser.createSceneFromZemax(): unknown source type in source number: " << k << std::endl;
			return false;
			break;
		} // end switch source type

	} // end loop source definitions

	// loop through geometry definitions
	GeometryParseParamStruct test;
	for (k=0; k<parseResults->geomNumber;k++)
	{
		test=parseResults->geometryParams[k];
		// set indices for scatter and coating to zero. When scatter or coating is present, this value will be overwritten and can be used to flag presence of the respective material
		coatingIndex=0;
		scatterIndex=0;

		/* create geometry */
		switch(parseResults->geometryParams[k].type)
		{
			case (GEOM_COSINENORMAL):
				/* allocate memory for plane surface */
				oGeometryPtr = new SinusNormalSurface_DiffRays();
				oSinusNormalPtr=dynamic_cast<SinusNormalSurface_DiffRays*>(oGeometryPtr);

				if (GEOM_NO_ERR != oSinusNormalPtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oSinusNormalPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_IDEALLENSE):
				/* allocate memory for plane surface */
				oGeometryPtr = new IdealLense_DiffRays();
				oIdealLensePtr=dynamic_cast<IdealLense_DiffRays*>(oGeometryPtr);

				if (GEOM_NO_ERR != oIdealLensePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oIdealLensePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_PLANESURF):
				/* allocate memory for plane surface */
				oGeometryPtr = new PlaneSurface_DiffRays();
				oPlaneSurfacePtr=dynamic_cast<PlaneSurface_DiffRays*>(oGeometryPtr);

				if (GEOM_NO_ERR != oPlaneSurfacePtr->processParseResults(parseResults->geometryParams[k],k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oPlaneSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_APERTURESTOP):
				/* allocate memory for plane surface */
				oGeometryPtr = new ApertureStop_DiffRays();
				oApertureStopPtr=dynamic_cast<ApertureStop_DiffRays*>(oGeometryPtr);

				if (GEOM_NO_ERR != oApertureStopPtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oApertureStopPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_SPHERICALSURF):
				/* allocate memory for spherical surface */
				oGeometryPtr = new SphericalSurface_DiffRays();
				oSphericalSurfacePtr=dynamic_cast<SphericalSurface_DiffRays*>(oGeometryPtr);

				if (GEOM_NO_ERR != oSphericalSurfacePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oSphericalSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;
			case (GEOM_ASPHERICALSURF):
				/* allocate memory for aspherical surface */
				oGeometryPtr = new AsphericalSurface_DiffRays();
				oAsphericalSurfacePtr=dynamic_cast<AsphericalSurface_DiffRays*>(oGeometryPtr);

				if (GEOM_NO_ERR != oAsphericalSurfacePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oAsphericalSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;
			case (GEOM_CYLPIPE):
				/* allocate memory for cylindrcial pipe */
				oGeometryPtr = new CylPipe_DiffRays();
				oCylPipePtr=dynamic_cast<CylPipe_DiffRays*>(oGeometryPtr);

				if (GEOM_NO_ERR != oCylPipePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCylPipePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;
			case (GEOM_CONEPIPE):
				/* allocate memory for cone pipe */
				oGeometryPtr = new ConePipe_DiffRays();
				oConePipePtr=dynamic_cast<ConePipe_DiffRays*>(oGeometryPtr);

				if (GEOM_NO_ERR != oConePipePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oConePipePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			default:
				std::cout << "error in createSceneFromZemax(): unknown geometry in geometry number: " << k << std::endl;
				return false;

				break;
		} // end switch

		// set material list length
		if (oGeometryPtr->setMaterialListLength(1)!=GEOM_NO_ERR)
		{
			std::cout << "error in createSceneFromZemax(): setMaterialListLength(1) returned an error" << std::endl;
			return false;
		}
		/* create scatter */
		switch (parseResults->geometryParams[k].materialParams.scatterType)
		{
			case ST_TORRSPARR1D:
				oScatterTorrSparr1DPtr=new Scatter_TorranceSparrow1D_DiffRays(); 

				if (SCAT_NO_ERROR != oScatterTorrSparr1DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): scatTorrSparr1DParamsPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterTorrSparr1DPtr;

				break;
			case ST_DOUBLECAUCHY1D:
				oScatterDoubleCauchy1DPtr=new Scatter_DoubleCauchy1D_DiffRays(); 

				if (SCAT_NO_ERROR != oScatterDoubleCauchy1DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oScatterDoubleCauchy1DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterDoubleCauchy1DPtr;
				break;
			case ST_LAMBERT2D:
				oScatterLambert2DPtr=new Scatter_Lambert2D_DiffRays(); 

				if (SCAT_NO_ERROR != oScatterLambert2DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oScatterLambert2DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterLambert2DPtr;
				break;

			case ST_NOSCATTER:
				oScatter_DiffRaysPtr=new Scatter_DiffRays(); 
				scatterParamsPtr=new Scatter_DiffRays_Params();
				scatterParamsPtr->type=ST_NOSCATTER;
				if ( SCAT_NO_ERROR != oScatter_DiffRaysPtr->setFullParams(scatterParamsPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): Scatter.setFullParams() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				oScatterPtr=oScatter_DiffRaysPtr;
				break;

			default:
				oScatterPtr=new Scatter_DiffRays(); 
				scatterParamsPtr=new Scatter_DiffRays_Params();
				scatterParamsPtr->type=ST_NOSCATTER;
				oScatterPtr->setFullParams(scatterParamsPtr);
				std::cout << "warning in createSceneFromZemax(): unknown scatter in geometry number: " << k << ". No scatter assumed..." << std::endl;
				break;
		} // end switch materialParams.scatterType
		

		/* create coating */
		switch (parseResults->geometryParams[k].materialParams.coatingType)
		{
			case CT_NOCOATING:
				oCoating_DiffRaysPtr=new Coating_DiffRays(); 
				coatingParamsPtr=new Coating_DiffRays_FullParams();
				coatingParamsPtr->type=CT_NOCOATING;
				if ( COAT_NO_ERROR != oCoating_DiffRaysPtr->setFullParams(coatingParamsPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): Coating.setFullParams() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				oCoatingPtr=oCoating_DiffRaysPtr;
				break;

			case CT_NUMCOEFFS:
				oCoatingNumCoeffPtr=new Coating_NumCoeffs_DiffRays(); 

				if (SCAT_NO_ERROR != oCoatingNumCoeffPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCoatingNumCoeffPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oCoatingPtr=oCoatingNumCoeffPtr;
				break;

			default:
				oCoatingPtr=new Coating_DiffRays(); 
				coatingParamsPtr=new Coating_DiffRays_FullParams();
				coatingParamsPtr->type=CT_NOCOATING;
				oCoatingPtr->setFullParams(coatingParamsPtr);
				std::cout << "warning in createSceneFromZemax(): unknown coating in geometry number: " << k << ". No coating assumed..." << std::endl;
				break;
		} // end switch coating type

		/* create glass material */
		switch (parseResults->geometryParams[k].materialParams.matType)
		{
			case MT_DIFFRACT:
				oMaterialPtr = new MaterialDiffracting_DiffRays();
				oMaterialDiffractingPtr=dynamic_cast<MaterialDiffracting_DiffRays*>(oMaterialPtr);

				parseImpArea_Material( parseResults, k);

				// if we have an importance object defined, we read its aperture data and use it for our importance area
				// we need to do this here and not in the materials processParseResulst() function as we need to have acces to the aperture data of the other objects in the scene
				//if (parseResults->geometryParams[k].materialParams.importanceObjNr > 0 )
				//{
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].materialParams.apertureHalfWidth;
				//	parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].root;
				//	parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].tilt;
				//	parseResults->geometryParams[k].materialParams.importanceAreaApertureType=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].aperture;
				//}
				//else //if we have a cone angle defined, we calculate our importance area accordingly
				//{
				//	// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.x=(tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.x)-tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.x))/2;
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.y=(tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.y)-tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.y))/2;
				//	parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[k].root+parseResults->geometryParams[k].normal;
				//	parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[k].tilt;
				//}

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if (PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				if (MAT_NO_ERR != oMaterialDiffractingPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialDiffractingPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				break;

			case MT_MIRROR:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialReflecting_DiffRays();
				break;

			case MT_IDEALLENSE:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialIdealLense_DiffRays();
				oMaterialIdealLensePtr=dynamic_cast<MaterialIdealLense_DiffRays*>(oMaterialPtr);

				if (MAT_NO_ERR != oMaterialIdealLensePtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialIdealLensePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case MT_REFRMATERIAL:
				/* allocate memory for refracting surface */
				oMaterialPtr = new MaterialRefracting_DiffRays();
				oMaterialRefractingPtr=dynamic_cast<MaterialRefracting_DiffRays*>(oMaterialPtr);

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassImmPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassImmPtr = NULL;
				}
				// if we neither have a user defined nor a standard glass we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.glassName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.glassName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.glassName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				if (MAT_NO_ERR != oMaterialRefractingPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr, parseResultsGlassImmPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialRefractingPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}				
				break;

			case MT_ABSORB:
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialAbsorbing_DiffRays();

				break;
			case MT_LINGRAT1D:
				oMaterialPtr = new MaterialLinearGrating1D_DiffRays();
//				oMaterialPtr->setCoatingParams(coatingParamsPtr); // set coating parameters
				oMaterialLinearGrating1DPtr=dynamic_cast<MaterialLinearGrating1D_DiffRays*>(oMaterialPtr);

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"MIRROR") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassImmPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassImmPtr = NULL;
				}
				// if we neither have a user defined nor a standard glass we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.glassName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.glassName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.glassName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				// check wether we need to parse the grating file
				if (parseResults->geometryParams[k].materialParams.gratingEffsFromFile || parseResults->geometryParams[k].materialParams.gratingLinesFromFile || parseResults->geometryParams[k].materialParams.gratingOrdersFromFile)
				{
					// get handle to grating file
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "grating.TXT");
					hfileGrating = fopen( filepath, "r" ) ;
					if (!hfileGrating)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open grating file: " << filepath << ".  in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open grating file");
						return false;
					}
					/* parse MicroSim Grating Data */
					if (PARSER_NO_ERR != parseMicroSimGratingData(&parseResultsGratingPtr, hfileGrating) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseMicroSimGratingData() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}

				if (MAT_NO_ERR != oMaterialLinearGrating1DPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr, parseResultsGlassImmPtr, parseResultsGratingPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialLinearGrating1DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				
				break;
			default:
				std::cout <<"warning: no material found in geometry number: " << k << " absorbing material is assumed." << std::endl;
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialAbsorbing_DiffRays();
				break;
		} // end switch materialParams.matType

		// if we have the detector at hands, we set its material fix to absorbing. The material that is assigned to the detector in prescription file is only for the coating...
		if (k==parseResults->geomNumber-1)
		{
			// discard the material we created in the parsing
			free(oMaterialPtr);
			oMaterialPtr=new MaterialAbsorbing_DiffRays();
		}

		// fuse coating, scatter, material and geometry into on object
		if (MAT_NO_ERR != oMaterialPtr->setCoating(oCoatingPtr) ) // set coating parameters
		{
			std::cout <<"error in Parser.createSceneFromZemax(): material.setCoating() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		if (MAT_NO_ERR != oMaterialPtr->setScatter(oScatterPtr) ) // set coating parameters
		{
			std::cout <<"error in Parser.createSceneFromZemax(): material.setScatter() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		/* copy the pointer to the material. Note that we do not release the allocated memory */
		/* here. This will be taken care of in the destructor of the geometry!!          */
		if (GEOM_NO_ERR != oGeometryPtr->setMaterial(oMaterialPtr,0) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): Geometry.set;aterial() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		//oGeometryPtr->setID(k);
		oGeometryPtr->setComment(parseResults->geometryParams[k].comment);

		/* copy the pointer to the geometrygroup. Note that we do not release the allocated memory */
		/* here. This will be taken care of in the destructor of the geometryGroup!!          */
		if (GEOMGROUP_NO_ERR != oGroupPtr->getGeometryGroup(0)->setGeometry(oGeometryPtr, k) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): geometryGroup.setGeometry() returned an error in geometry number: " << k << std::endl;
			return false;
		}
	} // end: for (k=0; k<parseResults->geomNumber;k++)
	return true;
};



/**
 * \detail createGeometricSceneFromZemax 
 *
 * parses the prescription files and creates an OptiX scene for geometric raytracing
 *
 * \param[in] Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long long *sourceNumberPtr, Detector ***detListPtr, long long *detNumberPtr, TraceMode mode
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool createPathTracingSceneFromZemax(Group *oGroupPtr, parseResultStruct* parseResults, RayField ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, SimMode mode)
{
	/* output the geometry-data for debugging purpose */
	/* get handle to parser-debug file */
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "geometry.TXT");
	FILE *hfileDebug = fopen( filepath, "w" ) ;
	if (!hfileDebug)
	{
		std::cout <<"error in Parser.createSceneFromZemax(): cannot open description file: " << filepath << std::endl;
		return false;
	}
	/* set number of geometry groups */
	if (GROUP_NO_ERR != oGroupPtr->setGeometryGroupListLength(1) )
	{
		std::cout <<"error in Parser.createSceneFromZemax(): group.setGeometryGroupListLength(1) returned an error" << std::endl;
		return false;
	}
	/* create a geometryGroup inside the group object at index 0 */
	if (GROUP_NO_ERR != oGroupPtr->createGeometryGroup(0) )
	{
		std::cout <<"error in Parser.createSceneFromZemax(): group.createGeometryGroup(0) returned an error" << std::endl;
		return false;
	}
	/* set number of geometries */
	if (GEOMGROUP_NO_ERR != oGroupPtr->getGeometryGroup(0)->setGeometryListLength(parseResults->geomNumber-1) ) // detector is not a geometry in pathtracing
	{
		std::cout <<"error in Parser.createSceneFromZemax(): group.getGeometryGroup(0)->setGeometryListLength(number) returned an error with number = " << parseResults->geomNumber << std::endl;
		return false;
	}

	/* create array for the sources of our simulation */
	*sourceListPtr=new RayField*[parseResults->sourceNumber];
	*sourceNumberPtr=parseResults->sourceNumber;
	/* create array for the detectors of our simulation */
	*detListPtr=new Detector*[parseResults->detectorNumber];
	*detNumberPtr=parseResults->detectorNumber;

	int k=0;
	double theta;
	// define a pointer to every object that can be parsed
	Detector *DetectorPtr;
	DetectorRaydata *DetectorRaydataPtr;
	DetectorIntensity *DetectorIntensityPtr;
	DetectorField *DetectorFieldPtr;
	Geometry *oGeometryPtr;
	PlaneSurface *oPlaneSurfacePtr;
	PlaneSurface_Params planeParams;
	ApertureStop *oApertureStopPtr;
	ApertureStop_Params apertureStopParams;
	IdealLense *oIdealLensePtr;
	IdealLense_Params idealLenseParams;
	CylLenseSurface *oCylLenseSurfPtr;
	CylLenseSurface_Params cylLenseSurfParams;
	SinusNormalSurface *oSinusNormalPtr;
	SinusNormalSurface_Params sinusNormalParams;
	rayFieldParams *rayParamsPtr;
	detParams *detParamsPtr;
	detRaydataParams *detRaydataParamsPtr;
	detIntensityParams *detIntensityParamsPtr;
	detFieldParams *detFieldParamsPtr;
	CylPipe_Params cylParams;
	CylPipe *oCylPipePtr;
	ConePipe_Params coneParams;
	ConePipe *oConePipePtr;
	SphericalSurface *oSphericalSurfacePtr;
	SphericalSurface_Params sphereParams;
	AsphericalSurface *oAsphericalSurfacePtr;
	AsphericalSurface_Params asphereParams;
	Material *oMaterialPtr;
	MaterialDiffracting *oMaterialDiffractingPtr;
	MaterialReflecting *oMaterialReflectingPtr;
	MaterialReflecting_CovGlass *oMaterialReflecting_CovGlassPtr;
	MaterialRefracting *oMaterialRefractingPtr;
	MaterialRefracting_DiffRays *oMaterialRefracting_DiffRaysPtr;
	MaterialIdealLense *oMaterialIdealLensePtr;
	MaterialFilter *oMaterialFilterPtr;
	MaterialAbsorbing *oMaterialAbsorbingPtr;
	MaterialLinearGrating1D *oMaterialLinearGrating1DPtr;
	MaterialPathTraceSource *oMaterialPathTraceSourcePtr;
	MatRefracting_params refrParams;
	MatDiffracting_params diffParams;
	MatIdealLense_DispersionParams idealLenseMatParams;
	MatLinearGrating1D_params gratParams;
	MatPathTraceSource_params pathTraceSourceParams;
	ScatTorranceSparrow2D_PathTrace_params torrSparr2DParams;
	ScatLambert2D_params lambert2DParams;
	Scatter *oScatterPtr;
	Scatter_TorranceSparrow2D_PathTrace *oScatterTorrSparr2DPtr;
	Scatter_Lambert2D *oScatterLambert2DPtr;
	Scatter_Params *scatterParamsPtr;
	ScatLambert2D_scatParams* scatLambert2DParamsPtr;
	Coating *oCoatingPtr;
	Coating_NumCoeffs *oCoatingNumCoeffPtr;
	Coating_NumCoeffs_FullParams *coatingNumCoeffsParamsPtr;
	Coating_FresnelCoeffs *oCoatingFresnelCoeffPtr;
	Coating_FresnelCoeffs_FullParams *coatingFresnelCoeffsParamsPtr;
	PathTracingRayField* PathTracingRayFieldPtr;

	int materialListLength;
	int coatingIndex;
	int scatterIndex;

	// define structures to hold the parse results
	parseGlassResultStruct* parseResultsGlassPtr;
	parseGlassResultStruct* parseResultsGlassImmPtr;
	ParseGratingResultStruct* parseResultsGratingPtr;
	MatLinearGrating1D_DiffractionParams* gratDiffractionParamsPtr;
	MatRefracting_DispersionParams* glassDispersionParamsPtr;
	MatRefracting_DispersionParams* immersionDispersionParamsPtr;
	MatDiffracting_DispersionParams* matDiffractParamsPtr;
	MatDiffracting_DispersionParams* matDiffImmersionDispersionParamsPtr;
	Coating_FullParams* coatingParamsPtr;//=new Coating_FullParams();

	/* define handle to the grating file */
	FILE *hfileGrating;
	/* get handle to glass catalog */
	FILE *hfileGlass;
	double3 tilt;

	// loop through detector definitions
	for (k=0; k<parseResults->detectorNumber;k++)
	{
		// In contrast to all other sim modes, in path tracing mode the source is actually a geometry. Therefore we have to increment the importance object number...
		parseResults->detectorParams[k].importanceObjNr=parseResults->detectorParams[k].importanceObjNr+1;
		switch (parseResults->detectorParams[k].detectorType)
		{
		case DET_RAYDATA:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_GLOBAL:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_RED:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_RAYDATA_RED_GLOBAL:
			DetectorPtr=new DetectorRaydata();
			DetectorRaydataPtr=dynamic_cast<DetectorRaydata*>(DetectorPtr);

			DetectorRaydataPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorRaydataPtr;
			break;
		case DET_INTENSITY:
			DetectorPtr=new DetectorIntensity();
			DetectorIntensityPtr=dynamic_cast<DetectorIntensity*>(DetectorPtr);

			DetectorIntensityPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorIntensityPtr;
			break;

		case DET_FIELD:
			DetectorPtr=new DetectorField();
			DetectorFieldPtr=dynamic_cast<DetectorField*>(DetectorPtr);

			DetectorFieldPtr->processParseResults(parseResults->detectorParams[k]);

			*detListPtr[k]=DetectorFieldPtr;
			break;
			
		default:
			// some error mechanism
			std::cout << "error in Parser.createSceneFromZemax(): unknown detector type" << std::endl;
			return false;
			break;
		} // end switch detector type
	} // end loop detector definitions

	// now we need to create a source, that consists of the path-rays starting from each pixel of the detector
	//*sourceListPtr[0] = new PathTracingRayField(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX);
	*sourceListPtr[0] = new PathTracingRayField();
	PathTracingRayFieldPtr=dynamic_cast<PathTracingRayField*>(*sourceListPtr[0]);

	parseResultsGlassPtr = NULL;
	// parse importance area
	parseImpArea_Det(parseResults, 0);
	// create a FieldParseParamStruct that contains the information of our "detector-source" so we can use the standard processResults() 
	FieldParseParamStruct l_parseResults_DetSrc;
	l_parseResults_DetSrc.apertureHalfWidth1=parseResults->detectorParams[0].apertureHalfWidth;
	l_parseResults_DetSrc.coherence=parseResults->sourceParams[0].coherence;
	l_parseResults_DetSrc.lambda=parseResults->sourceParams[0].lambda;
	l_parseResults_DetSrc.normal=parseResults->detectorParams[0].normal;
	l_parseResults_DetSrc.root=parseResults->detectorParams[0].root;
	l_parseResults_DetSrc.tilt=parseResults->detectorParams[0].tilt;
	l_parseResults_DetSrc.rayDirDistr=parseResults->detectorParams[0].rayDirDistr;
	l_parseResults_DetSrc.rayPosDistr=parseResults->detectorParams[0].rayPosDistr;
	l_parseResults_DetSrc.rayDirection=make_double3(0,0,1);
	l_parseResults_DetSrc.width=parseResults->detectorParams[0].detPixel.x;
	l_parseResults_DetSrc.height=parseResults->detectorParams[0].detPixel.y;
	l_parseResults_DetSrc.widthLayout=parseResults->sourceParams[0].widthLayout;
	l_parseResults_DetSrc.heightLayout=parseResults->sourceParams[0].heightLayout;
	l_parseResults_DetSrc.power=1;
	l_parseResults_DetSrc.alphaMax=make_double2(0,0);
	l_parseResults_DetSrc.alphaMin=make_double2(0,0);
	l_parseResults_DetSrc.nrRayDirections=parseResults->detectorParams[0].nrRaysPerPixel;
	l_parseResults_DetSrc.rayDirDistr=parseResults->detectorParams[0].rayDirDistr;
	l_parseResults_DetSrc.rayPosDistr=parseResults->detectorParams[0].rayPosDistr;

	parseImpArea_Det(parseResults, 0);

	l_parseResults_DetSrc.importanceAreaHalfWidth=parseResults->detectorParams[0].importanceAreaHalfWidth;
	l_parseResults_DetSrc.importanceAreaRoot=parseResults->detectorParams[0].importanceAreaRoot;
	l_parseResults_DetSrc.importanceAreaTilt=parseResults->detectorParams[0].importanceAreaTilt;
	l_parseResults_DetSrc.importanceAreaApertureType=parseResults->detectorParams[0].importanceAreaApertureType;
	l_parseResults_DetSrc.importanceConeAlphaMax=parseResults->detectorParams[0].importanceConeAlphaMax;
	l_parseResults_DetSrc.importanceConeAlphaMin=parseResults->detectorParams[0].importanceConeAlphaMin;
	l_parseResults_DetSrc.importanceArea=parseResults->detectorParams[0].importanceArea;
	l_parseResults_DetSrc.importanceObjNr=parseResults->detectorParams[0].importanceObjNr;

	if (FIELD_NO_ERR != PathTracingRayFieldPtr->processParseResults(l_parseResults_DetSrc, parseResultsGlassPtr) )
	{
		std::cout <<"error in Parser.createSceneFromZemax(): GeomRayField.processParseResults() returned an error in geometry number: " << k << std::endl;
		return false;
	}

	// loop through source definitions
	for (k=0; k<parseResults->sourceNumber;k++)
	{
		// create geometric rayField with rayListLength according to the parse raynumber
		//*sourceListPtr[k] = new GeometricRayField(parseResults->sourceParams->height*parseResults->sourceParams->width);
		// create geometric rayField with rayListLength according to the GPUSubset dimensions
		//*sourceListPtr[k] = new GeometricRayField(GPU_SUBSET_WIDTH_MAX*GPU_SUBSET_HEIGHT_MAX);
		//GeomRayFieldPtr=dynamic_cast<GeometricRayField*>(*sourceListPtr[k]);

		//// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
		//if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
		//{
		//	/* get handle to the glass catalog file */
		//	sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
		//	hfileGlass = fopen( filepath, "r" );
		//	if (!hfileGlass)
		//	{
		//		std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
		//		fprintf( hfileDebug, "could not open glass file");
		//		return false;
		//	}
		//	/* parse Zemax glass catalog */
		//	if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
		//	{
		//		std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
		//		return false;
		//	}
		//}
		//else
		//{
		//	parseResultsGlassPtr = NULL;
		//}
		//// parse importance area
		//parseImpArea_Source(parseResults, k);

		//if (FIELD_NO_ERR != GeomRayFieldPtr->processParseResults(parseResults->sourceParams[k], parseResultsGlassPtr) )
		//{
		//	std::cout <<"error in Parser.createSceneFromZemax(): GeomRayField.processParseResults() returned an error in geometry number: " << k << std::endl;
		//	return false;
		//}

	} // end loop source definitions

	// loop through geometry definitions
	GeometryParseParamStruct test;
	for (k=0; k<parseResults->geomNumber-1;k++) // last geometry is the detector that is not considered as a geometry in pathtracing
	{
		// In contrast to all other sim modes, in path tracing mode the source is actually a geometry. Therefore we have to increment the importance object number...
		parseResults->geometryParams[k].materialParams.importanceObjNr=parseResults->geometryParams[k].materialParams.importanceObjNr+1;

		test=parseResults->geometryParams[k];
		// set indices for scatter and coating to zero. When scatter or coating is present, this value will be overwritten and can be used to flag presence of the respective material
		coatingIndex=0;
		scatterIndex=0;

		/* create geometry */
		switch(parseResults->geometryParams[k].type)
		{
			case (GEOM_COSINENORMAL):
				/* allocate memory for plane surface */
				oGeometryPtr = new SinusNormalSurface();
				oSinusNormalPtr=dynamic_cast<SinusNormalSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oSinusNormalPtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oSinusNormalPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_IDEALLENSE):
				/* allocate memory for plane surface */
				oGeometryPtr = new IdealLense();
				oIdealLensePtr=dynamic_cast<IdealLense*>(oGeometryPtr);

				if (GEOM_NO_ERR != oIdealLensePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oIdealLensePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_PLANESURF):
				/* allocate memory for plane surface */
				oGeometryPtr = new PlaneSurface();
				oPlaneSurfacePtr=dynamic_cast<PlaneSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oPlaneSurfacePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oPlaneSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_APERTURESTOP):
				/* allocate memory for plane surface */
				oGeometryPtr = new ApertureStop();
				oApertureStopPtr=dynamic_cast<ApertureStop*>(oGeometryPtr);

				if (GEOM_NO_ERR != oApertureStopPtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oApertureStopPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_SPHERICALSURF):
				/* allocate memory for spherical surface */
				oGeometryPtr = new SphericalSurface();
				oSphericalSurfacePtr=dynamic_cast<SphericalSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oSphericalSurfacePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oSphericalSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_CYLLENSESURF):
				/* allocate memory for spherical surface */
				oGeometryPtr = new CylLenseSurface();
				oCylLenseSurfPtr=dynamic_cast<CylLenseSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oCylLenseSurfPtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCylLenseSurfPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case (GEOM_ASPHERICALSURF):
				/* allocate memory for aspherical surface */
				oGeometryPtr = new AsphericalSurface();
				oAsphericalSurfacePtr=dynamic_cast<AsphericalSurface*>(oGeometryPtr);

				if (GEOM_NO_ERR != oAsphericalSurfacePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oAsphericalSurfacePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;
			case (GEOM_CYLPIPE):
				/* allocate memory for cylindrcial pipe */
				oGeometryPtr = new CylPipe();
				oCylPipePtr=dynamic_cast<CylPipe*>(oGeometryPtr);

				if (GEOM_NO_ERR != oCylPipePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCylPipePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;
			case (GEOM_CONEPIPE):
				/* allocate memory for cone pipe */
				oGeometryPtr = new ConePipe();
				oConePipePtr=dynamic_cast<ConePipe*>(oGeometryPtr);

				if (GEOM_NO_ERR != oConePipePtr->processParseResults(parseResults->geometryParams[k], k+1) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oConePipePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			default:
				std::cout << "error in createSceneFromZemax(): unknown geometry in geometry number: " << k << std::endl;
				return false;

				break;
		} // end switch

		// set material list length. so far we set it const to one!!
		if (oGeometryPtr->setMaterialListLength(1)!=GEOM_NO_ERR)
		{
			std::cout << "error in createSceneFromZemax(): setMaterialListLength(1) returned an error" << std::endl;
			return false;
		}
		/* create scatter */
		switch (parseResults->geometryParams[k].materialParams.scatterType)
		{

			case ST_TORRSPARR2D:
				oScatterTorrSparr2DPtr=new Scatter_TorranceSparrow2D_PathTrace(); 
				parseImpArea_Material(parseResults,k);
				// set source area
				parseResults->geometryParams[k].materialParams.srcAreaHalfWidth=parseResults->pupilParams[0].apertureHalfWidth;
				parseResults->geometryParams[k].materialParams.srcAreaRoot=parseResults->pupilParams[0].root;
				parseResults->geometryParams[k].materialParams.srcAreaTilt=parseResults->pupilParams[0].tilt;
				parseResults->geometryParams[k].materialParams.srcAreaType=parseResults->pupilParams[0].apertureType;

				if (SCAT_NO_ERROR != oScatterTorrSparr2DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): scatTorrSparr2DParamsPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterTorrSparr2DPtr;

				break;
			case ST_LAMBERT2D:
				oScatterLambert2DPtr=new Scatter_Lambert2D(); 
				parseImpArea_Material(parseResults, parseResults->geometryParams[k].materialParams.importanceObjNr); 
				if (SCAT_NO_ERROR != oScatterLambert2DPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oScatterLambert2DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oScatterPtr=oScatterLambert2DPtr;
				break;

			case ST_NOSCATTER:
				oScatterPtr=new Scatter(); 
				scatterParamsPtr=new Scatter_Params();
				scatterParamsPtr->type=ST_NOSCATTER;
				if ( SCAT_NO_ERROR != oScatterPtr->setFullParams(scatterParamsPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): Scatter.setFullParams() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				break;

			default:
				oScatterPtr=new Scatter(); 
				scatterParamsPtr=new Scatter_Params();
				scatterParamsPtr->type=ST_NOSCATTER;
				oScatterPtr->setFullParams(scatterParamsPtr);
				std::cout << "warning in createSceneFromZemax(): unknown scatter in geometry number: " << k << ". No scatter assumed..." << std::endl;
				break;
		} // end switch materialParams.scatterType
		

		/* create coating */
		switch (parseResults->geometryParams[k].materialParams.coatingType)
		{
			case CT_NOCOATING:
				oCoatingPtr=new Coating(); 
				coatingParamsPtr=new Coating_FullParams();
				coatingParamsPtr->type=CT_NOCOATING;
				if ( COAT_NO_ERROR != oCoatingPtr->setFullParams(coatingParamsPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): Coating.setFullParams() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				break;

			case CT_NUMCOEFFS:
				oCoatingNumCoeffPtr=new Coating_NumCoeffs(); 

				if (SCAT_NO_ERROR != oCoatingNumCoeffPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCoatingNumCoeffPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oCoatingPtr=oCoatingNumCoeffPtr;
				break;

			case CT_FRESNELCOEFFS:
				oCoatingFresnelCoeffPtr=new Coating_FresnelCoeffs(); 
				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassImmPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassImmPtr = NULL;
				}
				// if we neither have a user defined nor a standard glass we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.glassName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.glassName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.glassName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				if (SCAT_NO_ERROR != oCoatingFresnelCoeffPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr, parseResultsGlassPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oCoatingFresnelCoeffPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				oCoatingPtr=oCoatingFresnelCoeffPtr;
				break;


			default:
				oCoatingPtr=new Coating(); 
				coatingParamsPtr=new Coating_FullParams();
				coatingParamsPtr->type=CT_NOCOATING;
				oCoatingPtr->setFullParams(coatingParamsPtr);
				std::cout << "warning in createSceneFromZemax(): unknown coating in geometry number: " << k << ". No coating assumed..." << std::endl;
				break;
		} // end switch coating type

		/* create glass material */
		switch (parseResults->geometryParams[k].materialParams.matType)
		{
			case MT_DIFFRACT:
				oMaterialPtr = new MaterialDiffracting();
				oMaterialDiffractingPtr=dynamic_cast<MaterialDiffracting*>(oMaterialPtr);

				parseImpArea_Material( parseResults, k);

				// if we have an importance object defined, we read its aperture data and use it for our importance area
				// we need to do this here and not in the materials processParseResulst() function as we need to have acces to the aperture data of the other objects in the scene
				//if (parseResults->geometryParams[k].materialParams.importanceObjNr > 0 )
				//{
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].materialParams.apertureHalfWidth;
				//	parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].root;
				//	parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].tilt;
				//	parseResults->geometryParams[k].materialParams.importanceAreaApertureType=parseResults->geometryParams[parseResults->geometryParams[k].materialParams.importanceObjNr].aperture;
				//}
				//else //if we have a cone angle defined, we calculate our importance area accordingly
				//{
				//	// we create an importance area that is 1mm in front of the surface. We orient this area parallel to the surface and calculate its half width according to the angle cone
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.x=(tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.x)-tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.x))/2;
				//	parseResults->geometryParams[k].materialParams.importanceAreaHalfWidth.y=(tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMax.y)-tan(parseResults->geometryParams[k].materialParams.importanceConeAlphaMin.y))/2;
				//	parseResults->geometryParams[k].materialParams.importanceAreaRoot=parseResults->geometryParams[k].root+parseResults->geometryParams[k].normal;
				//	parseResults->geometryParams[k].materialParams.importanceAreaTilt=parseResults->geometryParams[k].tilt;
				//}

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				if (MAT_NO_ERR != oMaterialDiffractingPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialDiffractingPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case MT_MIRROR:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialReflecting();
				break;

			case MT_IDEALLENSE:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialIdealLense();
				oMaterialIdealLensePtr=dynamic_cast<MaterialIdealLense*>(oMaterialPtr);

				if (MAT_NO_ERR != oMaterialIdealLensePtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialIdealLensePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case MT_FILTER:
				/* allocate memory for reflecting surface */
				oMaterialPtr = new MaterialFilter();
				oMaterialFilterPtr=dynamic_cast<MaterialFilter*>(oMaterialPtr);

				if (MAT_NO_ERR != oMaterialFilterPtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialFilterPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}

				break;

			case MT_REFRMATERIAL:
				/* allocate memory for refracting surface */
				oMaterialPtr = new MaterialRefracting();
				oMaterialRefractingPtr=dynamic_cast<MaterialRefracting*>(oMaterialPtr);

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassImmPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassImmPtr = NULL;
				}
				// if we neither have a user defined nor a standard glass we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.glassName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.glassName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.glassName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				if (MAT_NO_ERR != oMaterialRefractingPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr, parseResultsGlassImmPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialRefractingPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				
				break;

			case MT_ABSORB:
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialAbsorbing();
				break;

			case MT_COVGLASS:
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialReflecting_CovGlass();
				oMaterialReflecting_CovGlassPtr=dynamic_cast<MaterialReflecting_CovGlass*>(oMaterialPtr);
				if (MAT_NO_ERR != oMaterialReflecting_CovGlassPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResults->detectorParams[0]) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialReflecting_CovGlassPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				else
				{
					// as we handle reflection coefficient directly in the material we delete the previous defined coating and set it to nocoating
					free(oCoatingPtr);
					oCoatingPtr=new Coating(); 
					coatingParamsPtr=new Coating_FullParams();
					coatingParamsPtr->type=CT_NOCOATING;
					if ( COAT_NO_ERROR != oCoatingPtr->setFullParams(coatingParamsPtr) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): Coating.setFullParams() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}

				break;

			case MT_PATHTRACESRC:
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialPathTraceSource();
				oMaterialPathTraceSourcePtr=dynamic_cast<MaterialPathTraceSource*>(oMaterialPtr);
				if (MAT_NO_ERR != oMaterialPathTraceSourcePtr->processParseResults(parseResults->geometryParams[k].materialParams) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialPathTraceSourcePtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}
				else
				{
					// as we handle reflection coefficient directly in the material we delete the previous defined coating and set it to nocoating
					free(oCoatingPtr);
					oCoatingPtr=new Coating(); 
					coatingParamsPtr=new Coating_FullParams();
					coatingParamsPtr->type=CT_NOCOATING;
					if ( COAT_NO_ERROR != oCoatingPtr->setFullParams(coatingParamsPtr) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): Coating.setFullParams() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}

				break;

			case MT_LINGRAT1D:
				oMaterialPtr = new MaterialLinearGrating1D();
//				oMaterialPtr->setCoatingParams(coatingParamsPtr); // set coating parameters
				oMaterialLinearGrating1DPtr=dynamic_cast<MaterialLinearGrating1D*>(oMaterialPtr);

				// if we neither have a user defined nor a standard immersion medium we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.immersionName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.immersionName,"STANDARD") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassImmPtr, hfileGlass, parseResults->geometryParams[k].materialParams.immersionName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassImmPtr = NULL;
				}
				// if we neither have a user defined nor a standard glass we have to parse for it in the glass library
				if ( strcmp(parseResults->geometryParams[k].materialParams.glassName,"USERDEFINED") && strcmp(parseResults->geometryParams[k].materialParams.glassName,"MIRROR") )
				{
					/* get handle to the glass catalog file */
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "glass.AGF");
					hfileGlass = fopen( filepath, "r" );
					if (!hfileGlass)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open glass catalog " << filepath << ". in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open glass file");
						return false;
					}
					/* parse Zemax glass catalog */
					if ( PARSER_NO_ERR != parseZemaxGlassCatalogOld(&parseResultsGlassPtr, hfileGlass, parseResults->geometryParams[k].materialParams.glassName) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseZemaxGlassCatalogOld() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}
				else
				{
					parseResultsGlassPtr = NULL;
				}

				// check wether we need to parse the grating file
				if (parseResults->geometryParams[k].materialParams.gratingEffsFromFile || parseResults->geometryParams[k].materialParams.gratingLinesFromFile || parseResults->geometryParams[k].materialParams.gratingOrdersFromFile)
				{
					// get handle to grating file
					sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "grating.TXT");
					hfileGrating = fopen( filepath, "r" ) ;
					if (!hfileGrating)
					{
						std::cout <<"error in Parser.createSceneFromZemax(): could not open grating file: " << filepath << ".  in geometry number: " << k << std::endl;
						fprintf( hfileDebug, "could not open grating file");
						return false;
					}
					/* parse MicroSim Grating Data */
					if (PARSER_NO_ERR != parseMicroSimGratingData(&parseResultsGratingPtr, hfileGrating) )
					{
						std::cout <<"error in Parser.createSceneFromZemax(): parseMicroSimGratingData() returned an error in geometry number: " << k << std::endl;
						return false;
					}
				}

				if (MAT_NO_ERR != oMaterialLinearGrating1DPtr->processParseResults(parseResults->geometryParams[k].materialParams, parseResultsGlassPtr, parseResultsGlassImmPtr, parseResultsGratingPtr) )
				{
					std::cout <<"error in Parser.createSceneFromZemax(): oMaterialLinearGrating1DPtr.processParseResults() returned an error in geometry number: " << k << std::endl;
					return false;
				}				
				break;
			default:
				std::cout <<"warning: no material found in geometry number: " << k << " absorbing material is assumed." << std::endl;
				/* allocate memory for absorbing surface */
				oMaterialPtr = new MaterialAbsorbing();
				break;
		} // end switch materialParams.matType

		// if we have the detector at hands, we set its material fix to absorbing. The material that is assigned to the detector in prescription file is only for the coating...
		if (k==parseResults->geomNumber-1)
		{
			// discard the material we created in the parsing
			free(oMaterialPtr);
			oMaterialPtr=new MaterialAbsorbing();
		}
		// fuse coating, scatter, material and geometry into on object
		if (MAT_NO_ERR != oMaterialPtr->setCoating(oCoatingPtr) ) // set coating parameters
		{
			std::cout <<"error in Parser.createSceneFromZemax(): material.setCoating() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		if (MAT_NO_ERR != oMaterialPtr->setScatter(oScatterPtr) ) // set coating parameters
		{
			std::cout <<"error in Parser.createSceneFromZemax(): material.setScatter() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		/* copy the pointer to the material. Note that we do not release the allocated memory */
		/* here. This will be taken care of in the destructor of the geometry!!          */
		if (GEOM_NO_ERR != oGeometryPtr->setMaterial(oMaterialPtr,0) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): Geometry.set;aterial() returned an error in geometry number: " << k << std::endl;
			return false;
		}
		//oGeometryPtr->setID(k);
		oGeometryPtr->setComment(parseResults->geometryParams[k].comment);

		/* copy the pointer to the geometrygroup. Note that we do not release the allocated memory */
		/* here. This will be taken care of in the destructor of the geometryGroup!!          */
		if (GEOMGROUP_NO_ERR != oGroupPtr->getGeometryGroup(0)->setGeometry(oGeometryPtr, k) )
		{
			std::cout <<"error in Parser.createSceneFromZemax(): geometryGroup.setGeometry() returned an error in geometry number: " << k << std::endl;
			return false;
		}
	} // end: for (k=0; k<parseResults->geomNumber;k++)
	return true;
};