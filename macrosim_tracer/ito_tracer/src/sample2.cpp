#define _CRTDBG_MAP_ALLOC
/**\file sample2.cpp
* \brief entry point to the application containing the main routine
* 
*           
* \author Mauch
*/

//#include <omp.h>
#include "TopObject.h"
#include "cuda_runtime.h"
#include <optix.h>
#include <vector_functions.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>
#include <sampleConfig.h>
//#include <commonStructs.h>
#include "Group.h"
#include "Geometry.h"
#include "GeometryLib.h"
#include "MaterialLib.h"
#include "ScatterLib.h"
#include "CoatingLib.h"
#include "SimAssistantLib.h"
#include "DetectorLib.h"
//#include "Detector.h"
#include "Scatter.h"
#include "Coating.h"
#include "RayField.h"
#include "GeometricRayField.h"
#include <math.h>
#include "rayTracingMath.h"
#include <ctime>
#include "FlexZemax.h"
#include "GlobalConstants.h"
#include "DiffRayField.h"
#include "complex.h"
#include "Converter.h"
#include "ScalarLightField.h"
#include "VectorLightField.h"
#include "wavefrontIn.h"
#include "inputOutput.h"
#include "GlobalConstants.h"
#include "Parser.h"
#include "Parser_XML.h"

#include "PropagationMath.h"
#include "Interpolator.h"

//#include <fstream>
#include <iostream>
using namespace std;

//#pragma comment( lib, "Delayimp.lib" )
//#pragma comment( lib, "cuda.lib" )
//#pragma comment( lib, "cudart.lib" )
//#pragma comment( lib, "optix.1.lib" )


//#pragma comment(lib, "testdll.lib")
//// Importieren der Funktion aus der oben erstellten DLL
//extern "C" __declspec(dllimport)double testFunc (double a, double b);

/* define the size of ray structure */
unsigned long long width;
unsigned long long height;

/*definition of global constants*/
//constants for raytracing on CPU
int MAX_DEPTH_CPU=10;
const float MIN_FLUX_CPU=1e-8;
const double MAX_TOLERANCE=1e-10;

typedef struct struct_BoxExtent{
    float min[3];
    float max[3];
} BoxExtent;

// declare scene group
Group oGroup;

void testInterpolator();
void createContext( RTcontext* context, RTbuffer* buffer );
void printUsageAndExit( const char* argv0 );
//void doCPUTracing(Group &oGroup, GeometricRayField &oGeomRayField);
//void doCPUTracing(Group &oGroup, DiffRayField &oDiffRayField);
//void doCPUTracing(Group &oGroup, GaussBeamRayField &oGaussBeamRayField);
unsigned int createGaussianBeamRaysTest(GaussBeamRayField **ptrptrRayField, double2 width, double2 offset, long2 nrBeamlets, double lambda);
void initRayField_AsphereTestCPU(FILE *hfileQxy, FILE *hfilePxy, GeometricRayField* oGemRayFieldPtr, double RadiusSourceReference, double zSourceReference, double *MNmn, int width, int height, double lambda);
void initRayField_AsphereTestGPU(RTcontext &context, FILE *hfileQxy, FILE *hfilePxy, GeometricRayField* oGeomRayFieldPtr, double RadiusSourceReference, double zSourceReference, double *MNmn, int width, int height, double lambda);
bool doTheSimulation(Group *oGrouPtr, RayField **SourceListPtrPtr, bool RunOnCPU);
bool createSceneFromXML(Group *oGroupPtr, FILE *hfile, RayField ***sourceListPtr, Detector ***detListPtr);


void wait ()
{
    // Löscht etwaige Fehlerzustände die das Einlesen verhindern könnten
    cin.clear();
    // Ignoriert soviele Zeichen im Puffer wie im Puffer vorhanden sind
    // (= ignoriert alle Zeichen die derzeit im Puffer sind)
    cin.ignore(cin.rdbuf()->in_avail()); 
    // Füge alle eingelesenen Zeichen in den Puffer bis ein Enter gedrückt wird
    // cin.get() liefert dann das erste Zeichen aus dem Puffer zurück, welches wir aber ignorieren (interessiert uns ja nicht)
    cin.get();
} 

int main(int argc, char* argv[])
{

//	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	clock_t start, end;
	start=clock();

#ifdef _DEBUG
#else
	if (argc <4)
	{
		cout << "error in main: too few arguments." << endl;
		return (1);
	}
#endif

//	omp_set_num_threads(omp_get_max_threads()-1);

//	#pragma comment( lib, "Delayimp.lib" )
//	#pragma comment( lib, "cuda.lib" )
//	#pragma comment( lib, "cudart.lib" )
//	#pragma comment( lib, "optix.1.lib" )

	/* create scene */

	/*******************************************************/
	/* create geometry from parsing Zemax description file */
	/*******************************************************/

	//char *test1, *test2;
	/* get handle to Zemax description file */
	//char filepath[512];
	//char filepath;
	char* filepath=(char*)malloc(512*sizeof(char));


#ifdef _DEBUG
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILES_PATH, "BeamHomo_PhaseSpace.TXT");//prescription_Avinash_PathTrace.TXT");//PhaseSpaceTest_Stabmischer1D_CosineProfile.TXT");//DiffTest_f9mm_idealLense.TXT");//fiber_interferometer_opt50x05_in.TXT");//prescription_Geometry_test.TXT");//prescription_nonsequential_OptiX_FromSlit_CompleteHousing_Filter.TXT");//prescription_nonsequential_test.TXT");//prescription_nonsequential_OptiX_FromSlit_Housing.TXT");//OptiX_FromSlit_CompleteHousing.TXT");//scatMat.TXT");//test.TXT");//Sidelobe380.TXT");//CMIdeal1.TXT");//
	MAX_DEPTH_CPU=3;
#else
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILES_PATH, argv[1]);
	MAX_DEPTH_CPU=atoi(argv[3]);
#endif
	
	FILE *hfile = fopen( filepath, "r" ) ;
	if (!hfile)
	{
		cout << "error in main: cannot open geometry prescription file:" << filepath << endl;
		return (1);
	}
	
	// decide wether we want to do sequential or nonsequential simulations
	simMode mode=SIM_GEOMRAYS_NONSEQ;

	RayField** SourceList;
	SourceList=NULL;
	Detector** DetectorList;
	DetectorList=NULL;
	long sourceNumber, detNumber;

	FILE *hfile_xml = fopen( "MyXml.xml", "r" ) ;
	if (!hfile)
	{
		cout << "error in main: cannot open xml file:" << filepath << endl;
		return (1);
	}

	createSceneFromXML(&oGroup, hfile_xml, &SourceList, &DetectorList);

	oGroup.getGeometryGroup(0)->getGeometry(0)->getParamsPtr();

	if (!createSceneFromZemax(&oGroup, hfile, &SourceList, &sourceNumber, &DetectorList, &detNumber, mode))
	{
		cout << "error in main: createSceneFromZemax returned an error" << endl;
		fclose(hfile); // close file anyway
		return (1);
	}
	fclose(hfile);

	bool layoutMode;
#ifdef _DEBUG
	layoutMode=true;
#else
	// set wavelength from command line
	if (argc>4)
	{
		SourceList[0]->setLambda(atof(argv[4])*1e-6);
		cout << "lambda set from command line to: " << SourceList[0]->getParamsPtr()->lambda*1e6 << "nm" << endl;
	}
	if (argc>5)
	{
		if (atoi(argv[5]) == 1)
			layoutMode=true;
		else
			layoutMode=false;
	}
	else
		layoutMode=false;
#endif

	/****************************************************************/
	/* trace rays                                                   */
	/****************************************************************/

	simAssParams *oSimAssParamsPtr=NULL;
	SimAssistant *oSimAssPtr=NULL;
	if (!layoutMode)
	{
		oSimAssPtr = new SimAssistantSingleSim;
		oSimAssParamsPtr = new simAssSingleSimParams();
	}
	else
	{
		oSimAssPtr = new SimAssistantLayout;
		oSimAssParamsPtr = new simAssLayoutParams();
	}
	oSimAssParamsPtr->mode=mode;
	

#ifdef _DEBUG
	oSimAssParamsPtr->RunOnCPU=true;
#else
	if (atoi(argv[2])==0)
		oSimAssParamsPtr->RunOnCPU=false;
	else
		oSimAssParamsPtr->RunOnCPU=true;
#endif

	oSimAssPtr->setParamsPtr(oSimAssParamsPtr);


	//simAssParamSweepParams *oSimAssParamsPtr = new simAssParamSweepParams();

	//oSimAssParamsPtr->geomParamsSweepLength=1;
	//oSimAssParamsPtr->geomObjectIndex=2;
	//oSimAssParamsPtr->geometryParamsList=new Geometry_Params*[oSimAssParamsPtr->geomParamsSweepLength];

	//PlaneSurface_Params *planeParamsPtr=new PlaneSurface_Params();
	//planeParamsPtr->normal=make_double3(0,0,1);
	//planeParamsPtr->root=make_double3(0,0,241.8);
	//planeParamsPtr->apertureType=AT_ELLIPT;
	//planeParamsPtr->apertureRadius=make_double2(15,15);
	//planeParamsPtr->rotNormal=0;
	//planeParamsPtr->geometryID=oSimAssParamsPtr->geomObjectIndex+1;
	//oSimAssParamsPtr->geometryParamsList[0]=planeParamsPtr;

	//oSimAssParamsPtr->detParamsSweepLength=0;
	//oSimAssParamsPtr->srcParamsSweepLength=0;
	//oSimAssParamsPtr->mode=SIM_GEOMRAYS_NONSEQ;
	//oSimAssParamsPtr->RunOnCPU=true;

	//SimAssistantParamSweep oSimAss(oSimAssParamsPtr);
	
	if (SIMASS_NO_ERROR != oSimAssPtr->initSimulation(&oGroup, SourceList))
	{
		cout << "error in main(): SimAss.initSimulation() returned an error" << endl;
		return( 1 );
	}
	if (SIMASS_NO_ERROR != oSimAssPtr->run(&oGroup, SourceList, DetectorList))
	{
		cout << "error in main(): SimAss.run() returned an error" << endl;
		return( 1 );
	}

	end=clock();
	double msecs;
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	cout << endl;
	cout << msecs <<"ms to run the whole simulation." << endl;

	if (layoutMode)
	{
		// wait for user to hit enter to end the program
		wait();
	}

	/************************************************************/
	/*						Clean Up							*/
	/************************************************************/

	if (SourceList != NULL)
	{
		for (int j=0; j<sourceNumber;j++)
		{
			delete SourceList[j];
			SourceList[j]=NULL;
		}
		delete SourceList;
		SourceList=NULL;
	}

	if (DetectorList != NULL)
	{
		for (int j=0; j<detNumber;j++)
		{
			delete (DetectorList[j]);
			DetectorList[j]=NULL;
		}
		delete DetectorList;
		DetectorList=NULL;
	}

	if (oSimAssPtr != NULL)
	{
		delete oSimAssPtr;
		oSimAssPtr=NULL;
	}

	return( 0 );
}

bool runSimulation(char *xmlInput, Field** fieldOut_ptrptr, bool *breakCondition)
{
	return true;
}

void printUsageAndExit( const char* argv0 )
{
  fprintf( stderr, "Usage  : %s [options]\n", argv0 );
  fprintf( stderr, "Options: --help | -h             Print this usage message\n" );
  fprintf( stderr, "         --dim=<width>x<height>  Set image dimensions; defaults to 512x384\n" );
  exit(0);
}


unsigned int createGaussianBeamRaysTest(GaussBeamRayField **ptrptrRayField, double2 width, double2 offsetIn, long2 nrBeamlets, double lambda)
{
	unsigned long long ix=0;
	unsigned long long iy=0;
	if (*ptrptrRayField != NULL)
	{
		cout <<"error in createGaussianBeamRaysTest(): ptrptrRayField is not NULL." << endl;
		return 0; // return error
	}
	*ptrptrRayField = new GaussBeamRayField(nrBeamlets.x*nrBeamlets.y);
	gaussBeamRayStruct *rayList=(*(ptrptrRayField))->getRayList();
	double2 offset=offsetIn;

			//// waist ray in x
			//rayList[0].waistRayX.position=make_double3(-750,0,500);
			//rayList[0].waistRayX.direction=make_double3(0,0,1);
			//// waist ray in y
			//rayList[0].waistRayY.position=make_double3(0,-750,500);
			//rayList[0].waistRayY.direction=make_double3(0,0,1);
			//// div Ray in y
			//rayList[0].divRayY.position=make_double3(0,-2.1220659078919,500);
			//rayList[0].divRayY.direction=make_double3(0,-0.004244093592259,0.999990993794234);
			//// div Ray in x
			//rayList[0].divRayX.position=make_double3(-2.1220659078919,0,500);
			//rayList[0].divRayX.direction=make_double3(-0.004244093592259,0,0.999990993794234);
			//// base ray
			//rayList[0].baseRay.position=make_double3(0,0,500);
			//rayList[0].baseRay.direction=make_double3(0,0,1);
			//rayList[0].baseRay.lambda=lambda;
			//rayList[0].baseRay.flux=100000000000;
			//rayList[0].baseRay.depth=0;
			//rayList[0].baseRay.nImmersed=1;
			//rayList[0].baseRay.currentGeometryID=0;
			//rayList[0].baseRay.opl=500;


	for (ix=0;ix<nrBeamlets.x;ix++)
	{
		for (iy=0;iy<nrBeamlets.y;iy++)
		{
			double z=0;
			double3 r0;
			double2 w0;
			if (nrBeamlets.x==1)
			{
				r0.x=offset.x;
				w0.x=width.x;
			}
			else
			{
				r0.x=-width.x/2+width.x/nrBeamlets.x*ix+offset.x;
				w0.x=width.x/(nrBeamlets.x-1)*1.6;
			}
			if (nrBeamlets.y==1)
			{
				r0.y=offset.y;
				w0.y=width.y;
			}
			else
			{
				r0.y=-width.y/2+width.y/nrBeamlets.y*iy+offset.y;
				w0.y=width.y/(nrBeamlets.y-1)*1.6;
			}
			r0.z=z;
			// create geometric rays representing the beamlets
			// waist ray in x
			rayList[ix+nrBeamlets.x*iy].waistRayX.position=make_double3((r0.x-w0.x),r0.y,r0.z);
			rayList[ix+nrBeamlets.x*iy].waistRayX.direction=make_double3(0,0,1);
			// waist ray in y
			rayList[ix+nrBeamlets.x*iy].waistRayY.position=make_double3(r0.x,(r0.y-w0.y),r0.z);
			rayList[ix+nrBeamlets.x*iy].waistRayY.direction=make_double3(0,0,1);
			// div Ray in y
			rayList[ix+nrBeamlets.x*iy].divRayY.position=r0;
			rayList[ix+nrBeamlets.x*iy].divRayY.direction.x=0;
			rayList[ix+nrBeamlets.x*iy].divRayY.direction.y=cos(PI/2+atan(lambda/(PI*w0.y)));
			rayList[ix+nrBeamlets.x*iy].divRayY.direction.z=sqrt(1-pow(rayList[ix+nrBeamlets.x*iy].divRayY.direction.y,2));
			// div Ray in x
			rayList[ix+nrBeamlets.x*iy].divRayX.position=r0;
			rayList[ix+nrBeamlets.x*iy].divRayX.direction.x=cos(PI/2+atan(lambda/(PI*w0.x)));
			rayList[ix+nrBeamlets.x*iy].divRayX.direction.y=0;
			rayList[ix+nrBeamlets.x*iy].divRayX.direction.z=sqrt(1-pow(rayList[ix+nrBeamlets.x*iy].divRayX.direction.x,2));
			// base ray
			rayList[ix+nrBeamlets.x*iy].baseRay.position=r0;
			rayList[ix+nrBeamlets.x*iy].baseRay.direction=make_double3(0,0,1);
			rayList[ix+nrBeamlets.x*iy].baseRay.lambda=lambda;
			rayList[ix+nrBeamlets.x*iy].baseRay.flux=1;
			rayList[ix+nrBeamlets.x*iy].baseRay.depth=0;
			rayList[ix+nrBeamlets.x*iy].baseRay.nImmersed=1;
			rayList[ix+nrBeamlets.x*iy].baseRay.currentGeometryID=0;
			rayList[ix+nrBeamlets.x*iy].baseRay.opl=0;
		}
	}
	return 1; // signal success
}

void testInterpolator()
{
	unsigned long N=51;

	double*x=(double*)calloc(N,sizeof(double));
	double*y=(double*)calloc(N,sizeof(double));

	double xmax=10;
	double ymax=10;
	double dx=2*xmax/N;
	double dy=2*ymax/N;

	for (unsigned int j=0;j<N;j++)
	{
		x[j]=-1.0*xmax+j*dx;
		y[j]=-1.0*ymax+j*dy;
	}

	complex<double> *Uin=(complex<double>*)calloc(N*N,sizeof(complex<double>));
//	double *UinAbs=(double*)calloc(N*N,sizeof(double));
	double periodX=5;
	double periodY=10;

	 for (unsigned int jx=0;jx<N;jx++)
	 {
		 for (unsigned int jy=0;jy<N;jy++)
		 {
			Uin[jx+jy*N]=complex<double>(cos(2*M_PI/periodX*x[jx])+1,M_PI*cos(2*M_PI/periodY*y[jy]));
//			UinAbs[jx+jy*N]=cos(2*M_PI/periodX*x[jx])+1;
		 }
	 }

	 char t_filename[512] = "Real.txt";

	 FILE* hFileReal;
	 hFileReal = fopen( t_filename, "w" ) ;

	 sprintf(t_filename, "Imag.txt");
	 FILE* hFileImag;
	 hFileImag = fopen( t_filename, "w" ) ;


//	 if ( (hFileReal == NULL) || (hFileImag == NULL) )
//		 return 1;

	 for (unsigned int jy=0;jy<N;jy++)
	 {
		 for (unsigned int jx=0;jx<N;jx++)
		 {
			 if (jx+1 == N)
			 {
//				fprintf(hFileReal, " %.16e;\n", UinAbs[jx+jy*N]);
				fprintf(hFileReal, " %.16e;\n", Uin[jx+jy*N].real());
				fprintf(hFileImag, " %.16e;\n", Uin[jx+jy*N].imag());
			 }
			 else
			 {
//				fprintf(hFileReal, " %.16e;", UinAbs[jx+jy*N]);
				fprintf(hFileReal, " %.16e;", Uin[jx+jy*N].real());
				fprintf(hFileImag, " %.16e;", Uin[jx+jy*N].imag());
			 }

		 }
	}

	fclose(hFileReal);
	fclose(hFileImag);

	Interpolator oInterp;

	oInterp.initInterpolation(x, y, Uin, N, N);
	//oInterp.initInterpolation(x, y, UinAbs, N, N);

     sprintf(t_filename, "Real.txt");
	 hFileReal = fopen( t_filename, "w" ) ;

	 sprintf(t_filename, "Imag.txt");
	 hFileImag = fopen( t_filename, "w" ) ;


//	 if ( (hFileReal == NULL) || (hFileImag == NULL) )
//		 return 1;

	 for (unsigned int jy=0;jy<N;jy++)
	 {
		 for (unsigned int jx=0;jx<N;jx++)
		 {
			 if (jx+1 == N)
			 {
//				fprintf(hFileReal, " %.16e;\n", UinAbs[jx+jy*N]);
				fprintf(hFileReal, " %.16e;\n", oInterp.y2ca_ptr[jx+jy*N].real());
				fprintf(hFileImag, " %.16e;\n", oInterp.y2ca_ptr[jx+jy*N].imag());
			 }
			 else
			 {
//				fprintf(hFileReal, " %.16e;", UinAbs[jx+jy*N]);
				fprintf(hFileReal, " %.16e;", oInterp.y2ca_ptr[jx+jy*N].real());
				fprintf(hFileImag, " %.16e;", oInterp.y2ca_ptr[jx+jy*N].imag());
			 }

		 }
	}
	fclose(hFileReal);
	fclose(hFileImag);

	unsigned long NInterpX=256;
	unsigned long NInterpY=256;

	double*xInterp=(double*)calloc(NInterpX,sizeof(double));
	double*yInterp=(double*)calloc(NInterpY,sizeof(double));
	complex<double> *UinInterp=(complex<double>*)calloc(NInterpX*NInterpY,sizeof(complex<double>));
	double *UinAbsInterp=(double*)calloc(NInterpX*NInterpY,sizeof(double));

	double dxInterp=2*xmax/NInterpX;
	double dyInterp=2*ymax/NInterpY;
	for (unsigned int j=0;j<NInterpX;j++)
	{
		xInterp[j]=-1.0*xmax+j*dxInterp;
	}
	for (unsigned int j=0;j<NInterpY;j++)
	{
		yInterp[j]=-1.0*ymax+j*dyInterp;
	}


	 for (unsigned int jy=0;jy<NInterpY;jy++)
	 {
		 for (unsigned int jx=0;jx<NInterpX;jx++)
		 {
			 oInterp.doInterpolation(x, y, N, N, Uin, xInterp[jx], yInterp[jy], &UinInterp[jx+jy*NInterpX]);
//			 oInterp.doInterpolation(x, y, N, N, UinAbs, xInterp[jx], yInterp[jy], &UinAbsInterp[jx+jy*NInterpX]);
		 }
	 }

	 sprintf(t_filename, "RealInterp.txt");

	 hFileReal = fopen( t_filename, "w" ) ;

	 sprintf(t_filename, "ImagInterp.txt");
	 hFileImag = fopen( t_filename, "w" ) ;


//	 if ( (hFileReal == NULL) || (hFileImag == NULL) )
//		 return 1;

	 for (unsigned int jy=0;jy<NInterpY;jy++)
	 {
		 for (unsigned int jx=0;jx<NInterpX;jx++)
		 {
			 if (jx+1 == NInterpY)
			 {
//				 fprintf(hFileReal, " %.16e;\n", UinAbsInterp[jx+jy*NInterp]);
				fprintf(hFileReal, " %.16e;\n", UinInterp[jx+jy*NInterpX].real());
				fprintf(hFileImag, " %.16e;\n", UinInterp[jx+jy*NInterpX].imag());
			 }
			 else
			 {
//				 fprintf(hFileReal, " %.16e;", UinAbsInterp[jx+jy*NInterp]);
				fprintf(hFileReal, " %.16e;", UinInterp[jx+jy*NInterpX].real());
				fprintf(hFileImag, " %.16e;", UinInterp[jx+jy*NInterpX].imag());
			 }

		 }
	}

	fclose(hFileReal);
	fclose(hFileImag);

	// cleanup
	delete xInterp;
	delete yInterp;
	delete x;
	delete y;
	delete UinInterp;
	delete Uin;
};

/**
 * \detail createSceneFromXML 
 *
 * parses the XML and creates an OptiX scene 
 *
 * \param[in] Group *oGroupPtr, FILE *hfile, RayField ***sourceListPtr, long long *sourceNumberPtr, Detector ***detListPtr, long long *detNumberPtr, simMode mode
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool createSceneFromXML(Group *oGroupPtr, FILE *hfile, RayField ***sourceListPtr, Detector ***detListPtr)
{

	cout <<"********************************************" << endl;
	cout <<"starting to parse prescritpion files..." << endl;

	xml_document doc;
	xml_parse_result result=doc.load_file("myXml.xml");

	// get the root element
	xml_node scene =doc.first_child();
	if (strcmp(scene.name(), "scene") != 0)
	{
		cout << "error in createSceneFromXML: Root element of file is not scene. File is not a valid scene description" << endl;
		return false;
	}
	// create instance of xml parser
	Parser_XML* l_pParser=new Parser_XML();
	
	// read simulation mode from xml file
	simMode l_mode;
	const char* l_pModeString=l_pParser->attrValByName(scene, "mode");

	if (strcmp(l_pModeString,"GEOMRAYS_SEQ") == 0)
		l_mode=SIM_GEOMRAYS_SEQ;
	if (strcmp(l_pModeString,"GEOMRAYS_NONSEQ") == 0)
		l_mode=SIM_GEOMRAYS_NONSEQ;	

	// get number all geometryGroups
	vector<xml_node>* l_pGeometryGroups;
	l_pGeometryGroups=l_pParser->childsByTagName(scene, "geometryGroup");

	/* set number of geometry groups */
	if (GROUP_NO_ERR != oGroupPtr->setGeometryGroupListLength(l_pGeometryGroups->size()) )
	{
		std::cout <<"error in Parser.createSceneFromZemax(): group.setGeometryGroupListLength(1) returned an error" << std::endl;
		return false;
	}

	/* create geometryGroups inside the group */
	for (int i=0; i<l_pGeometryGroups->size(); i++)
	{
		if (GROUP_NO_ERR != oGroupPtr->createGeometryGroup(i) )
		{
			std::cout <<"error in Parser_XML.createSceneFromXML(): group.createGeometryGroup(" << i << ") returned an error" << std::endl;
			return false;
		}

		// determine number of surfaces in the current geometryGroup
		int nrSurfaces=0;
		for (xml_node child = l_pGeometryGroups->at(i).first_child(); child; child=child.next_sibling())
		{
			// check wether child is geometry
			if ( strcmp(child.name(), "geometry") == 0)
			{
				const char* t_str;
				switch (l_mode)
				{
				case SIM_GEOMRAYS_SEQ:
					t_str = l_pParser->attrValByName(child, "nrSurfacesSeq");
					if (t_str==NULL)
					{
						cout << "error in createSceneFromXML: nrSurfacesSeq is not defined for current node." << endl;
						return false;
					}
					nrSurfaces=nrSurfaces+atoi(t_str);
					break;
				case SIM_GEOMRAYS_NONSEQ:
					t_str = l_pParser->attrValByName(child, "nrSurfacesNonSeq");
					if (t_str==NULL)
					{
						cout << "error in createSceneFromXML: nrSurfacesNonSeq is not defined for current node." << endl;
						return false;
					}
					nrSurfaces=nrSurfaces+atoi(t_str);
					break;
				default:
					cout << "error in createSceneFromXML: unknown simulation mode." << endl;
				}
			}
		}	
		// collect all geometry nodes from the current geometryGroup
		vector<xml_node>* l_pGeometries = l_pParser->childsByTagName(l_pGeometryGroups->at(i),"geometry");

		/* set number of geometries in current geometryGroup*/
		if (GEOMGROUP_NO_ERR != oGroupPtr->getGeometryGroup(i)->setGeometryListLength(nrSurfaces) )
		{
			std::cout <<"error in Parser_XML.createSceneFromXML(): group.getGeometryGroup(" << i << ")->setGeometryListLength(" << nrSurfaces << ") returned an error"  << std::endl;
			return false;
		}
		
		GeometryFab l_GeomFab;
		int globalSurfaceCount=0; // as some of the geometries consist of different number of surfaces in different simulation modes, we need to keep track of the number of surfaces in each geometryGroup here...
		// now, create the objects and add them to current geometryGroup
		for (int j=0; j<l_pGeometries->size(); j++)
		{			
			vector<Geometry*> l_geomVec;
			l_geomVec=l_GeomFab.createGeomInstFromXML(l_pGeometries->at(j), l_mode);
			if (l_geomVec.size() == 0)
			{
				cout << "error in createSceneFromXML: l_GeomFab.createGeomInstFromXML() returned nothing for given XML node " << j << "in geometryGroup " << i << endl;
				return false;
			}
			for (int k=0; k<l_geomVec.size(); k++)
			{
				if (GEOMGROUP_NO_ERR != oGroupPtr->getGeometryGroup(i)->setGeometry(l_geomVec.at(k),globalSurfaceCount))
				{
					cout << "error in createSceneFromXML: getGeometryGroup(i)->setGeometry() returned an error at index" << k << endl;
					return false;
				}
				globalSurfaceCount++;
			}
		}
	}

	
	delete l_pParser;

	return true;
}