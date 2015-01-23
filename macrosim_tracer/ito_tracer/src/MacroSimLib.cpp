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

#define _CRTDBG_MAP_ALLOC
/**\file sample2.cpp
* \brief entry point to the application containing the main routine
* 
*           
* \author Mauch
*/

#include "MacroSimLib.h"
//#include <omp.h>
#include "TopObject.h"
#include "cuda_runtime.h"
#include <optix.h>
#include "macrosim_functions.h"
#include "myUtil.h"
#include <sampleConfig.h>
#include "Group.h"
#include "Geometry.h"
#include "GeometryLib.h"
//#include "MaterialLib.h"
//#include "ScatterLib.h"
//#include "CoatingLib.h"
#include "differentialRayTracing\GeometryLib_DiffRays.h"
//#include "differentialRayTracing\MaterialLib_DiffRays.h"
//#include "differentialRayTracing\CoatingLib_DiffRays.h"
//#include "differentialRayTracing\ScatterLib_DiffRays.h"
#include "geometricRender\GeometryLib_GeomRender.h"
#include "SimAssistantLib.h"
#include "DetectorLib.h"
#include "Scatter.h"
#include "Coating.h"
#include "RayField.h"
#include "GeometricRayField.h"
#include <math.h>
#include "rayTracingMath.h"
#include <ctime>
#include "FlexZemax.h"
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

#include <iostream>
#include <fstream>

#include "Interpolator.h"

#include <nvModel.h>
#include <optixu/optixpp_namespace.h>

#define MACROSIMSTART
#include "GlobalConstants.h"

//#include <fstream>
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
const float SCENE_EPSILON=1.0e-4f;

typedef struct struct_BoxExtent{
    float min[3];
    float max[3];
} BoxExtent;

// declare scene group
Group *oGroup;

void testInterpolator();
void createContext( RTcontext* context, RTbuffer* buffer );
void printUsageAndExit( const char* argv0 );
//void doCPUTracing(Group &oGroup, GeometricRayField &oGeomRayField);
//void doCPUTracing(Group &oGroup, DiffRayField &oDiffRayField);
//void doCPUTracing(Group &oGroup, GaussBeamRayField &oGaussBeamRayField);
unsigned int createGaussianBeamRaysTest(GaussBeamRayField **ptrptrRayField, double2 width, double2 offset, long2 nrBeamlets, double lambda);
void initRayField_AsphereTestCPU(FILE *hfileQxy, FILE *hfilePxy, GeometricRayField* oGemRayFieldPtr, double RadiusSourceReference, double zSourceReference, double *MNmn, int width, int height, double lambda);
void initRayField_AsphereTestGPU(RTcontext &context, FILE *hfileQxy, FILE *hfilePxy, GeometricRayField* oGeomRayFieldPtr, double RadiusSourceReference, double zSourceReference, double *MNmn, int width, int height, double lambda);
bool doTheSimulation(Group *oGrouPtr, RayField *SourceListPtrPtr, bool RunOnCPU);
//bool createSceneFromXML(Group *oGroupPtr, FILE *hfile, RayField ***sourceListPtr, Detector ***detListPtr);
bool createSceneFromXML(Group **oGroupPtrPtr, char *sceneChar, Field ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, MacroSimTracerParams &tracerParams);


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

//#ifdef _DEBUG
//#else
//	if (argc <4)
//	{
//		cout << "error in main: too few arguments." << endl;
//		return (1);
//	}
//#endif

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
//	char* filepath=(char*)malloc(512*sizeof(char));


//#ifdef _DEBUG
//	sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, "BeamHomo_PhaseSpace.TXT");//prescription_Avinash_PathTrace.TXT");//PhaseSpaceTest_Stabmischer1D_CosineProfile.TXT");//DiffTest_f9mm_idealLense.TXT");//fiber_interferometer_opt50x05_in.TXT");//prescription_Geometry_test.TXT");//prescription_nonsequential_OptiX_FromSlit_CompleteHousing_Filter.TXT");//prescription_nonsequential_test.TXT");//prescription_nonsequential_OptiX_FromSlit_Housing.TXT");//OptiX_FromSlit_CompleteHousing.TXT");//scatMat.TXT");//test.TXT");//Sidelobe380.TXT");//CMIdeal1.TXT");//
//	MAX_DEPTH_CPU=3;
//#else
//	sprintf(filepath, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH, argv[1]);
//	MAX_DEPTH_CPU=atoi(argv[3]);
//#endif
	
//	FILE *hfile = fopen( filepath, "r" ) ;
//	if (!hfile)
//	{
//		cout << "error in main: cannot open geometry prescription file:" << filepath << endl;
//		return (1);
//	}
	
	// decide wether we want to do sequential or nonsequential simulations
//	TraceMode mode=TRACE_NONSEQ;
	char inFile[512];
	sprintf(inFile, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH,"MacroSimDoubleGaussSeq_Timing.xml");

	std::ifstream is;
	is.open( inFile, ifstream::in );
	if (is.fail())
	{
		std::cout << "error in main: can not optn scene description file" << "...\n";
		return 1;
	}

	// get length of file
	is.seekg(0, ios::end);
	int length = is.tellg();
	is.seekg(0, ios::beg);

	// allocate memory:
	char *buffer = new char [length];

	// read data as a block
	is.read(buffer, length);
	is.close();

//	if (!hfile)
//	{
//		cout << "error in main: cannot open xml file:" << filepath << endl;
//		return (1);
//	}

	sprintf(FILE_GLASSCATALOG, "%s" PATH_SEPARATOR "%s", INPUT_FILEPATH,"glass.AGF");
	sprintf(OUTPUT_FILEPATH, "E:\\mauch\MacroSim_out");
	sprintf(INPUT_FILEPATH, "E:\\mauch\MacroSim_in");

	Group* l_pGroup=NULL;
	Field** SourceList;
	SourceList=NULL;
	Detector** DetectorList;
	DetectorList=NULL;
	long sourceNumber, detNumber;

	MacroSimTracerParams l_simParams;

	createSceneFromXML(&l_pGroup, buffer, &SourceList, &sourceNumber, &DetectorList, &detNumber, l_simParams);

//	if (!createSceneFromZemax(&oGroup, hfile, &SourceList, &sourceNumber, &DetectorList, &detNumber, mode))
//	{
//		cout << "error in main: createSceneFromZemax returned an error" << endl;
//		fclose(hfile); // close file anyway
//		return (1);
//	}
//	fclose(hfile);


	/****************************************************************/
	/* trace rays                                                   */
	/****************************************************************/

	SourceList[0]->setSubsetHeightMax(2000);
	SourceList[0]->setSubsetWidthMax(2000);

	// load xml document
	xml_document doc;
	xml_parse_result result=doc.load(buffer);

	// get the root element
	xml_node scene =doc.first_child();
	if (strcmp(scene.name(), "scene") != 0)
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): Root element of file is not scene. File is not a valid scene description" << endl;
		return false;
	}

	simAssParams *oSimAssParamsPtr=NULL;
	SimAssistant *oSimAssPtr=NULL;

	Parser_XML l_parser;

	oSimAssPtr = new SimAssistantSingleSim;
	//oSimAssParamsPtr = new simAssSingleSimParams();
	oSimAssParamsPtr = new simAssParams();


	// read simulation mode from xml file
	const char* l_pString=l_parser.attrValByName(scene, "mode");

	if (strcmp(l_pString,"SEQUENTIAL") == 0)
		oSimAssParamsPtr->simParams.traceMode=TRACE_SEQ;
	if (strcmp(l_pString,"NONSEQUENTIAL") == 0)
		oSimAssParamsPtr->simParams.traceMode=TRACE_NONSEQ;	

	l_pString=l_parser.attrValByName(scene, "GPUacceleration");

	if ((strcmp(l_pString,"TRUE") == 0))
		oSimAssParamsPtr->RunOnCPU=false;
	else
		oSimAssParamsPtr->RunOnCPU=true;

	oSimAssPtr->setParamsPtr(oSimAssParamsPtr);

	RayField** RaySourceList=reinterpret_cast<RayField**>(SourceList);
	if (SIMASS_NO_ERROR != oSimAssPtr->initSimulation(l_pGroup, RaySourceList[0]))
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): SimAss.initSimulation() returned an error" << endl;
		return( false );
	}
	if (SIMASS_NO_ERROR != oSimAssPtr->run(l_pGroup, RaySourceList[0], DetectorList))
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): SimAss.run() returned an error" << endl;
		return( false );
	}

	end=clock();
	double msecs;
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	cout << endl;
	cout << msecs <<"ms to run the whole simulation." << endl;

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

/**
 * \detail createSceneFromXML 
 *
 * parses the XML and creates an OptiX scene 
 *
 * \param[in] Group *oGroupPtr, FILE *hfile, RayField ***sourceListPtr, long long *sourceNumberPtr, Detector ***detListPtr, long long *detNumberPtr, TraceMode mode, MacroSimTracerParams &tracerParams
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool createSceneFromXML(Group **oGroupPtrPtr, char *sceneChar, Field ***sourceListPtr, long *sourceNumberPtr, Detector ***detListPtr, long *detNumberPtr, MacroSimTracerParams &tracerParams)
{

	cout <<"********************************************" << endl;
	cout <<"starting to parse prescription files..." << endl;

	xml_document doc;
	xml_parse_result result=doc.load(sceneChar);
	//xml_parse_result result=doc.load_file("myXml.xml");

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
	const char* l_pModeString=l_pParser->attrValByName(scene, "mode");

	if (strcmp(l_pModeString,"SEQUENTIAL") == 0)
		tracerParams.simParams.traceMode=TRACE_SEQ;
	if (strcmp(l_pModeString,"NONSEQUENTIAL") == 0)
		tracerParams.simParams.traceMode=TRACE_NONSEQ;	

	// read file paths from xml
	const char* l_pGlassFilePath=l_pParser->attrValByName(scene, "glassCatalog");
	if (!l_pGlassFilePath)
	{
		cout << "error in createSceneFromXML: glassCatalog is not defined" << endl;
		return false;
	}
	memcpy(tracerParams.glassFilePath, l_pGlassFilePath, sizeof(char)*512);
	const char* l_pOutPath=l_pParser->attrValByName(scene, "outputFilePath");
	if (!l_pOutPath)
	{
		cout << "error in createSceneFromXML: outputFilePath is not defined" << endl;
		return false;
	}
	memcpy(tracerParams.outputFilesPath, l_pOutPath, sizeof(char)*512);
	const char* l_pInPath=l_pParser->attrValByName(scene, "inputFilePath");
	if (!l_pInPath)
	{
		cout << "error in createSceneFromXML: inputFilePath is not defined" << endl;
		return false;
	}
	memcpy(tracerParams.inputFilesPath, l_pInPath, sizeof(char)*512);
	const char* l_pPtxPath=l_pParser->attrValByName(scene, "ptxPath");
	if (!l_pPtxPath)
	{
		cout << "error in createSceneFromXML: ptxPath is not defined" << endl;
		return false;
	}
	memcpy(tracerParams.path_to_ptx, l_pPtxPath, sizeof(char)*512);

	// init global variables
	memcpy(FILE_GLASSCATALOG, tracerParams.glassFilePath, sizeof(char)*512);
	memcpy(OUTPUT_FILEPATH, tracerParams.outputFilesPath, sizeof(char)*512);
	memcpy(INPUT_FILEPATH, tracerParams.inputFilesPath, sizeof(char)*512);
	memcpy(PATH_TO_PTX, tracerParams.path_to_ptx, sizeof(char)*512);

	if (NULL!=l_pParser->attrByNameToInt(scene, "numCPU", tracerParams.numCPU))
	{
		cout << "error in createSceneFromXML: numCPU is not defined" << endl;
		return false;
	}
	unsigned long l_long;
	if (NULL!=l_pParser->attrByNameToLong(scene, "rayTilingHeight", l_long))
	{
		cout << "error in createSceneFromXML: subsetHeight is not defined" << endl;
		return false;
	}
	tracerParams.subsetHeight=l_long;
	if (NULL!=l_pParser->attrByNameToLong(scene, "rayTilingWidth", l_long))
	{
		cout << "error in createSceneFromXML: subsetWidth is not defined" << endl;
		return false;
	}
	tracerParams.subsetWidth=l_long;

	// get all sources
	vector<xml_node>* l_pSources;
	l_pSources=l_pParser->childsByTagName(scene, "field");

	// so far we only allow for exactly on source
	if (l_pSources->size() != 1)
	{
		cout << "error in createSceneFromXML: the scene has to have exactly one source to be vaild!" << endl;
		return false;
	}

	FieldFab l_fieldFab;
	/* create array for the sources of our simulation */
	*sourceListPtr=new Field*[l_pSources->size()];
	*sourceNumberPtr=l_pSources->size();

	/* create sources */
	for (int j=0; j<l_pSources->size(); j++)
	{
		vector<Field*> l_sourceVec;
        if (!l_fieldFab.createFieldInstFromXML(l_pSources->at(j), l_sourceVec, tracerParams.simParams))
		{
			cout << "error in createSceneFromXML: l_FieldFab.createFieldInstFromXML() returned an error for given XML node " << j << endl;
			return false;
		}
		// so far we don't allow for compound sources, therefore we know the size here will be equal to 1
		for (int k=0; k<1;k++)//l_sourceVec.size(); k++)
		{
			*sourceListPtr[k]=l_sourceVec.at(k);
		}
	}

    // determine simulation mode from source
    (*sourceListPtr[0])->setSimMode(tracerParams.simParams.simMode);

	// get all geometryGroups
	vector<xml_node>* l_pGeometryGroups;
	l_pGeometryGroups=l_pParser->childsByTagName(scene, "geometryGroup");

	*oGroupPtrPtr=new Group();
	/* set number of geometry groups */
	if (GROUP_NO_ERR != (*oGroupPtrPtr)->setGeometryGroupListLength(l_pGeometryGroups->size()) )
	{
		std::cout <<"error in Parser.createSceneFromZemax(): group.setGeometryGroupListLength(1) returned an error" << "...\n";
		return false;
	}

	/* create geometryGroups inside the group */
	for (int i=0; i<l_pGeometryGroups->size(); i++)
	{
		if (GROUP_NO_ERR != (*oGroupPtrPtr)->createGeometryGroup(i) )
		{
			std::cout <<"error in Parser_XML.createSceneFromXML(): group.createGeometryGroup(" << i << ") returned an error" << "...\n";
			return false;
		}
		// parse params of geometryGroup
		(*oGroupPtrPtr)->getGeometryGroup(i)->parseXml(l_pGeometryGroups->at(i));

		// determine number of surfaces in the current geometryGroup
		int nrSurfaces=0;
		for (xml_node child = l_pGeometryGroups->at(i).first_child(); child; child=child.next_sibling())
		{
			// check wether child is geometry
			if ( strcmp(child.name(), "geometry") == 0)
			{
				const char* t_str;
				switch (tracerParams.simParams.traceMode)
				{
				case TRACE_SEQ:
					t_str = l_pParser->attrValByName(child, "nrSurfacesSeq");
					if (t_str==NULL)
					{
						cout << "error in createSceneFromXML: nrSurfacesSeq is not defined for current node." << endl;
						return false;
					}
					nrSurfaces=nrSurfaces+atoi(t_str);
					break;
				case TRACE_NONSEQ:
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
		if (GEOMGROUP_NO_ERR != (*oGroupPtrPtr)->getGeometryGroup(i)->setGeometryListLength(nrSurfaces) )
		{
			std::cout <<"error in Parser_XML.createSceneFromXML(): group.getGeometryGroup(" << i << ")->setGeometryListLength(" << nrSurfaces << ") returned an error"  << "...\n";
			return false;
		}
		
        GeometryFab* l_pGeomFab;
        switch (tracerParams.simParams.simMode)
        {
        case SIM_GEOM_RT:
            l_pGeomFab=new GeometryFab();
            break;
        case SIM_DIFF_RT:
            l_pGeomFab=new GeometryFab_DiffRays();
            break;
        case SIM_GEOM_RENDER:
            l_pGeomFab=new GeometryFab_GeomRender();
            break;
        default:
			std::cout <<"error in Parser_XML.createSceneFromXML(): unknown trace mode"  << "...\n";
			return false;
            break;
        }

		int globalSurfaceCount=0; // as some of the geometries consist of different number of surfaces in different simulation modes, we need to keep track of the number of surfaces in each geometryGroup here...
		// now, create the objects and add them to current geometryGroup
		for (int j=0; j<l_pGeometries->size(); j++)
		{			
			vector<Geometry*> l_geomVec;
			if (!l_pGeomFab->createGeomInstFromXML(l_pGeometries->at(j), tracerParams.simParams, l_geomVec))
			{
				cout << "error in createSceneFromXML: l_GeomFab.createGeomInstFromXML() returned an error for given XML node " << j << "in geometryGroup " << i << endl;
				return false;
			}
			for (int k=0; k<l_geomVec.size(); k++)
			{
				l_geomVec.at(k)->getParamsPtr()->geometryID=globalSurfaceCount+1; // geometryID=0 is for the source, so we start counting our geometries at 1
				if (GEOMGROUP_NO_ERR != (*oGroupPtrPtr)->getGeometryGroup(i)->setGeometry(l_geomVec.at(k),globalSurfaceCount))
				{
					cout << "error in createSceneFromXML: getGeometryGroup(i)->setGeometry() returned an error at index" << k << endl;
					return false;
				}
				globalSurfaceCount++;
			}
		}
		delete l_pGeometries;
        delete l_pGeomFab;
	} // end lopp geometryGroups

	// get all detectors
	vector<xml_node>* l_pDetectors;
	l_pDetectors=l_pParser->childsByTagName(scene, "detector");

	// so far we only allow for exactly one detector
	if ( (l_pDetectors->size() != 1) )
	{
		cout << "error in createSceneFromXML: the scene has to have exactly one detector to be vaild!" << endl;
		return false;
	}

	DetectorFab l_detFab;
	*detListPtr=new Detector*[l_pDetectors->size()];
	*detNumberPtr=l_pDetectors->size();

	/* create detectors */
	for (int j=0; j<l_pDetectors->size(); j++)
	{
		vector<Detector*> l_detVec;
		if (!l_detFab.createDetInstFromXML(l_pDetectors->at(j), l_detVec))
		{
			cout << "error in createSceneFromXML: l_detFab.createDetInstFromXML() returned an error for given XML node " << j << endl;
			return false;
		}
		// so far we don't allow for compound detectors, therefore we know the size here will be equal to 1
		for (int k=0; k<1;k++)//l_sourceVec.size(); k++)
		{
			*detListPtr[k]=l_detVec.at(k);
		}
	}

	delete l_pParser;
	delete l_pDetectors;
	delete l_pSources;

	return true;
}

bool MacroSimTracer::runMacroSimRayTrace(char *xmlInput, void** fieldOut_ptrptr, ItomFieldParams* fieldOutParams, void* p2ProgCallbackObject, void (*callbackProgress)(void* p2Object, int progressValue))
{
	clock_t start, end;
	start=clock();

	Field** SourceList;
	SourceList=NULL;
	Detector** DetectorList;
	DetectorList=NULL;
	long sourceNumber, detNumber;
    oGroup=NULL;
//	Group* l_pGroup=oGroup;

	MacroSimTracerParams l_simParams;

	if (!createSceneFromXML(&oGroup, xmlInput, &SourceList, &sourceNumber, &DetectorList, &detNumber, l_simParams))
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): createSceneFromXML() returned an error" << endl;
		return false;
	}


//	streambuf* old = cout.rdbuf(pOutBuffer->rdbuf());
//	cout << "bla" << endl;

	SourceList[0]->setSubsetHeightMax(l_simParams.subsetHeight);
	SourceList[0]->setSubsetWidthMax(l_simParams.subsetWidth);
	if ((l_simParams.subsetHeight==0) || (l_simParams.subsetWidth==0))
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): a subset size of zero is not allowed." << endl;
		return false;
	}
	SourceList[0]->setNumCPU(l_simParams.numCPU);
	if (l_simParams.numCPU==0)
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): at least on CPU core needs to be assigned to tracing." << endl;
		return false;
	}

	// load xml document
	xml_document doc;
	xml_parse_result result=doc.load(xmlInput);

	// get the root element
	xml_node scene =doc.first_child();
	if (strcmp(scene.name(), "scene") != 0)
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): Root element of file is not scene. File is not a valid scene description" << endl;
		return false;
	}

	simAssParams *oSimAssParamsPtr=NULL;
	SimAssistant *oSimAssPtr=NULL;

	Parser_XML l_parser;

	//// read layout mode from xml file
	//const char* l_pString=l_parser.attrValByName(scene, "layoutMode");
	//bool l_layoutMode;

	//if ((strcmp(l_pString,"TRUE") == 0))
	//{
	//	oSimAssPtr = new SimAssistantLayout();
	//	oSimAssParamsPtr = new simAssLayoutParams();
	//	l_layoutMode=true;
	//}
	//else
	//{
	//	oSimAssPtr = new SimAssistantSingleSim;
	//	oSimAssParamsPtr = new simAssSingleSimParams();
	//	l_layoutMode=false;
	//}

	oSimAssPtr = new SimAssistantSingleSim;
	//oSimAssParamsPtr = new simAssSingleSimParams();
	oSimAssParamsPtr = new simAssParams();


	// read simulation mode from xml file
	const char* l_pString=l_parser.attrValByName(scene, "mode");

	if (strcmp(l_pString,"SEQUENTIAL") == 0)
		oSimAssParamsPtr->simParams.traceMode=TRACE_SEQ;
	if (strcmp(l_pString,"NONSEQUENTIAL") == 0)
		oSimAssParamsPtr->simParams.traceMode=TRACE_NONSEQ;	

	l_pString=l_parser.attrValByName(scene, "GPUacceleration");

	if ((strcmp(l_pString,"TRUE") == 0))
		oSimAssParamsPtr->RunOnCPU=false;
	else
		oSimAssParamsPtr->RunOnCPU=true;

	oSimAssPtr->setParamsPtr(oSimAssParamsPtr);

	//oSimAssPtr->setCallback(p2ProgCallbackObject,callbackProgress);
	SourceList[0]->setProgressCallback(p2ProgCallbackObject, callbackProgress);

	RayField** RaySourceList=reinterpret_cast<RayField**>(SourceList);
	if (SIMASS_NO_ERROR != oSimAssPtr->initSimulation(oGroup, RaySourceList[0]))
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): SimAss.initSimulation() returned an error" << endl;
		return( false );
	}
	if (SIMASS_NO_ERROR != oSimAssPtr->run(oGroup, RaySourceList[0], DetectorList))
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): SimAss.run() returned an error" << endl;
		return( false );
	}

	// the result of the simulation will be stored in oSimassPtr->fieldPtr. We transfer this to the Itom-gui here...
	if (FIELD_NO_ERR != oSimAssPtr->getResultFieldPtr()->convert2ItomObject(fieldOut_ptrptr, fieldOutParams))
	{
		cout << "error in MacroSimTracer.runMacroSimRayTrace(): field.convert2ItomObject() returned an error" << endl;
		return( false );
	}


	end=clock();
	double msecs;
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	cout << endl;
	cout << msecs <<"ms to run the whole simulation." << endl;

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

	if (oGroup != NULL)
	{
		delete oGroup;
		oGroup=NULL;
	}

	return true;
}

bool MacroSimTracer::runMacroSimLayoutTrace(char *xmlInput, void* p2CallbackObject, void (*callbackRayPlotData)(void* p2Object, double* rayPlotData, RayPlotDataParams *params))
{
	Field** SourceList;
	SourceList=NULL;
	Detector** DetectorList;
	DetectorList=NULL;
	long sourceNumber, detNumber;

	MacroSimTracerParams l_simParams;

    oGroup=NULL;
	//Group* l_pGroup=NULL;

	if (!createSceneFromXML(&oGroup, xmlInput, &SourceList, &sourceNumber, &DetectorList, &detNumber, l_simParams))
	{
		cout << "error in MacroSimTracer.runMacroSimLayoutTrace(): createSceneFromXML() returned an error" << endl;
		return false;
	}

	// init global variables
	memcpy(FILE_GLASSCATALOG, l_simParams.glassFilePath, sizeof(char)*512);
	memcpy(INPUT_FILEPATH, l_simParams.inputFilesPath, sizeof(char)*512);
	memcpy(OUTPUT_FILEPATH, l_simParams.outputFilesPath, sizeof(char)*512);
	memcpy(PATH_TO_PTX, l_simParams.path_to_ptx, sizeof(char)*512);



	SourceList[0]->setSubsetHeightMax(l_simParams.subsetHeight);
	SourceList[0]->setSubsetWidthMax(l_simParams.subsetWidth);

	// load xml document
	xml_document doc;
	xml_parse_result result=doc.load(xmlInput);

	// get the root element
	xml_node scene =doc.first_child();
	if (strcmp(scene.name(), "scene") != 0)
	{
		cout << "error in MacroSimTracer.runMacroSimLayoutTrace(): Root element of file is not scene. File is not a valid scene description" << endl;
		return false;
	}

	simAssParams *oSimAssParamsPtr=new simAssParams();
	//simAssLayoutParams *oSimAssParamsPtr=new simAssLayoutParams();
	SimAssistantLayout *oSimAssPtr=new SimAssistantLayout();

	Parser_XML l_parser;

	// read simulation mode from xml file
	const char* l_pString=l_parser.attrValByName(scene, "mode");

	if (strcmp(l_pString,"SEQUENTIAL") == 0)
		oSimAssParamsPtr->simParams.traceMode=TRACE_SEQ;
	if (strcmp(l_pString,"NONSEQUENTIAL") == 0)
		oSimAssParamsPtr->simParams.traceMode=TRACE_NONSEQ;	

	l_pString=l_parser.attrValByName(scene, "GPUacceleration");

	if ((strcmp(l_pString,"TRUE") == 0))
	{
		cout << "warning in MacroSimTracer.runMacroSimLayoutTrace(): GPU acceleration is not implemented for layout mode. continuing on CPU anyways..." << endl;
		oSimAssParamsPtr->RunOnCPU=true;
	}
	else
		oSimAssParamsPtr->RunOnCPU=true;

	oSimAssPtr->setParamsPtr(oSimAssParamsPtr);

	oSimAssPtr->setCallbackRayPlotData(p2CallbackObject,callbackRayPlotData);

	RayField** RaySourceList=reinterpret_cast<RayField**>(SourceList);
	if (SIMASS_NO_ERROR != oSimAssPtr->initSimulation(oGroup, RaySourceList[0]))
	{
		cout << "error in MacroSimTracer.runMacroSimLayoutTrace(): SimAss.initSimulation() returned an error" << endl;
		return( 1 );
	}
	if (SIMASS_NO_ERROR != oSimAssPtr->run(oGroup, RaySourceList[0], DetectorList))
	{
		cout << "error in MacroSimTracer.runMacroSimLayoutTrace(): SimAss.run() returned an error" << endl;
		return( 1 );
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

	if (oGroup != NULL)
	{
		delete oGroup;
		oGroup=NULL;
	}

	return true;
}

bool MacroSimTracer::calcCuFFT(cuDoubleComplex *pUin, int dimX, int dimY)
{
	if (PROP_NO_ERR!=cu_ft2(pUin, dimX, dimY))
	{
		cout << "error in MacroSimTracer.calcCuFFT(): cu_ft2() returned an error." << endl;
		return false;
	}
	return true;
}

bool MacroSimTracer::runConfRawSigSim(ConfPoint_Params &params, double** res_ptrptr)
{
	bool runOnCPU=false;
	if (PROP_NO_ERR!=simConfRawSig(res_ptrptr, params, runOnCPU))
	{
		cout << "error in MacroSimTracer.runConfRawSigSim(): simConfRawSig() returned an error." << endl;
		return false;
	}
	return true;
};

bool MacroSimTracer::runConfSensorSigSim(ConfPoint_Params &params, ConfPointObject_Params &paramsObject, double** res_ptrptr)
{
	bool runOnCPU=false;
	if (PROP_NO_ERR!=simConfSensorSig(res_ptrptr, params, paramsObject, runOnCPU))
	{
		cout << "error in MacroSimTracer.runConfSensorigSim(): simConfRawSig() returned an error." << endl;
		return false;
	}
	return true;
};

bool MacroSimTracer::checkVisibility(char *objectInput_filename)
{
//	RTcontext context; //!> this is where the instances of the OptiX simulation will be stored
	RTbuffer vertex_buffer_obj; //!> this is where the buffers for the OptiX simulation will be stored
	RTbuffer   index_buffer_obj; //!> this is where the buffers for the OptiX simulation will be stored

//	rtContextCreate( &context );

	nv::Model* model = new nv::Model();

	if(!model->loadModelFromFile(objectInput_filename)) {
		std::cerr << "Unable to load model '" << objectInput_filename << "'" << "...\n";
		exit(-1);
	}

	model->compileModel();

	nv::vec3f modelBBMin, modelBBMax, modelBBCenter;

	model->computeBoundingBox(modelBBMin, modelBBMax);
	modelBBCenter = (modelBBMin + modelBBMax) * 0.5;

	try {
		optix::Context        rtContext;
		rtContext = optix::Context::create();
		rtContext->setRayTypeCount(1);
		rtContext->setEntryPointCount(1);
	
		rtContext["scene_epsilon"]->setFloat( 1e-3f);
		// Limit number of devices to 1 as this is faster for this particular sample.
//		std::vector<int> enabled_devices = rtContext->getEnabledDevices();
//		rtContext->setDevices(enabled_devices.begin(), enabled_devices.begin()+1);

		optix::Geometry rtModel = rtContext->createGeometry();
		rtModel->setPrimitiveCount( model->getCompiledIndexCount()/3 );

		char path_to_ptx_RayGeneration[512];
		sprintf( path_to_ptx_RayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "macrosim_tracer_generated_rayGeneration_checkVisibility.cu.ptx" );

		rtModel->setIntersectionProgram( rtContext->createProgramFromPTXFile( path_to_ptx_RayGeneration, "mesh_intersect" ) );
		rtModel->setBoundingBoxProgram( rtContext->createProgramFromPTXFile( path_to_ptx_RayGeneration, "mesh_bounds" ) );

		GLuint modelVB;
		GLuint modelIB;

		int num_vertices = model->getCompiledVertexCount();
		optix::Buffer vertex_buffer = rtContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		vertex_buffer->setFormat(RT_FORMAT_USER);
		vertex_buffer->setElementSize(3*2*sizeof(float));
		vertex_buffer->setSize(num_vertices);
		rtModel["vertex_buffer"]->setBuffer(vertex_buffer);
		void* h_vertex_buffer=vertex_buffer->map();
		memcpy(h_vertex_buffer, model->getCompiledVertices(), model->getCompiledVertexCount()*model->getCompiledVertexSize()*sizeof(float));
		vertex_buffer->unmap();


		optix::Buffer index_buffer = rtContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		index_buffer->setFormat(RT_FORMAT_INT3);
		index_buffer->setSize(model->getCompiledIndexCount()/3);
		rtModel["index_buffer"]->setBuffer(index_buffer);
		void* h_index_buffer=index_buffer->map();
		memcpy(h_index_buffer, model->getCompiledIndices(), model->getCompiledIndexCount()*sizeof(int));
		index_buffer->unmap();
	} catch ( optix::Exception& e ) {
		cout << "error in MacroSimTracer.checkVisibility(): " << ( e.getErrorString().c_str() ) << endl;
		return false;
	}

	return true;
};

ostream* MacroSimTracer::getCout()
{
	return &cout;
}
