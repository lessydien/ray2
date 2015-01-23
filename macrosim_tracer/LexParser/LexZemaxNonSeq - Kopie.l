%option noyywrap
%{
  // Übersetzt wird das ganze mit flex++ xml.lex und dann linken mit -lfl
  // flex++ xml.lex; g++ lex.yy.cc -lfl; ./a.out < beispiel.xml
  // flex++ ito.lex; g++ lex.yy.cc -lfl; . do.sh > test

  // Vorgehen:
  // 1. Das automatische Anlegen der Objekte für den Bereicht OBJECT hinkriegen
  // 2. Den ganzen Kram mehr oder weniger kopieren und für MeasurementEntity machen
  // 3. Den Kram in unsere Task Klasse einbauen / dazulinken / Makefile
  // 4. Dokumentation
  // 5. Die ganzen set/get Funktionen für Task machen, so dass bequemes Interface
  //    vorhanden ist.
  // 6. Definition der ganzen Variablen, die für uns relevant sind
  // 7. Output 
  // 8. Tests für Mikrolinsen und ein paar andere Objekte
  // Fehlerdetektion in den Input Files !

#include<stdio.h>
#include <iostream>
#include <stdlib.h>
#include "FlexZemax.h"

char yylval[20];

extern FILE *yyin;

extern FILE *yyout;

char testchar[20];
double testdouble;

SimulationType simType=SIMTYPE_GEOM_RAYS; // until we know of something else, we assume geometric raytracing

parseGlassResultStruct *glassResultStructPtr;
short glassParamsIndex=0;
char search_glassName[PARSERR_LENGTH];

ParseGratingResultStruct *gratingResultStructPtr;
int gratingOrderIndex=0;
int gratingEffIndexOrder=0;
int gratingEffIndexLambda=0;
int gratingLambdaIndex=0;

// flag to signal error inside yylex()
bool LexerErr;
// flag to signal that glass name was found in glass catalog
bool glassFound;

ZemaxParamStruct *surfaceArray;
/* index counting the zemax surfaces. init to -1 even though we start with index 0 later */
long long int surfIndex = -1;
/* index counting the surface details. init to -1 even though we start with index 0 later */
long long int surfDetIndex = -1;
/* index counting the rayTrace geometries ( remember that e.g. coordinate breaks are Zemax surfaces but no raytRace geometries */
long long geomIndex = 0;
/* index counting the rayTrace sources */
long long int srcIndex = 0;
/* index counting the detectors */
long long detIndex = 0;
/* index counting the pupil definitions in the system */
long long pupilIndex = 0;

long long int faceIndex = 0;

double3 globalRoot;
double3 globalNormal;

char teststring[5];

long long int surfLength=0;

long long int refObjIndex=0;

long long int asphere_coeff_index=0;
int diffOrderIndex=0;
int diffEffIndex=0;

double scale2m=0.001;
double scale2mm=1;
double scale2um=1000;

simMode l_mode;

// flags to check wether all geometry parameters were defined
	bool geomParseFlag_ObjName=false;
	bool geomParseFlag_ObjType=false;
	bool geomParseFlag_DetPixX=false;
	bool geomParseFlag_DetPixY=false;
	bool geomParseFlag_DetType=false;
	bool geomParseFlag_SrcRayNrHeight=false;
	bool geomParseFlag_SrcRayNrWidth=false;
	bool geomParseFlag_SrcPower=false;
	bool geomParseFlag_SrcDirDist=false;
	bool geomParseFlag_SrcPosDist=false;
	bool geomParseFlag_SrcLambda=false;
	bool geomParseFlag_RefObj=false;
	bool geomParseFlag_Tilt=false;
	bool geomParseFlag_PosMtrxZ=false;
	bool geomParseFlag_PosMtrxY=false;
	bool geomParseFlag_PosMtrxX=false;
	bool geomParseFlag_Mat=false;
	bool geomParseFlag_Immerse=false;
	bool geomParseFlag_Face=false;
	bool geomParseFlag_Scatter=false;
	bool geomParseFlag_Coating=false;
	bool geomParseFlag_Rad1=false;
	bool geomParseFlag_Rad2=false;
	bool geomParseFlag_AsphCoeff=false;
	bool geomParseFlag_Con2=false;
	bool geomParseFlag_Con1=false;
    bool geomParseFlag_Apert1X=false;
    bool geomParseFlag_Apert1Y=false;
    bool geomParseFlag_Apert2X=false;
    bool geomParseFlag_Apert2Y=false;
    bool geomParseFlag_MinApert1X=false;
    bool geomParseFlag_MinApert1Y=false;
    bool geomParseFlag_ApertType=false;
    bool geomParseFlag_IdealLenseLambda0=false;
    bool geomParseFlag_IdealLenseF0=false;
    bool geomParseFlag_IdealLenseA=false;
    bool geomParseFlag_DiffLines=false;
    bool geomParseFlag_DiffOrder=false;
    bool geomParseFlag_DiffEffs=false;
    bool geomParseFlag_Thickness=false;
    bool geomParseFlag_SrcCoherence=false;
    bool geomParseFlag_ImpAreaXmin=false;
    bool geomParseFlag_ImpAreaXmax=false;
    bool geomParseFlag_ImpAreaYmin=false;
    bool geomParseFlag_ImpAreaYmax=false;
    bool geomParseFlag_SrcDiffNr=false;
    bool geomParseFlag_CosNormPeriod=false;
    bool geomParseFlag_CosNormAmpl=false;
    bool geomParseFlag_CosNormAxis=false;
    bool geomParseFlag_FilterMax=false;
    bool geomParseFlag_FilterMin=false;
    bool geomParseFlag_NrRaysPerPix=false;

/* flag to indicate the last surface in parsing the surface summary */
int lastSurfSumFlag=0;


%}

%s GEOMETRY
%s GLASS
%s GRATING

%s GRATINGDET
%s GRATINGDET_NRLAMBDA
%s GRATINGDET_LAMBDA
%s GRATINGDET_NRORDERS
%s GRATINGDET_ORDERS
%s GRATINGDET_RTP01Ptr
%s GRATINGDET_RTP10Ptr
%s GRATINGDET_RTS01Ptr
%s GRATINGDET_RTS10Ptr
%s GRATINGDET_G


%s GLASSDET
%s GLASSDET_NAME
%s GLASSDET_FORMULA
%s GLASSDET_RANGEMIN
%s GLASSDET_RANGEMAX

%s GLASSDET_NOM
%s GLASSDET_DENOM

%s GENERAL_LENS_DATA
%s GENERAL_LENS_DATA_UNITS

%s OBJDET
%s OBJDET_ERROR
%s OBJDET_OBJNR
%s OBJDET_OBJDEF
%s OBJDET_OBJTYPE
%s OBJDET_OBJTYPE_END

%s OBJDET_SRC_REYNR_ANALYSE_WIDTH
%s OBJDET_SRC_REYNR_ANALYSE_HEIGHT
%s OBJDET_SRC_REYNR_LAYOUT_WIDTH
%s OBJDET_SRC_REYNR_LAYOUT_HEIGHT
%s OBJDET_SRC_DIFFRAY_NRDIRECTIONS
%s OBJDET_SRC_PWR
%s OBJDET_SRC_COHERENCE
%s OBJDET_SRC_DIRDIS_UNIFORMY
%s OBJDET_SRC_DIRDIS_UNIFORMX
%s OBJDET_SRC_DIRDIS
%s OBJDET_SRC_DIRDIS_RAND
%s OBJDET_SRC_DIRDIS_RANDXMAX
%s OBJDET_SRC_DIRDIS_RANDXMIN
%s OBJDET_SRC_DIRDIS_RANDYMAX
%s OBJDET_SRC_DIRDIS_RANDYMIN
%s OBJDET_SRC_DIRDIS_GRIDRECT
%s OBJDET_SRC_DIRDIS_GRIDRECT_XMIN
%s OBJDET_SRC_DIRDIS_GRIDRECT_XMAX
%s OBJDET_SRC_DIRDIS_GRIDRECT_YMIN
%s OBJDET_SRC_DIRDIS_GRIDRECT_YMAX
%s OBJDET_SRC_POSDIS
%s OBJDET_SRC_LAMBDA

%s OBJDET_A
%s OBJDET_F0
%s OBJDET_LAMBDA0

%s OBJDET_DET_TYPE
%s OBJDET_DET_PIXELY
%s OBJDET_DET_PIXELX
%s OBJDET_DET_RAYSPERPIXELX
%s OBJDET_DET_RAYSPERPIXELY
%s OBJDET_DET_PIXELDIRX
%s OBJDET_DET_PIXELDIRY
%s OBJDET_DET_DIRXHALF
%s OBJDET_DET_DIRYHALF

%s OBJDET_TILTX
%s OBJDET_TILTY
%s OBJDET_TILTZ

%s OBJDET_POSX_1
%s OBJDET_POSX_2
%s OBJDET_POSX_3
%s OBJDET_POSX
%s OBJDET_POSY_1
%s OBJDET_POSY_2
%s OBJDET_POSY_3
%s OBJDET_POSY
%s OBJDET_POSZ_1
%s OBJDET_POSZ_2
%s OBJDET_POSZ_3
%s OBJDET_POSZ

%s OBJDET_REFOBJ

%s OBJDET_MAT
%s OBJDET_MAT_N2
%s OBJDET_MAT_N1
%s OBJDET_IMM
%s OBJDET_MATFILTER
%s OBJDET_MATFILTER_MAX
%s OBJDET_MATFILTER_MIN

%s OBJDET_IMPAREA
%s OBJDET_IMPAREA_OBJECT
%s OBJDET_IMPAREA_ALPHAXMAX
%s OBJDET_IMPAREA_ALPHAXMIN
%s OBJDET_IMPAREA_ALPHAYMAX
%s OBJDET_IMPAREA_ALPHAYMIN

%s OBJDET_FACE

%s OBJDET_SCAT
%s OBJDET_SCAT_TORSPARR1D
%s OBJDET_SCAT_TORSPARR1D_KDL
%s OBJDET_SCAT_TORSPARR1D_KSL
%s OBJDET_SCAT_TORSPARR1D_KSP
%s OBJDET_SCAT_TORSPARR1D_SIGMASL
%s OBJDET_SCAT_TORSPARR1D_SIGMASP
%s OBJDET_SCAT_TORSPARR1D_SCATAXISX
%s OBJDET_SCAT_TORSPARR1D_SCATAXISY
%s OBJDET_SCAT_TORSPARR1D_SCATAXISZ

%s OBJDET_SCAT_TORSPARR2D
%s OBJDET_SCAT_TORSPARR2D_KDL
%s OBJDET_SCAT_TORSPARR2D_KSL
%s OBJDET_SCAT_TORSPARR2D_KSP
%s OBJDET_SCAT_TORSPARR2D_SIGMASL
%s OBJDET_SCAT_TORSPARR2D_SIGMASP
%s OBJDET_SCAT_TORSPARR2D_SCATAXISX
%s OBJDET_SCAT_TORSPARR2D_SCATAXISY
%s OBJDET_SCAT_TORSPARR2D_SCATAXISZ

%s OBJDET_SCAT_DOUBLECAUCHY1D
%s OBJDET_SCAT_DOUBLECAUCHY1D_KSL
%s OBJDET_SCAT_DOUBLECAUCHY1D_KSP
%s OBJDET_SCAT_DOUBLECAUCHY1D_GAMMASL
%s OBJDET_SCAT_DOUBLECAUCHY1D_GAMMASP
%s OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISX
%s OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISY
%s OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISZ
%s OBJDET_SCAT_DOUBLECAUCHY1D_CYSL
%s OBJDET_SCAT_DOUBLECAUCHY1D_AYSL
%s OBJDET_SCAT_DOUBLECAUCHY1D_AYSP
%s OBJDET_SCAT_DOUBLECAUCHY1D_CYSP
%s OBJDET_SCAT_DOUBLECAUCHY1D_CKSL
%s OBJDET_SCAT_DOUBLECAUCHY1D_CKSP
%s OBJDET_SCAT_DOUBLECAUCHY1D_AKSL
%s OBJDET_SCAT_DOUBLECAUCHY1D_AKSP

%s OBJDET_SCAT_LAMBERT2D_TIR


%s OBJDET_COAT
%s OBJDET_COAT_COEFFR
%s OBJDET_COAT_COEFFT
%s OBJDET_COAT_COEFFAR
%s OBJDET_COAT_COEFFCR
%s OBJDET_COAT_COEFFAT
%s OBJDET_COAT_COEFFCT

%s OBJDET_XHALFWIDTH
%s OBJDET_YHALFWIDTH
%s OBJDET_MAXAPRT
%s OBJDET_MINAPRTX
%s OBJDET_MINAPRTY
%s OBJDET_BACKAPRT
%s OBJDET_FRONTAPRT
%s OBJDET_EDGE
%s OBJDET_EDGE1
%s OBJDET_EDGE2
%s OBJDET_MAXAPRT_X
%s OBJDET_MAXAPRT_Y
%s OBJDET_RADAPRT
%s OBJDET_APRTTYPE
%s OBJDET_LINES
%s OBJDET_DIFFORDER
%s OBJDET_EFFS

%s OBJDET_CON
%s OBJDET_CON1
%s OBJDET_CON2

%s OBJDET_RAD
%s OBJDET_RAD1
%s OBJDET_RAD2
%s OBJDET_ASPH_COEFF
%s OBJDET_ASPH_PAR

%s OBJDET_THK
%s OBJDET_ZLEN

%s OBJDET_COSINENORMAL_AMPL
%s OBJDET_COSINENORMAL_PERIOD
%s OBJDET_COSINENORMAL_AXISX
%s OBJDET_COSINENORMAL_AXISY
%s OBJDET_COSINENORMAL_AXISZ
%s OBJDET_COSINENORMAL_ITACCU


ws      [ \t]+

string  \"[^\n"]+\"
alpha   [A-Za-z]
dig     [0-9]
name    ({alpha}|{dig}|\$)({alpha}|{dig}|[_.\-/$])*
typename "STRING"{name}
num1    [-+]?{dig}+\.?([eE][-+]?{dig}+)?
num2    [-+]?{dig}*\.{dig}+([eE][-+]?{dig}+)?
number  {num1}|{num2}
numberunit  {number}|{number}mm|{number}mu

int     [+-]?{dig}+
num     -?(([0-9]+)|([0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?)
surfnum	 [0-9]+|"OBJ"|"STO"
surflast "IMA"
glass    "MT_MIRROR"|"N-MT_BK7"
asphere_coeff "Coeff on r "{dig}+

%%

"System/Prescription Data" {
		  BEGIN GEOMETRY;
		  }
		  
"CC" {
		  fprintf( yyout, "found beginning of glass library \n");
		  BEGIN	GLASS;
		  }
		  
"GRATING DATA" {
		  BEGIN GRATINGDET;
		  }
		  
".*" {
		fprintf( yyout, "surprised finding %s \n", yytext);
	 }
		  
 /* -----------------------Grating Parser--------------------------*/

<GRATINGDET>"number of wavelengths:" {
		  fprintf( yyout, "nr of wavelengths:");
		  BEGIN GRATINGDET_NRLAMBDA;
		  }

<GRATINGDET_NRLAMBDA>{number} {
		  fprintf( yyout, " %i;", atoi(yytext));
		  gratingResultStructPtr->nrWavelengths=atoi(yytext);
		  BEGIN GRATINGDET;
		  }		  
		  
<GRATINGDET>"number of diffraction orders:" {
		  fprintf( yyout, "nr of diff orders:");
		  BEGIN GRATINGDET_NRORDERS;
		  }

<GRATINGDET_NRORDERS>{number} {
		  fprintf( yyout, " %i;", atoi(yytext));
		  gratingResultStructPtr->nrOrders=atoi(yytext);
		  BEGIN GRATINGDET;
		  }		  

<GRATINGDET>"diffraction orders:" {
		  fprintf( yyout, "diff orders: ");
		  gratingOrderIndex=0;
		  gratingResultStructPtr->diffOrdersPtr=(short*) calloc(gratingResultStructPtr->nrOrders,sizeof(short));
		  BEGIN GRATINGDET_ORDERS;
		  }
		  
<GRATINGDET_ORDERS>{number} {
		fprintf(yyout, "%i, ", atoi(yytext));
		gratingResultStructPtr->diffOrdersPtr[gratingOrderIndex]=atoi(yytext);
		gratingOrderIndex++;
		if (gratingOrderIndex>=gratingResultStructPtr->nrOrders)
		{	
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET;
		}
		else
			BEGIN GRATINGDET_ORDERS;
		}	

 /*-------------------------RTp01---------------------------------------*/

<GRATINGDET>"RTp01:" {
		  fprintf( yyout, "RTp01: ");
		  gratingEffIndexLambda=0;
		  gratingEffIndexOrder=0;
		  gratingResultStructPtr->RTP01Ptr=(double*) calloc(gratingResultStructPtr->nrOrders*gratingResultStructPtr->nrWavelengths,sizeof(double));
		  BEGIN GRATINGDET_RTP01Ptr;
		  }			

<GRATINGDET_RTP01Ptr>{number} {
		fprintf(yyout, "%f, ", atof(yytext));
		gratingResultStructPtr->RTP01Ptr[gratingEffIndexLambda+gratingResultStructPtr->nrWavelengths*gratingEffIndexOrder]=atof(yytext);
		gratingEffIndexLambda++;
		if (gratingEffIndexLambda>=gratingResultStructPtr->nrWavelengths)
		{	
			gratingEffIndexLambda=0;
			gratingEffIndexOrder++;
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET_RTP01Ptr;
		}
		if (gratingEffIndexOrder>=gratingResultStructPtr->nrOrders)
		{
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET;
		}		
		}	

 /*-------------------------RTp10---------------------------------------*/

<GRATINGDET>"RTp10:" {
		  fprintf( yyout, "RTp10: ");
		  gratingEffIndexLambda=0;
		  gratingEffIndexOrder=0;
		  gratingResultStructPtr->RTP10Ptr=(double*) calloc(gratingResultStructPtr->nrOrders*gratingResultStructPtr->nrWavelengths,sizeof(double));
		  BEGIN GRATINGDET_RTP10Ptr;
		  }			

<GRATINGDET_RTP10Ptr>{number} {
		fprintf(yyout, "%f, ", atof(yytext));
		gratingResultStructPtr->RTP10Ptr[gratingEffIndexLambda+gratingResultStructPtr->nrWavelengths*gratingEffIndexOrder]=atof(yytext);
		gratingEffIndexLambda++;
		if (gratingEffIndexLambda>=gratingResultStructPtr->nrWavelengths)
		{	
			gratingEffIndexLambda=0;
			gratingEffIndexOrder++;
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET_RTP10Ptr;
		}
		if (gratingEffIndexOrder>=gratingResultStructPtr->nrOrders)
		{
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET;
		}		
		}	

 /*-------------------------RTs01---------------------------------------*/

<GRATINGDET>"RTs01:" {
		  fprintf( yyout, "RTs01: ");
		  gratingEffIndexLambda=0;
		  gratingEffIndexOrder=0;
		  gratingResultStructPtr->RTS01Ptr=(double*) calloc(gratingResultStructPtr->nrOrders*gratingResultStructPtr->nrWavelengths,sizeof(double));
		  BEGIN GRATINGDET_RTS01Ptr;
		  }			

<GRATINGDET_RTS01Ptr>{number} {
		fprintf(yyout, "%f, ", atof(yytext));
		gratingResultStructPtr->RTS01Ptr[gratingEffIndexLambda+gratingResultStructPtr->nrWavelengths*gratingEffIndexOrder]=atof(yytext);
		gratingEffIndexLambda++;
		if (gratingEffIndexLambda>=gratingResultStructPtr->nrWavelengths)
		{	
			gratingEffIndexLambda=0;
			gratingEffIndexOrder++;
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET_RTS01Ptr;
		}
		if (gratingEffIndexOrder>=gratingResultStructPtr->nrOrders)
		{
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET;
		}		
		}	

 /*-------------------------RTs10---------------------------------------*/

<GRATINGDET>"RTs10:" {
		  fprintf( yyout, "RTs10: ");
		  gratingEffIndexLambda=0;
		  gratingEffIndexOrder=0;
		  gratingResultStructPtr->RTS10Ptr=(double*) calloc(gratingResultStructPtr->nrOrders*gratingResultStructPtr->nrWavelengths,sizeof(double));
		  BEGIN GRATINGDET_RTS10Ptr;
		  }			

<GRATINGDET_RTS10Ptr>{number} {
		fprintf(yyout, "%f, ", atof(yytext));
		gratingResultStructPtr->RTS10Ptr[gratingEffIndexLambda+gratingResultStructPtr->nrWavelengths*gratingEffIndexOrder]=atof(yytext);
		gratingEffIndexLambda++;
		if (gratingEffIndexLambda>=gratingResultStructPtr->nrWavelengths)
		{	
			gratingEffIndexLambda=0;
			gratingEffIndexOrder++;
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET_RTS10Ptr;
		}
		if (gratingEffIndexOrder>=gratingResultStructPtr->nrOrders)
		{
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET;
		}		
		}		

 /*-------------------------lambda---------------------------------------*/
				
<GRATINGDET>"lambda:" {
		  fprintf( yyout, "lambda: ");
		  gratingLambdaIndex=0;
		  gratingResultStructPtr->lambdaPtr=(double*) calloc(gratingResultStructPtr->nrWavelengths,sizeof(double));
		  BEGIN GRATINGDET_LAMBDA;
		  }			

<GRATINGDET_LAMBDA>{number} {
		fprintf(yyout, "%f, ", atof(yytext));
		gratingResultStructPtr->lambdaPtr[gratingLambdaIndex]=atof(yytext)*1e-3;
		gratingLambdaIndex++;
		if (gratingLambdaIndex>=gratingResultStructPtr->nrWavelengths)
		{	
			fprintf(yyout, ";\n");
			BEGIN GRATINGDET;
		}
		else
			BEGIN GRATINGDET_LAMBDA;
		}				

<GRATINGDET_LAMBDA>"\n" {
		fprintf(yyout, "error!!! number of wavelengths is less than specified");
		std::cout << "error in yylex(): number of wavelentghs is less than specified" << std::endl;
		LexerErr=true;
		BEGIN GRATINGDET;
		}
		

 /*-------------------------grating constant------------------------------------*/
				
<GRATINGDET>"grating constant:" {
		  fprintf( yyout, "g: ");
		  BEGIN GRATINGDET_G;
		  }			

<GRATINGDET_G>{number} {
		fprintf(yyout, "%f, ", atof(yytext));
		gratingResultStructPtr->g=atof(yytext)*1e-3;
		BEGIN GRATINGDET;
		}			
			
 /*----------------------------------------------------------------*/				  
 /* -----------------------Glass Parser----------------------------*/
 /*----------------------------------------------------------------*/
 
<GLASS>"NM " {
		fprintf( yyout, "found beginning of glass definition");
		BEGIN GLASSDET_NAME;
		}
		
<GLASSDET_NAME>{name} {
		fprintf( yyout, "name: %s", yytext );
		// if we found the glass we were looking for
		if (!strcmp(yytext, search_glassName))
		{
			fprintf( yyout, "name: %s", yytext);
			//sprintf( glassResultStructPtr->glassName, yytext);
			glassFound=true;
			BEGIN GLASSDET_FORMULA;
		}
		else
			BEGIN GLASS;
		}
		
<GLASSDET_FORMULA>{dig}+ {
		fprintf( yyout, "formula: %i;", atoi(yytext));		
		glassResultStructPtr->dispersionFormulaIndex=atoi(yytext);
		BEGIN GLASSDET;
		}

 /* ------------------------ wavelength range -------------------------*/
		
<GLASSDET>"LD " {
		fprintf(yyout, "wavelength range: ");
		BEGIN GLASSDET_RANGEMIN;
		}
		
<GLASSDET_RANGEMIN>{number} {
		fprintf(yyout, "%f, ", atof(yytext)*scale2um);
		glassResultStructPtr->lambdaMin=atof(yytext);
		BEGIN GLASSDET_RANGEMAX;
		}

<GLASSDET_RANGEMAX>{number} {
		fprintf(yyout, "%f \n", atof(yytext));
		glassResultStructPtr->lambdaMax=atof(yytext);
		BEGIN GLASS;
		}

 /* ------------------------- nominator of dispersion formula ----------*/
 
 <GLASSDET>"CD " {
		fprintf(yyout, "nominator: ");
		glassParamsIndex=0;
		BEGIN GLASSDET_NOM;
		}
		
<GLASSDET_NOM>{number} {
		fprintf(yyout, "%f ", atof(yytext));
		glassResultStructPtr->paramsNom[glassParamsIndex]=atof(yytext);
		glassParamsIndex++;
		if (glassParamsIndex>5)
			BEGIN GLASSDET;
		else
			BEGIN GLASSDET_NOM;
		}		
		
 /* ------------------------- denominator of dispersion formula ----------*/
 
 <GLASSDET>"TD " {
		fprintf(yyout, "denominator: ");
		glassParamsIndex=0;
		BEGIN GLASSDET_DENOM;
		}
		
<GLASSDET_DENOM>{number} {
		fprintf(yyout, "%f ", atof(yytext));
		glassResultStructPtr->paramsDenom[glassParamsIndex]=atof(yytext);
		glassParamsIndex++;
		if (glassParamsIndex>5)
			BEGIN GLASSDET;
		else
			BEGIN GLASSDET_DENOM;
		}				
 /*-----------------------------------------------------------------*/
 /* -----------------------Geometry Parser------------------------- */
 /*-----------------------------------------------------------------*/

<GEOMETRY>"GENERAL LENS DATA:" {
           BEGIN GENERAL_LENS_DATA;
          }
          
<GEOMETRY>"OBJECT DATA DETAIL:" {
		   BEGIN OBJDET;
		  }      
		  

 /* ---------------------------------------------------------------*/
 
 <GENERAL_LENS_DATA>"Lens Units"{ws}*":" {
			fprintf( yyout, "units:"); 
			BEGIN GENERAL_LENS_DATA_UNITS;
 }
	<GENERAL_LENS_DATA_UNITS>"Millimeters" {
			fprintf(yyout, " mm;");
			scale2mm=1;
			BEGIN GEOMETRY;
	}

 /*--------------------------parse object type------------------------------------------*/

 <OBJDET>"There are " {
			 fprintf( yyout, "number of objects:");
			 BEGIN OBJDET_OBJNR;
 }
		<OBJDET_OBJNR>{dig}+ {
				 fprintf ( yyout, " %s;", yytext);
 				 surfLength=atoi(yytext);
				 surfaceArray = new ZemaxParamStruct[surfLength];
				 long long int i;
				 for (i=0; i<surfLength; i++)
				 {
					initSurfaceStruct(&surfaceArray[i]);
				 }
				 BEGIN OBJDET;
		}

 <OBJDET>"Object  " {
			 // check wether surf Length is big enough to hold this object
			 if (surfIndex>=(surfLength-1))
			 {
				fprintf( yyout, "error: object number %i is higher than number of objects %i !!!", (surfIndex+1), surfLength);
				std::cout << "error in yylex(): object number " << (surfIndex+1) << " is higher than number of objects " << surfLength << std::endl;
				LexerErr=true;
				BEGIN OBJDET_ERROR;
			 }
			 else
			 {
				 // check wether old object was fully defined
				 if (surfIndex>=0)
				 {
					 if (!checkObjectDefinition(&surfaceArray[surfIndex]) )
					 {
						fprintf( yyout, "error: object %i was not fully defined!!!", (surfIndex+1));
						std::cout << "error in yylex(): object " << (surfIndex+1) << " was not fully defined" << std::endl;
						LexerErr=true;
					 }
				 }
				 // init parser flags for upcoming object
				 initGeomParseFlags();
				 fprintf( yyout, "\n object");
				 faceIndex=0; // init face index of current object
				 BEGIN OBJDET_OBJDEF;
			}
 }

		<OBJDET_OBJDEF>{dig}+ {
				 fprintf ( yyout, " %s:", yytext);
				 surfIndex=atoi(yytext);
				 surfIndex=surfIndex-1; // we start counting with zero. Zemax starts with 1
				 geomIndex=geomIndex+1;
				 if (surfIndex >= surfLength)
				 {
					// implement some error mechanism
					std::cout << "error in yylex(): number of objects found in prescription file exceeds prescribed number of objects" << std::endl;
					LexerErr=true;
				 }
				 BEGIN OBJDET_OBJDEF;
		}
		
		<OBJDET_OBJDEF>{name} {
				 geomParseFlag_ObjName=true;
				 fprintf(yyout, "%s;", yytext);
				 sprintf(surfaceArray[surfIndex].comment, "%s", yytext);
				 BEGIN OBJDET;
		}

		<OBJDET_OBJDEF>"\n" {
				 fprintf(yyout, "warning: no name definition found!!;");
				 BEGIN OBJDET;
		}
 
  <OBJDET>"Object type" {
			 fprintf(yyout, " Object Type: ");
			 geomParseFlag_ObjType=true;
			 BEGIN OBJDET_OBJTYPE;
  }

		<OBJDET_OBJTYPE>"Biconic Lens" {
				 fprintf(yyout, " Biconin Lens;");
				 surfaceArray[surfIndex].type=OT_BICONLENSE;
				 surfaceArray[surfIndex].details.aperture=AT_RECT; // biconic lenses have rectangular aperture per default
				 geomIndex=geomIndex+1; // biconic lenses consist of not only 1 but at least 2 OptiX geometries ( front surface, back surface). Therefore we have to add one more geometrie here !!				 
				 // we ignore the side face for now...
//				 if (l_mode==SIM_GEOMRAYS_NONSEQ)
//					geomIndex=geomIndex+1; // standard lenses consist of one additional OptiX geometries ( pipe/rectPipe ). Therefore we have to add one more geometrie here !!
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Cosine Normal" {
				 fprintf(yyout, " Cosine Normal;");
				 surfaceArray[surfIndex].type=OT_COSINENORMAL;
				 surfaceArray[surfIndex].details.aperture=AT_ELLIPT; // cosine normals by default
				 BEGIN OBJDET_OBJTYPE_END;
		}
  
		<OBJDET_OBJTYPE>"Ideal Lense" {
				 fprintf(yyout, " Ideal Lense;");
				 surfaceArray[surfIndex].type=OT_IDEALLENSE;
				 surfaceArray[surfIndex].details.aperture=AT_ELLIPT; // standard lenses have elliptical aperture
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Differential Source Freeform" {
				 fprintf(yyout, " Differential Source Freeform;");
				 surfaceArray[surfIndex].type=OT_DIFFSRC_FREEFORM;
				 geomIndex=geomIndex-1; // we count an object of type source not as an geometry. Therefore we have to subtract this object from the geometryIndex after having added it when detecting "Object"
				 srcIndex=srcIndex+1; // instead we increment sourceIndex
				 simType=SIMTYPE_DIFF_RAYS;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Differential Source Holo" {
				 fprintf(yyout, " Differential Source Holo;");
				 surfaceArray[surfIndex].type=OT_DIFFSRC_HOLO;
				 geomIndex=geomIndex-1; // we count an object of type source not as an geometry. Therefore we have to subtract this object from the geometryIndex after having added it when detecting "Object"
				 srcIndex=srcIndex+1; // instead we increment sourceIndex
				 simType=SIMTYPE_DIFF_RAYS;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Differential Source" {
				 fprintf(yyout, " Differential Source;");
				 surfaceArray[surfIndex].type=OT_DIFFSRC;
				 geomIndex=geomIndex-1; // we count an object of type source not as an geometry. Therefore we have to subtract this object from the geometryIndex after having added it when detecting "Object"
				 srcIndex=srcIndex+1; // instead we increment sourceIndex
				 simType=SIMTYPE_DIFF_RAYS;
				 BEGIN OBJDET_OBJTYPE_END;
		}
		
		<OBJDET_OBJTYPE>"Differential RayAiming Source" {
				 fprintf(yyout, " Differential RayAiming Source;");
				 surfaceArray[surfIndex].type=OT_DIFFSRC_RAYAIM;
				 geomIndex=geomIndex-1; // we count an object of type source not as an geometry. Therefore we have to subtract this object from the geometryIndex after having added it when detecting "Object"
				 srcIndex=srcIndex+1; // instead we increment sourceIndex
				 simType=SIMTYPE_DIFF_RAYS;
				 BEGIN OBJDET_OBJTYPE_END;
		}		

		<OBJDET_OBJTYPE>"PathTracing Source" {
				 fprintf(yyout, " PathTracing Source;");
				 surfaceArray[surfIndex].type=OT_PATHTRACINGSRC;
				 geomIndex=geomIndex; // we do count an object of type source not as an geometry.
				 srcIndex=srcIndex+1; // instead we increment sourceIndex
				 simType=SIMTYPE_PATHTRACING;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Illumination Pupil" {
				 fprintf(yyout, " Illumination Pupil;");
				 surfaceArray[surfIndex].type=OT_ILLPUPIL;
				 geomIndex=geomIndex; // we do count an object of type pupil not as an geometry.
				 pupilIndex=pupilIndex+1;
				 BEGIN OBJDET_OBJTYPE_END;
		}		

		
		<OBJDET_OBJTYPE>"Source" {
				 fprintf(yyout, " Source;");
				 surfaceArray[surfIndex].type=OT_SRC;
				 geomIndex=geomIndex-1; // we count an object of type source not as an geometry. Therefore we have to subtract this object from the geometryIndex after having added it when detecting "Object"
				 srcIndex=srcIndex+1; // instead we increment sourceIndex
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Standard Lens" {
				 fprintf(yyout, " Standard Lens;");
				 surfaceArray[surfIndex].type=OT_STLENSE;
				 surfaceArray[surfIndex].details.aperture=AT_ELLIPT; // standard lenses have elliptical aperture
				 geomIndex=geomIndex+1; // standard lenses consist of not only 1 but at least 2 OptiX geometries ( front surface, back surface). Therefore we have to add one more geometrie here !!				 
				 if (l_mode==SIM_GEOMRAYS_NONSEQ)
					geomIndex=geomIndex+1; // standard lenses consist of one additional OptiX geometries ( pipe ). Therefore we have to add one more geometrie here !!
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Ellipse" {
				 fprintf(yyout, " Ellipse;");
				 surfaceArray[surfIndex].type=OT_STANDARD;
				 surfaceArray[surfIndex].details.aperture=AT_ELLIPT;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Cylinder Pipe" {
				 fprintf(yyout, " Cyl. Pipe;");
				 surfaceArray[surfIndex].type=OT_CYLPIPE;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Rectangle" {
				 fprintf(yyout, " Rectangle;");
				 surfaceArray[surfIndex].type=OT_STANDARD;
				 surfaceArray[surfIndex].details.aperture=AT_RECT;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Detector Rect" {
				 fprintf(yyout, " Detector rectangle;");
				 surfaceArray[surfIndex].type=OT_STANDARD;
				 surfaceArray[surfIndex].details.detector=true;
				 surfaceArray[surfIndex].details.aperture=AT_RECT;
				 detIndex=detIndex+1;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Standard Surface" {
				 fprintf(yyout, " Rectangle;");
				 surfaceArray[surfIndex].type=OT_STANDARD;
				 surfaceArray[surfIndex].details.aperture=AT_ELLIPT;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Aspheric Surface" {
				 fprintf(yyout, " Aspheric surface;");
				 surfaceArray[surfIndex].type=OT_ASPHSURF;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Aperture Stop" {
				 fprintf(yyout, " Aperture stop;");
				 surfaceArray[surfIndex].type=OT_APTSTOP;
				 surfaceArray[surfIndex].details.aperture=AT_ELLIPT;
				 BEGIN OBJDET_OBJTYPE_END;
		}

		<OBJDET_OBJTYPE>"Diffraction Grating" {
				 fprintf(yyout, " Diffraction Grating;");
				 surfaceArray[surfIndex].type=OT_DGRATING;
				 surfaceArray[surfIndex].details.aperture=AT_ELLIPT; // so far all diffraction gratings are asuumed to have circular aperture !!!
				 // we ignore the two additional surfaces of the grating for now...
				 if (l_mode==SIM_GEOMRAYS_NONSEQ)
					geomIndex=geomIndex+2; // diffraction consist of not only 1 but 3 OptiX geometries ( 1st surface, 2nd surface, pipe ). Therefore we have to add two more geometries here !!
				 BEGIN OBJDET_OBJTYPE_END;
		}

				<OBJDET_OBJTYPE_END>"\n" {
						 //fprintf(yyout, "\n");
						 BEGIN OBJDET;
				}

		<OBJDET_OBJTYPE>"\n" {
				 fprintf(yyout, "warning: no valid object type definition found!!");
				 geomParseFlag_ObjType=false;
				 BEGIN OBJDET;
    	}
 			
 /* ----------------------end object type----------------------- */

 /* ----------------------detector----------------------- */

 <OBJDET>"DirX HalfWidth"{ws}*":" {
			 fprintf(yyout, "detector DirX HalfWidth:");
			 BEGIN OBJDET_DET_DIRXHALF;
  }

		<OBJDET_DET_DIRXHALF>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.dirHalfWidth.x=atof(yytext)/360*2*M_PI;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_DIRXHALF>"\n" {
				 fprintf(yyout, " no dirX HalfWidth definition found!!");
				 surfaceArray[surfIndex].details.dirHalfWidth.x=0;
				 BEGIN OBJDET;
		}
		
 <OBJDET>"DirY HalfWidth"{ws}*":" {
			 fprintf(yyout, "detector DirY HalfWidth:");
			 BEGIN OBJDET_DET_DIRYHALF;
  }

		<OBJDET_DET_DIRYHALF>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.dirHalfWidth.y=atof(yytext)/360*2*M_PI;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_DIRYHALF>"\n" {
				 fprintf(yyout, " no dirY HalfWidth definition found!!");
				 surfaceArray[surfIndex].details.dirHalfWidth.y=0;
				 BEGIN OBJDET;
		}		
 
 <OBJDET>"# X Pixels"{ws}*":" {
			 fprintf(yyout, "detector pixelX:");
			 BEGIN OBJDET_DET_PIXELX;
  }

		<OBJDET_DET_PIXELX>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detPixel.x=atoi(yytext);
				 geomParseFlag_DetPixX=true;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_PIXELX>"\n" {
				 fprintf(yyout, " no detPixelX definition found!!");
				 surfaceArray[surfIndex].details.detPixel.x=0;
				 BEGIN OBJDET;
		}

 <OBJDET>"# Y Pixels"{ws}*":" {
			 fprintf(yyout, "detector pixelY:");
			 BEGIN OBJDET_DET_PIXELY;
  }

		<OBJDET_DET_PIXELY>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detPixel.y=atoi(yytext);
				 geomParseFlag_DetPixY=true;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_PIXELY>"\n" {
				 fprintf(yyout, " no detPixelY definition found!!");
				 surfaceArray[surfIndex].details.detPixel.y=0;
				 BEGIN OBJDET;
		}
		
 <OBJDET>"# DirX Pixels"{ws}*":" {
			 fprintf(yyout, "detector DirPixelX:");
			 BEGIN OBJDET_DET_PIXELDIRX;
  }

		<OBJDET_DET_PIXELDIRX>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detPixel_PhaseSpace.x=atoi(yytext);
				 geomParseFlag_DetPixX=true;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_PIXELDIRX>"\n" {
				 fprintf(yyout, " no detPixelX definition found!!");
				 surfaceArray[surfIndex].details.detPixel_PhaseSpace.x=0;
				 BEGIN OBJDET;
		}

 <OBJDET>"# DirY Pixels"{ws}*":" {
			 fprintf(yyout, "detector DirPixelY:");
			 BEGIN OBJDET_DET_PIXELDIRY;
  }

		<OBJDET_DET_PIXELDIRY>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detPixel_PhaseSpace.y=atoi(yytext);
				 geomParseFlag_DetPixY=true;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_PIXELDIRY>"\n" {
				 fprintf(yyout, " no detPixelY definition found!!");
				 surfaceArray[surfIndex].details.detPixel_PhaseSpace.y=0;
				 BEGIN OBJDET;
		}		
		
 <OBJDET>"Detector Type"{ws}*":" {
			 fprintf(yyout, "detector Type: ");
			 geomParseFlag_DetType=true;
			 BEGIN OBJDET_DET_TYPE;
  } 

		<OBJDET_DET_TYPE>"RAYDATA" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detectorType=DET_RAYDATA;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_TYPE>"RAYDATA_RED" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detectorType=DET_RAYDATA_RED;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_TYPE>"RAYDATA_GLOBAL" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detectorType=DET_RAYDATA_GLOBAL;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_TYPE>"RAYDATA_RED_GLOBAL" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detectorType=DET_RAYDATA_RED_GLOBAL;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_TYPE>"INTENSITY" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detectorType=DET_INTENSITY;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_TYPE>"PHASESPACE" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detectorType=DET_PHASESPACE;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_TYPE>"FIELD" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.detectorType=DET_FIELD;
				 BEGIN OBJDET;
		}

		<OBJDET_DET_TYPE>"\n" {
				 fprintf(yyout, " no detector Type definition found!!");
				 surfaceArray[surfIndex].details.detectorType=DET_UNKNOWN;
				 geomParseFlag_DetType=false;
				 std::cout << "error: no detector Type definition found!!" << std::endl;
				 BEGIN OBJDET;
		}
		
 <OBJDET>"# Rays/Pixel"{ws}*":" {
			 fprintf(yyout, "# rays per pixel: ");
			 BEGIN OBJDET_DET_RAYSPERPIXELX;
  } 
  
  		<OBJDET_DET_RAYSPERPIXELX>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.nrRaysPerPixel.x=atoi(yytext);
				 BEGIN OBJDET_DET_RAYSPERPIXELY;
		}
		
		<OBJDET_DET_RAYSPERPIXELX>"\n" {
				 fprintf(yyout, " no definition of rays per pixel found!!");
				 surfaceArray[surfIndex].details.nrRaysPerPixel=make_ulong2(0,0);
				 std::cout << "error: no definition of rays per pixel found!!" << std::endl;
				 BEGIN OBJDET;
		}
					
  			<OBJDET_DET_RAYSPERPIXELY>{num} {
					 fprintf(yyout, " %s;", yytext);
					 geomParseFlag_NrRaysPerPix=true;
					 surfaceArray[surfIndex].details.nrRaysPerPixel.y=atoi(yytext);
					 BEGIN OBJDET;
			}	
			
				<OBJDET_DET_RAYSPERPIXELY>"\n" {
						 fprintf(yyout, " no definition of rays per pixel found!!");
						 surfaceArray[surfIndex].details.nrRaysPerPixel=make_ulong2(0,0);
						 std::cout << "error: no definition of rays per pixel found!!" << std::endl;
						 BEGIN OBJDET;
				}				


		

 /* ----------------------source----------------------- */
 <OBJDET>"# Layout Rays"{ws}*":" {
			 fprintf(yyout, "raynumber layout:");
			 BEGIN OBJDET_SRC_REYNR_LAYOUT_WIDTH;
  }

		<OBJDET_SRC_REYNR_LAYOUT_WIDTH>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayFieldWidthLayout=atoi(yytext);
				 BEGIN OBJDET_SRC_REYNR_LAYOUT_HEIGHT;
		}

		<OBJDET_SRC_REYNR_LAYOUT_WIDTH>"\n" {
				 fprintf(yyout, " no ray number definition found!!");
				 surfaceArray[surfIndex].details.rayFieldHeightLayout=0;
				 surfaceArray[surfIndex].details.rayFieldWidthLayout=0;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_REYNR_LAYOUT_HEIGHT>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayFieldHeightLayout=atoi(yytext);
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_REYNR_LAYOUT_HEIGHT>"\n" {
				 fprintf(yyout, " only definition of rayfield height found!! we'll assume a rectangular rayfield that bests fit the number of rays specified");
				 surfaceArray[surfIndex].details.rayFieldWidthLayout=(long long)(sqrtf(surfaceArray[surfIndex].details.rayFieldHeightLayout));
				 surfaceArray[surfIndex].details.rayFieldHeightLayout=(long long)(sqrtf(surfaceArray[surfIndex].details.rayFieldHeightLayout));
				 BEGIN OBJDET;
		}
 
 <OBJDET>"# Analysis Rays"{ws}*":" {
			 fprintf(yyout, "raynumber analysis:");
			 BEGIN OBJDET_SRC_REYNR_ANALYSE_WIDTH;
  }

		<OBJDET_SRC_REYNR_ANALYSE_WIDTH>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayFieldWidth=atoi(yytext);
				 geomParseFlag_SrcRayNrWidth=true;
				 BEGIN OBJDET_SRC_REYNR_ANALYSE_HEIGHT;
		}

		<OBJDET_SRC_REYNR_ANALYSE_WIDTH>"\n" {
				 fprintf(yyout, " no ray number definition found!!");
				 surfaceArray[surfIndex].details.rayFieldHeight=0;
				 surfaceArray[surfIndex].details.rayFieldWidth=0;
				 geomParseFlag_SrcRayNrWidth=false;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_REYNR_ANALYSE_HEIGHT>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayFieldHeight=atoi(yytext);
				 geomParseFlag_SrcRayNrHeight=true;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_REYNR_ANALYSE_HEIGHT>"\n" {
				 fprintf(yyout, " only definition of rayfield height found!! we'll assume a rectangular rayfield that bests fit the number of rays specified");
				 surfaceArray[surfIndex].details.rayFieldWidth=(long long)(sqrtf(surfaceArray[surfIndex].details.rayFieldHeight));
				 surfaceArray[surfIndex].details.rayFieldHeight=(long long)(sqrtf(surfaceArray[surfIndex].details.rayFieldHeight));
				 BEGIN OBJDET;
		}

 <OBJDET>"# Ray Directions"{ws}*":" {
			 fprintf(yyout, "raydirections per point source:");
			 BEGIN OBJDET_SRC_DIFFRAY_NRDIRECTIONS;
  }

		<OBJDET_SRC_DIFFRAY_NRDIRECTIONS>{num} {
				 fprintf(yyout, " %s", yytext);
				 if (!geomParseFlag_SrcDiffNr)
				 {
					geomParseFlag_SrcDiffNr=true;
					surfaceArray[surfIndex].details.nrRayDirections.x=atoi(yytext);
				 }
				 else
					surfaceArray[surfIndex].details.nrRayDirections.y=atoi(yytext);
				 BEGIN OBJDET_SRC_DIFFRAY_NRDIRECTIONS;
		}

		<OBJDET_SRC_DIFFRAY_NRDIRECTIONS>"\n" {
				 if (!geomParseFlag_SrcDiffNr)
				 {
					 fprintf(yyout, " definition of number of differential ray directions is missing!!");
					 surfaceArray[surfIndex].details.nrRayDirections.x=(long long)(1);
					 LexerErr=true;
				 }
				 BEGIN OBJDET;
		}

 <OBJDET>"Power(Watts)"{ws}*":" {
			 fprintf(yyout, "power [W]:");
			 BEGIN OBJDET_SRC_PWR;
  } 

		<OBJDET_SRC_PWR>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.power=atof(yytext);
				 geomParseFlag_SrcPower=true;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_PWR>"\n" {
				 fprintf(yyout, " no power definition found!!");
				 surfaceArray[surfIndex].details.power=0;
				 BEGIN OBJDET;
		}

 <OBJDET>"Direction Distribution"{ws}*":" {
			 fprintf(yyout, "dir dis:");
			 geomParseFlag_SrcDirDist=true;
			 BEGIN OBJDET_SRC_DIRDIS;
  } 

		<OBJDET_SRC_DIRDIS>"GRID_RECT" {
				 fprintf(yyout, " %s", yytext);
				 surfaceArray[surfIndex].details.rayDirDistr=RAYDIR_GRID_RECT;
				 BEGIN OBJDET_SRC_DIRDIS_GRIDRECT;
		}
				
			<OBJDET_SRC_DIRDIS_GRIDRECT>"X" {
					 fprintf(yyout, " xMax: ");
					 surfaceArray[surfIndex].details.rayDirDistr=RAYDIR_GRID_RECT_FARFIELD;
					 BEGIN OBJDET_SRC_DIRDIS_GRIDRECT_XMAX;
			}

				<OBJDET_SRC_DIRDIS_GRIDRECT_XMAX>{num} {
						 fprintf(yyout, "%s", yytext);
						 surfaceArray[surfIndex].details.alphaMax.x=atof(yytext)/360*2*M_PI;
						 BEGIN OBJDET_SRC_DIRDIS_GRIDRECT_XMIN;
				}

				<OBJDET_SRC_DIRDIS_GRIDRECT_XMIN>{num} {
						 fprintf(yyout, " %s", yytext);
						 surfaceArray[surfIndex].details.alphaMin.x=atof(yytext)/360*2*M_PI;
						 BEGIN OBJDET_SRC_DIRDIS_GRIDRECT;
				}

			<OBJDET_SRC_DIRDIS_GRIDRECT>"Y" {
					 fprintf(yyout, " Y:");
					 BEGIN OBJDET_SRC_DIRDIS_GRIDRECT_YMAX;
			}

				<OBJDET_SRC_DIRDIS_GRIDRECT_YMAX>{num} {
						 fprintf(yyout, "%s", yytext);
						 surfaceArray[surfIndex].details.alphaMax.y=atof(yytext)/360*2*M_PI;
						 BEGIN OBJDET_SRC_DIRDIS_GRIDRECT_YMIN;
				}

				<OBJDET_SRC_DIRDIS_GRIDRECT_YMIN>{num} {
						 fprintf(yyout, " %s", yytext);
						 surfaceArray[surfIndex].details.alphaMin.y=atof(yytext)/360*2*M_PI;
						 BEGIN OBJDET;
				}

		<OBJDET_SRC_DIRDIS_GRIDRECT>"\n" {
				 BEGIN OBJDET;
		}

		
		<OBJDET_SRC_DIRDIS>"GRID_RAD" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayDirDistr=RAYDIR_GRID_RAD;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_DIRDIS>"RAND_NORM" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayDirDistr=RAYDIR_RAND_NORM;
				 BEGIN OBJDET_SRC_DIRDIS_RAND;
		}

		<OBJDET_SRC_DIRDIS>"RAND" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayDirDistr=RAYDIR_RAND;
				 BEGIN OBJDET_SRC_DIRDIS_RAND;
		}

			<OBJDET_SRC_DIRDIS_RAND>"X" {
					 fprintf(yyout, " %s;", yytext);
					 BEGIN OBJDET_SRC_DIRDIS_RANDXMIN;
			}

				<OBJDET_SRC_DIRDIS_RANDXMIN>"\n" {
						 fprintf(yyout, " no minimum angle found!!");
						 surfaceArray[surfIndex].details.alphaMin.x=0;
						 geomParseFlag_SrcDirDist=false;
						 BEGIN OBJDET;
				}

				<OBJDET_SRC_DIRDIS_RANDXMIN>{num} {
						 fprintf(yyout, " %s;", yytext);
						 surfaceArray[surfIndex].details.alphaMin.x=atof(yytext)/360*2*PI;
						 BEGIN OBJDET_SRC_DIRDIS_RANDXMAX;
				}

				<OBJDET_SRC_DIRDIS_RANDXMAX>"\n" {
						 fprintf(yyout, " no maximum angle found!!");
						 surfaceArray[surfIndex].details.alphaMax.x=0;
						 geomParseFlag_SrcDirDist=false;
						 BEGIN OBJDET;
				}

				<OBJDET_SRC_DIRDIS_RANDXMAX>{num} {
						 fprintf(yyout, " %s;", yytext);
						 surfaceArray[surfIndex].details.alphaMax.x=atof(yytext)/360*2*PI;
						 BEGIN OBJDET_SRC_DIRDIS_RAND;
				}

			<OBJDET_SRC_DIRDIS_RAND>"Y" {
					 fprintf(yyout, " %s;", yytext);
					 BEGIN OBJDET_SRC_DIRDIS_RANDYMIN;
			}
			
				<OBJDET_SRC_DIRDIS_RANDYMIN>"\n" {
						 fprintf(yyout, " no minimum angle found!!");
						 surfaceArray[surfIndex].details.alphaMin.y=0;
						 geomParseFlag_SrcDirDist=false;
						 BEGIN OBJDET;
				}

				<OBJDET_SRC_DIRDIS_RANDYMIN>{num} {
						 fprintf(yyout, " %s;", yytext);
						 surfaceArray[surfIndex].details.alphaMin.y=atof(yytext)/360*2*PI;
						 BEGIN OBJDET_SRC_DIRDIS_RANDYMAX;
				}

				<OBJDET_SRC_DIRDIS_RANDYMAX>"\n" {
						 fprintf(yyout, " no minimum angle found!!");
						 surfaceArray[surfIndex].details.alphaMax.y=0;
						 geomParseFlag_SrcDirDist=false;
						 BEGIN OBJDET;
				}

				<OBJDET_SRC_DIRDIS_RANDYMAX>{num} {
						 fprintf(yyout, " %s;", yytext);
						 surfaceArray[surfIndex].details.alphaMax.y=atof(yytext)/360*2*PI;
						 BEGIN OBJDET;
				}

			<OBJDET_SRC_DIRDIS_RAND>"\n" {
					 fprintf(yyout, " full hemisphere;");
					 surfaceArray[surfIndex].details.alphaMax.y=PI/2;
					 surfaceArray[surfIndex].details.alphaMax.x=PI/2;
					 surfaceArray[surfIndex].details.alphaMin.y=-PI/2;
					 surfaceArray[surfIndex].details.alphaMin.x=-PI/2;
					 BEGIN OBJDET;
			}
				


		<OBJDET_SRC_DIRDIS>"TiltX" {
				 fprintf(yyout, " UNIFORM: TiltX ");
				 surfaceArray[surfIndex].details.rayDirDistr=RAYDIR_UNIFORM;
				 BEGIN OBJDET_SRC_DIRDIS_UNIFORMX;
		}

			<OBJDET_SRC_DIRDIS_UNIFORMX>{num} {
					 fprintf(yyout, "%s", yytext);
					 surfaceArray[surfIndex].details.rayDirectionTilt.x=atof(yytext)/360*2*PI;
					 BEGIN OBJDET_SRC_DIRDIS;
			}		

		<OBJDET_SRC_DIRDIS>"TiltY" {
				 fprintf(yyout, " TiltY ");
				 surfaceArray[surfIndex].details.rayDirDistr=RAYDIR_UNIFORM;
				 BEGIN OBJDET_SRC_DIRDIS_UNIFORMY;
		}

			<OBJDET_SRC_DIRDIS_UNIFORMY>{num} {
					 fprintf(yyout, "%s", yytext);
					 surfaceArray[surfIndex].details.rayDirectionTilt.y=atof(yytext)/360*2*PI;
					 BEGIN OBJDET;
			}		

		<OBJDET_SRC_DIRDIS>"\n" {
				 fprintf(yyout, " no ray direction distribution definition found!!");
				 surfaceArray[surfIndex].details.rayDirDistr=RAYDIR_UNKNOWN;
				 geomParseFlag_SrcDirDist=false;
				 BEGIN OBJDET;
		}
		
 <OBJDET>"Position Distribution"{ws}*":" {
			 fprintf(yyout, "pos dis:");
			 geomParseFlag_SrcPosDist=true;
			 BEGIN OBJDET_SRC_POSDIS;
  } 

		<OBJDET_SRC_POSDIS>"RAND_RECT_NORM" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayPosDistr=RAYPOS_RAND_RECT_NORM;
				 BEGIN OBJDET;
		}	

		<OBJDET_SRC_POSDIS>"RAND_RECT" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayPosDistr=RAYPOS_RAND_RECT;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_POSDIS>"RAND_RAD_NORM" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayPosDistr=RAYPOS_RAND_RAD_NORM;
				 BEGIN OBJDET;
		}		
		
		<OBJDET_SRC_POSDIS>"RAND_RAD" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayPosDistr=RAYPOS_RAND_RAD;
				 BEGIN OBJDET;
		}
		
		<OBJDET_SRC_POSDIS>"GRID_RECT" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayPosDistr=RAYPOS_GRID_RECT;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_POSDIS>"GRID_RAD" {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.rayPosDistr=RAYPOS_GRID_RAD;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_POSDIS>"\n" {
				 fprintf(yyout, " no ray position distribution definition found!!");
				 surfaceArray[surfIndex].details.rayPosDistr=RAYPOS_UNKNOWN;
				 geomParseFlag_SrcPosDist=false;
				 BEGIN OBJDET;
		}		

 <OBJDET>"Wavelength(um)"{ws}*":" {
			 fprintf(yyout, "wavelength:");
			 BEGIN OBJDET_SRC_LAMBDA;
  } 

		<OBJDET_SRC_LAMBDA>{num} {
				 fprintf(yyout, " %s ", yytext);
				 surfaceArray[surfIndex].details.lambda=atof(yytext);
				 geomParseFlag_SrcLambda=true;
				 BEGIN OBJDET;
		}		
		
		<OBJDET_SRC_LAMBDA>"\n" {
				 fprintf(yyout, " no wavelength definition found!!");
				 surfaceArray[surfIndex].details.lambda=0;
				 BEGIN OBJDET;
		}		  

 <OBJDET>"Coherence"{ws}*":" {
			 fprintf(yyout, "Coherence:");
			 BEGIN OBJDET_SRC_COHERENCE;
  } 

		<OBJDET_SRC_COHERENCE>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.srcCoherence=atof(yytext);
				 geomParseFlag_SrcCoherence=true;
				 BEGIN OBJDET;
		}

		<OBJDET_SRC_COHERENCE>"\n" {
				 fprintf(yyout, " no coherence definition found!!");
				 surfaceArray[surfIndex].details.srcCoherence=0;
				 BEGIN OBJDET;
		}
 /* ----------------------position------------------------------ */

 <OBJDET>"Reference object"{ws}*":" {
			 fprintf(yyout, "Ref object");
			 BEGIN OBJDET_REFOBJ;
  }

		<OBJDET_REFOBJ>{num} {
				 fprintf(yyout, " %s;", yytext);
				 refObjIndex=atoi(yytext)-1;
				 geomParseFlag_RefObj=true;
				 BEGIN OBJDET;
		}

 <OBJDET>"Tilt about XYZ"{ws}*":" {
			 fprintf(yyout, "Tilt:");
			 geomParseFlag_Tilt=true;
			 BEGIN OBJDET_TILTX;
 }
		
		<OBJDET_TILTX>{num} {
				 fprintf(yyout, " %s", yytext);
				 surfaceArray[surfIndex].details.tilt1.x=atof(yytext)/360*2*PI;
				 if (refObjIndex>=0)
					surfaceArray[surfIndex].details.tilt1.x=surfaceArray[surfIndex].details.tilt1.x+surfaceArray[refObjIndex].details.tilt1.x;
				 BEGIN OBJDET_TILTY;
		}

			<OBJDET_TILTX>"\n" {
					 fprintf(yyout, " warning: definition of tilt is incomplete!!", yytext);
					 geomParseFlag_Tilt=false;
					 BEGIN OBJDET;
			}

		<OBJDET_TILTY>{num} {
				 fprintf(yyout, " %s", yytext);
				 surfaceArray[surfIndex].details.tilt1.y=atof(yytext)/360*2*PI;
				 if (refObjIndex>=0)
					surfaceArray[surfIndex].details.tilt1.y=surfaceArray[surfIndex].details.tilt1.y+surfaceArray[refObjIndex].details.tilt1.y;
				 BEGIN OBJDET_TILTZ;
		}

			<OBJDET_TILTY>"\n" {
					 fprintf(yyout, " warning: definition of tilt is incomplete!!", yytext);
					 geomParseFlag_Tilt=false;
					 BEGIN OBJDET;
			}


		<OBJDET_TILTZ>{num} {
				 fprintf(yyout, " %s", yytext);
				 surfaceArray[surfIndex].details.tilt1.z=atof(yytext)/360*2*PI;
				 if (refObjIndex>=0)
					surfaceArray[surfIndex].details.tilt1.z=surfaceArray[surfIndex].details.tilt1.z+surfaceArray[refObjIndex].details.tilt1.z;
				 BEGIN OBJDET;
		}
		
			<OBJDET_TILTZ>"\n" {
					 fprintf(yyout, " warning: definition of tilt is incomplete!!", yytext);
					 geomParseFlag_Tilt=false;
					 BEGIN OBJDET;
			}
		

  <OBJDET>"Pos. Mtrx. R11 R12 R13 X"{ws}*":" {
			 fprintf(yyout, "x-pos:");
			 geomParseFlag_PosMtrxX=true;
			 BEGIN OBJDET_POSX_1;
  }

		<OBJDET_POSX_1>{num} {
				 BEGIN OBJDET_POSX_2;
		}

		<OBJDET_POSX_1>"\n" {
				 fprintf(yyout, " definition of position x is incomplete,");
				 geomParseFlag_PosMtrxX=false;
				 BEGIN OBJDET;
		}

			<OBJDET_POSX_2>{num} {
					 BEGIN OBJDET_POSX_3;
			}

			<OBJDET_POSX_2>"\n" {
					 fprintf(yyout, " definition of position x is incomplete,");
					 geomParseFlag_PosMtrxX=false;
					 BEGIN OBJDET;
			}

				<OBJDET_POSX_3>{num} {
						 BEGIN OBJDET_POSX;
				}
				
				<OBJDET_POSX_3>"\n" {
						 fprintf(yyout, " definition of position x is incomplete,");
						 geomParseFlag_PosMtrxX=false;
						 BEGIN OBJDET;
				}
				
					<OBJDET_POSX>{num} {
							 fprintf(yyout, " %s,", yytext);
							 surfaceArray[surfIndex].details.root.x=atof(yytext)*scale2mm;
							 geomParseFlag_PosMtrxX=true;
							 BEGIN OBJDET;
					}

					<OBJDET_POSX>"\n" {
							 geomParseFlag_PosMtrxX=false;
							 fprintf(yyout, " definition of position x is incomplete,");
							 BEGIN OBJDET;
					}
				
  <OBJDET>"Pos. Mtrx. R21 R22 R23 Y"{ws}*":" {
			 fprintf(yyout, "y-pos:");
			 BEGIN OBJDET_POSY_1;
  }

		<OBJDET_POSY_1>"\n" {
				 fprintf(yyout, " definition of position y is incomplete,");
				 BEGIN OBJDET;
		}

		<OBJDET_POSY_1>{num} {
				 BEGIN OBJDET_POSY_2;
		}

			<OBJDET_POSY_2>{num} {
					 BEGIN OBJDET_POSY_3;
			}

			<OBJDET_POSY_2>"\n" {
					 fprintf(yyout, " definition of position y is incomplete,");
					 BEGIN OBJDET;
			}

				<OBJDET_POSY_3>{num} {
						 BEGIN OBJDET_POSY;
				}
				
				<OBJDET_POSY_3>"\n" {
						 fprintf(yyout, " definition of position y is incomplete,");
						 BEGIN OBJDET;
				}
				
					<OBJDET_POSY>{num} {
							 fprintf(yyout, " %s,", yytext);
							 surfaceArray[surfIndex].details.root.y=atof(yytext)*scale2mm;
							 geomParseFlag_PosMtrxY=true;
							 BEGIN OBJDET;
					}
 
 					<OBJDET_POSY>"\n" {
							 fprintf(yyout, " definition of position y is incomplete,");
							 BEGIN OBJDET;
					}

			
  <OBJDET>"Pos. Mtrx. R31 R32 R33 Z"{ws}*":" {
			 fprintf(yyout, "z-pos:");
			 BEGIN OBJDET_POSZ_1;
  }

		<OBJDET_POSZ_1>"\n" {
				 fprintf(yyout, " definition of position z is incomplete,");
				 BEGIN OBJDET;
		}

		<OBJDET_POSZ_1>{num} {
				 BEGIN OBJDET_POSZ_2;
		}

			<OBJDET_POSZ_2>"\n" {
					 fprintf(yyout, " definition of position z is incomplete,");
					 BEGIN OBJDET;
			}

			<OBJDET_POSZ_2>{num} {
					 BEGIN OBJDET_POSZ_3;
			}

				<OBJDET_POSZ_3>"\n" {
						 fprintf(yyout, " definition of position z is incomplete,");
						 BEGIN OBJDET;
				}

				<OBJDET_POSZ_3>{num} {
						 BEGIN OBJDET_POSZ;
				}
								
					<OBJDET_POSZ>{num} {
							 fprintf(yyout, " %s;", yytext);
							 surfaceArray[surfIndex].details.root.z=atof(yytext)*scale2mm;
							 geomParseFlag_PosMtrxZ=true;
							 BEGIN OBJDET;
					}
					
					<OBJDET_POSZ>"\n" {
							 fprintf(yyout, " definition of position z is incomplete,");
							 BEGIN OBJDET;
					}
					

 /*-------------------------- material ----------------------------------------*/ 

  <OBJDET>"Material"{ws}*":" {
			 fprintf(yyout, "Material:");
			 geomParseFlag_Mat=true;
			 BEGIN OBJDET_MAT;
  }

		<OBJDET_MAT>"BK7" {
				 fprintf(yyout, " MT_BK7;");
				 surfaceArray[surfIndex].glass=MT_REFRMATERIAL;
				 sprintf(surfaceArray[surfIndex].glassName, "%s", yytext);
				 BEGIN OBJDET;
		}

		<OBJDET_MAT>"ABSORB" {
				 surfaceArray[surfIndex].glass=MT_ABSORB;
				 sprintf(surfaceArray[surfIndex].glassName, "%s", yytext);
				 fprintf(yyout, " MT_ABSORB;");
				 BEGIN OBJDET;
		}


		<OBJDET_MAT>"MIRROR" {
				 surfaceArray[surfIndex].glass=MT_MIRROR;
				 sprintf(surfaceArray[surfIndex].glassName, "%s", yytext);
				 fprintf(yyout, " MT_MIRROR;");
				 BEGIN OBJDET;
		}

		<OBJDET_MAT>"COVGLASS" {
				 surfaceArray[surfIndex].glass=MT_COVGLASS;
				 sprintf(surfaceArray[surfIndex].glassName, "%s", yytext);
				 fprintf(yyout, " MT_COVGLASS;");
				 BEGIN OBJDET;
		}

		<OBJDET_MAT>"DIFFRACT" {
				 surfaceArray[surfIndex].glass=MT_DIFFRACT;
				 sprintf(surfaceArray[surfIndex].glassName, "%s", yytext);
				 fprintf(yyout, " MT_DIFFRACT;");
				 BEGIN OBJDET;
		}	
		
		<OBJDET_MAT>"FILTER" {
				 surfaceArray[surfIndex].glass=MT_FILTER;
				 sprintf(surfaceArray[surfIndex].glassName, "%s", yytext);
				 fprintf(yyout, " MT_FILTER");
				 BEGIN OBJDET_MATFILTER;
		}	
		
			<OBJDET_MATFILTER>"MAX" {
					 fprintf(yyout, " MAX");
					 geomParseFlag_FilterMax=true;
					 BEGIN OBJDET_MATFILTER_MAX;
			}	
		
				<OBJDET_MATFILTER_MAX>{num} {
						 fprintf(yyout, "%s", yytext);
						 surfaceArray[surfIndex].details.filterMax=atof(yytext);
						 BEGIN OBJDET_MATFILTER;
				}			
				
				<OBJDET_MATFILTER_MAX>"\n" {
					 surfaceArray[surfIndex].details.filterMax=0;
					 fprintf(yyout, " nothing found;");					
					 geomParseFlag_FilterMax=false;
					 BEGIN OBJDET;
				}							

			<OBJDET_MATFILTER>"MIN" {
					 fprintf(yyout, " MIN");
					 geomParseFlag_FilterMin=true;
					 BEGIN OBJDET_MATFILTER_MIN;
			}	

				<OBJDET_MATFILTER_MIN>{num} {
						 fprintf(yyout, "%s", yytext);
						 surfaceArray[surfIndex].details.filterMin=atof(yytext);
						 BEGIN OBJDET_MATFILTER;
				}			

				<OBJDET_MATFILTER_MIN>"\n" {
					 surfaceArray[surfIndex].details.filterMin=0;
					 fprintf(yyout, " nothing found;");					
					 geomParseFlag_FilterMin=false;
					 BEGIN OBJDET;
				}							
				
			<OBJDET_MATFILTER>"\n" {
				 if (!geomParseFlag_FilterMin || !geomParseFlag_FilterMax)
				 {
					 surfaceArray[surfIndex].details.filterMax=0;
					 surfaceArray[surfIndex].details.filterMin=0;
				 }
				 BEGIN OBJDET;
			}							

		
		<OBJDET_MAT>{num} {
				 surfaceArray[surfIndex].glass=MT_REFRMATERIAL;
				 fprintf(yyout, " MT_REFRMATERIAL;");
				 sprintf(surfaceArray[surfIndex].glassName, "%s", "USERDEFINED");
				 surfaceArray[surfIndex].details.nRefr.x=atof(yytext);
				 BEGIN OBJDET;
		}
		
		<OBJDET_MAT>"n1"{ws}*= {
				 surfaceArray[surfIndex].glass=MT_REFRMATERIAL;
				 fprintf(yyout, " MT_REFRMATERIAL;");
				 sprintf(surfaceArray[surfIndex].glassName, "%s", "USERDEFINED");
				 BEGIN OBJDET_MAT_N1;
		}

			<OBJDET_MAT_N1>{num} {
					 fprintf(yyout, " n1=%s", yytext);
					 surfaceArray[surfIndex].details.nRefr.x=atof(yytext);
					 if (surfaceArray[surfIndex].details.nRefr.y==0)
						BEGIN OBJDET_MAT;
					else
						BEGIN OBJDET;
			}

		<OBJDET_MAT>"n2"{ws}*= {
				 fprintf(yyout, " Immersion;");
				 sprintf(surfaceArray[surfIndex].details.immersionName, "%s", "USERDEFINED");
				 BEGIN OBJDET_MAT_N2;
		}

			<OBJDET_MAT_N2>{num} {
					 fprintf(yyout, " n2=%s", yytext);
					 surfaceArray[surfIndex].details.nRefr.y=atof(yytext);
					 if (surfaceArray[surfIndex].details.nRefr.x==0)
						BEGIN OBJDET_MAT;
					else
						BEGIN OBJDET;
			}

		<OBJDET_MAT>{name} {
				 surfaceArray[surfIndex].glass=MT_REFRMATERIAL;
				 fprintf(yyout, " MT_REFRMATERIAL;");
				 sprintf(surfaceArray[surfIndex].glassName, "%s", yytext);
				 BEGIN OBJDET;
		}

		<OBJDET_MAT>"\n" {
				 surfaceArray[surfIndex].glass=MT_UNKNOWNMATERIAL;
				 fprintf(yyout, " MT_UNKNOWNMATERIAL;");
				 
				 geomParseFlag_Mat=false;
				 BEGIN OBJDET;
		}
		
  <OBJDET>"Immerse"{ws}*":"{ws}* {
			 fprintf(yyout, "Immersion:");
			 geomParseFlag_Immerse=true;
			 BEGIN OBJDET_IMM;
  }

		<OBJDET_IMM>{name} {
				 fprintf(yyout, yytext);
				 sprintf(surfaceArray[surfIndex].details.immersionName, "%s", yytext);
				 BEGIN OBJDET;
		}
		
		<OBJDET_IMM>{num} {
				 surfaceArray[surfIndex].glass=MT_REFRMATERIAL;
				 fprintf(yyout, " Immersion;");
				 sprintf(surfaceArray[surfIndex].details.immersionName, "%s", "USERDEFINED");
				 surfaceArray[surfIndex].details.nRefr.y=atof(yytext);
				 BEGIN OBJDET;
		}		

		<OBJDET_IMM>"\n" {
				 fprintf(yyout, "warning: definition of immersion is incomplete!!!");
				 geomParseFlag_Immerse=false;
				 BEGIN OBJDET;
		}

 /*---------------------------------------- scattering ------------------------------*/
   <OBJDET>"Face"{ws}*{num}{ws}*":"{ws}* {
			 fprintf(yyout, "face: ");
			 geomParseFlag_Face=true;
			 BEGIN OBJDET_FACE;
  }
  
	  <OBJDET_FACE>"Side Faces" {
				fprintf(yyout, "side ");
				faceIndex=0;
				BEGIN OBJDET;
	  }

	  <OBJDET_FACE>"Front Face" {
				fprintf(yyout, "front ");
				faceIndex=1;
				BEGIN OBJDET;
	  }
	  
	  <OBJDET_FACE>"Back Face" {
				fprintf(yyout, "back ");
				faceIndex=2;
				BEGIN OBJDET;
	  }
	  
	  <OBJDET_FACE>"All Faces" {
				fprintf(yyout, "all ");
				faceIndex=3;
				BEGIN OBJDET;
	  }

	  <OBJDET_FACE>"\n" {
				fprintf(yyout, "error! no valid face specifier found. Front face assumed");
				faceIndex=1;
				BEGIN OBJDET;
	  }

  
   <OBJDET>"Scattering"{ws}*":"{ws}* {
			 fprintf(yyout, "Scattering:");
			 geomParseFlag_Scatter=true;
			 BEGIN OBJDET_SCAT;
  }

		<OBJDET_SCAT>"None" {
				 switch (faceIndex)
				 {
					case 0:
						fprintf(yyout, "face0 ST_NOSCATTER;");
						break;
					case 1:
						fprintf(yyout, "face1 ST_NOSCATTER;");
						break;
					case 2:
						fprintf(yyout, "face2 ST_NOSCATTER;");
						break;
				 }						
				 BEGIN OBJDET;
		}
 
		<OBJDET_SCAT>"TorranceSparrow1D" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.scatterSide=ST_TORRSPARR1D;
						fprintf(yyout, "face0 ST_TORRSPARR1D;");
						break;
					case 1:
						surfaceArray[surfIndex].details.scatterFront=ST_TORRSPARR1D;
						fprintf(yyout, "face1 ST_TORRSPARR1D;");
						break;
					case 2:
						surfaceArray[surfIndex].details.scatterBack=ST_TORRSPARR1D;
						fprintf(yyout, "face2 ST_TORRSPARR1D;");
						break;
					case 3:
						surfaceArray[surfIndex].details.scatterBack=ST_TORRSPARR1D;
						surfaceArray[surfIndex].details.scatterFront=ST_TORRSPARR1D;
						surfaceArray[surfIndex].details.scatterSide=ST_TORRSPARR1D;
						fprintf(yyout, "all faces ST_TORRSPARR1D;");
						break;
				 }						
				 BEGIN OBJDET_SCAT_TORSPARR1D;
		}
				 
			<OBJDET_SCAT_TORSPARR1D>"Kdl"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR1D_KDL;
			}
							
				<OBJDET_SCAT_TORSPARR1D_KDL>{num} {
						 surfaceArray[surfIndex].details.matParams[0]=atof(yytext);
						 fprintf(yyout, " Kdl= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR1D;		
				}
			
			<OBJDET_SCAT_TORSPARR1D>"Ksl"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR1D_KSL;
			}
			
				<OBJDET_SCAT_TORSPARR1D_KSL>{num} {
						 surfaceArray[surfIndex].details.matParams[1]=atof(yytext);
						 fprintf(yyout, " Ksl= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR1D;		
				}
			
			<OBJDET_SCAT_TORSPARR1D>"Ksp"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR1D_KSP;
			}
			
				<OBJDET_SCAT_TORSPARR1D_KSP>{num} {
						 surfaceArray[surfIndex].details.matParams[2]=atof(yytext);
						 fprintf(yyout, " Ksp= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR1D;		
				}
			
			<OBJDET_SCAT_TORSPARR1D>"sigmaXsl"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR1D_SIGMASL;
			}
			
				<OBJDET_SCAT_TORSPARR1D_SIGMASL>{num} {
						 surfaceArray[surfIndex].details.matParams[3]=atof(yytext);
						 fprintf(yyout, " sigmaXsl= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR1D;		
				}
		
			<OBJDET_SCAT_TORSPARR1D>"sigmaXsp"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR1D_SIGMASP;
			}
		
				<OBJDET_SCAT_TORSPARR1D_SIGMASP>{num} {
						 surfaceArray[surfIndex].details.matParams[4]=atof(yytext);
						 fprintf(yyout, " sigmaXsp= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR1D;		
				}
							 
			<OBJDET_SCAT_TORSPARR1D>"scatAxis"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR1D_SCATAXISX;
			}
		
				<OBJDET_SCAT_TORSPARR1D_SCATAXISX>{num} {
						 surfaceArray[surfIndex].details.matParams[5]=atof(yytext);
						 fprintf(yyout, " scatX= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR1D_SCATAXISY;		
				}
								 
				<OBJDET_SCAT_TORSPARR1D_SCATAXISY>{num} {
						 surfaceArray[surfIndex].details.matParams[6]=atof(yytext);
						 fprintf(yyout, " scatY= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR1D_SCATAXISZ;		
				}
				
				<OBJDET_SCAT_TORSPARR1D_SCATAXISZ>{num} {
						 surfaceArray[surfIndex].details.matParams[7]=atof(yytext);
						 fprintf(yyout, " scatZ= %s; ", yytext);
						 BEGIN OBJDET;		
				}	

		<OBJDET_SCAT>"TorranceSparrow2D" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.scatterSide=ST_TORRSPARR2D;
						fprintf(yyout, "face0 ST_TORRSPARR2D;");
						break;
					case 1:
						surfaceArray[surfIndex].details.scatterFront=ST_TORRSPARR2D;
						fprintf(yyout, "face1 ST_TORRSPARR2D;");
						break;
					case 2:
						surfaceArray[surfIndex].details.scatterBack=ST_TORRSPARR2D;
						fprintf(yyout, "face2 ST_TORRSPARR2D;");
						break;
					case 3:
						surfaceArray[surfIndex].details.scatterBack=ST_TORRSPARR2D;
						surfaceArray[surfIndex].details.scatterFront=ST_TORRSPARR2D;
						surfaceArray[surfIndex].details.scatterSide=ST_TORRSPARR2D;
						fprintf(yyout, "all faces ST_TORRSPARR2D;");
						break;
				 }						
				 BEGIN OBJDET_SCAT_TORSPARR2D;
		}
				 
			<OBJDET_SCAT_TORSPARR2D>"Kdl"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR2D_KDL;
			}
							
				<OBJDET_SCAT_TORSPARR2D_KDL>{num} {
						 surfaceArray[surfIndex].details.matParams[0]=atof(yytext);
						 fprintf(yyout, " Kdl= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR2D;		
				}
			
			<OBJDET_SCAT_TORSPARR2D>"Ksl"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR2D_KSL;
			}
			
				<OBJDET_SCAT_TORSPARR2D_KSL>{num} {
						 surfaceArray[surfIndex].details.matParams[1]=atof(yytext);
						 fprintf(yyout, " Ksl= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR2D;		
				}
			
			<OBJDET_SCAT_TORSPARR2D>"Ksp"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR2D_KSP;
			}
			
				<OBJDET_SCAT_TORSPARR2D_KSP>{num} {
						 surfaceArray[surfIndex].details.matParams[2]=atof(yytext);
						 fprintf(yyout, " Ksp= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR2D;		
				}
			
			<OBJDET_SCAT_TORSPARR2D>"sigmaXsl"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR2D_SIGMASL;
			}
			
				<OBJDET_SCAT_TORSPARR2D_SIGMASL>{num} {
						 surfaceArray[surfIndex].details.matParams[3]=atof(yytext);
						 fprintf(yyout, " sigmaXsl= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR2D;		
				}
		
			<OBJDET_SCAT_TORSPARR2D>"sigmaXsp"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR2D_SIGMASP;
			}
		
				<OBJDET_SCAT_TORSPARR2D_SIGMASP>{num} {
						 surfaceArray[surfIndex].details.matParams[4]=atof(yytext);
						 fprintf(yyout, " sigmaXsp= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR2D;		
				}
							 
			<OBJDET_SCAT_TORSPARR2D>"scatAxis"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_TORSPARR2D_SCATAXISX;
			}
		
				<OBJDET_SCAT_TORSPARR2D_SCATAXISX>{num} {
						 surfaceArray[surfIndex].details.matParams[5]=atof(yytext);
						 fprintf(yyout, " scatX= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR2D_SCATAXISY;		
				}
								 
				<OBJDET_SCAT_TORSPARR2D_SCATAXISY>{num} {
						 surfaceArray[surfIndex].details.matParams[6]=atof(yytext);
						 fprintf(yyout, " scatY= %s; ", yytext);
						 BEGIN OBJDET_SCAT_TORSPARR2D_SCATAXISZ;		
				}
				
				<OBJDET_SCAT_TORSPARR2D_SCATAXISZ>{num} {
						 surfaceArray[surfIndex].details.matParams[7]=atof(yytext);
						 fprintf(yyout, " scatZ= %s; ", yytext);
						 BEGIN OBJDET;		
				}	

		<OBJDET_SCAT>"DoubleCauchy1D" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.scatterSide=ST_DOUBLECAUCHY1D;
						fprintf(yyout, "face0 DoubleCauchy1D; ");				
						break;
					case 1:
						surfaceArray[surfIndex].details.scatterFront=ST_DOUBLECAUCHY1D;
						fprintf(yyout, "face1 DoubleCauchy1D; ");				
						break;
					case 2:
						surfaceArray[surfIndex].details.scatterBack=ST_DOUBLECAUCHY1D;
						fprintf(yyout, "face2 DoubleCauchy1D; ");				
						break;
					case 3:
						surfaceArray[surfIndex].details.scatterBack=ST_DOUBLECAUCHY1D;
						surfaceArray[surfIndex].details.scatterFront=ST_DOUBLECAUCHY1D;
						surfaceArray[surfIndex].details.scatterSide=ST_DOUBLECAUCHY1D;
						fprintf(yyout, "all faces DoubleCauchy1D; ");				
						break;
				 }		
				 
				 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;
		}
				 
			<OBJDET_SCAT_DOUBLECAUCHY1D>"Ksl"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_KSL;
			}
							
				<OBJDET_SCAT_DOUBLECAUCHY1D_KSL>{num} {
						 surfaceArray[surfIndex].details.matParams[0]=atof(yytext);
						 fprintf(yyout, " Ksl= %s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}
				
			<OBJDET_SCAT_DOUBLECAUCHY1D>"Ksp"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_KSP;
			}
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_KSP>{num} {
						 surfaceArray[surfIndex].details.matParams[1]=atof(yytext);
						 fprintf(yyout, " Ksp= %s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}
			
			<OBJDET_SCAT_DOUBLECAUCHY1D>"gammaXsl"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_GAMMASL;
			}
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_GAMMASL>{num} {
						 surfaceArray[surfIndex].details.matParams[2]=atof(yytext);
						 fprintf(yyout, " gammaXsl= %s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}
		
			<OBJDET_SCAT_DOUBLECAUCHY1D>"gammaXsp"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_GAMMASP;
			}
		
				<OBJDET_SCAT_DOUBLECAUCHY1D_GAMMASP>{num} {
						 surfaceArray[surfIndex].details.matParams[3]=atof(yytext);
						 fprintf(yyout, " gammaXsp= %s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}
							 
			<OBJDET_SCAT_DOUBLECAUCHY1D>"scatAxis"[\t ]*"=" {
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISX;
			}
		
				<OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISX>{num} {
						 surfaceArray[surfIndex].details.matParams[4]=atof(yytext);
						 fprintf(yyout, " scatX= %s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISY;		
				}
								 
				<OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISY>{num} {
						 surfaceArray[surfIndex].details.matParams[5]=atof(yytext);
						 fprintf(yyout, " scatY= %s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISZ;		
				}
				
				<OBJDET_SCAT_DOUBLECAUCHY1D_SCATAXISZ>{num} {
						 surfaceArray[surfIndex].details.matParams[6]=atof(yytext);
						 fprintf(yyout, " scatZ= %s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	
				
			<OBJDET_SCAT_DOUBLECAUCHY1D>"a_ysl" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.scatterSide=ST_DISPDOUBLECAUCHY1D;
						fprintf(yyout, "face0 DispDoubleCauchy1D; ");				
						break;
					case 1:
						surfaceArray[surfIndex].details.scatterFront=ST_DISPDOUBLECAUCHY1D;
						fprintf(yyout, "face1 DispDoubleCauchy1D; ");				
						break;
					case 2:
						surfaceArray[surfIndex].details.scatterBack=ST_DISPDOUBLECAUCHY1D;
						fprintf(yyout, "face2 DispDoubleCauchy1D; ");				
						break;
					case 3:
						surfaceArray[surfIndex].details.scatterBack=ST_DISPDOUBLECAUCHY1D;
						surfaceArray[surfIndex].details.scatterFront=ST_DISPDOUBLECAUCHY1D;
						surfaceArray[surfIndex].details.scatterSide=ST_DISPDOUBLECAUCHY1D;
						fprintf(yyout, "all faces DispDoubleCauchy1D; ");				
						break;
				 }		
			
					 fprintf(yyout, " a_ysl= ");
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_AYSL;		
			}	
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_AYSL>{num} {
						 surfaceArray[surfIndex].details.matParams[7]=atof(yytext);
						 fprintf(yyout, "%s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	

			<OBJDET_SCAT_DOUBLECAUCHY1D>"c_ysl" {
					 fprintf(yyout, " c_ysl= ");
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_CYSL;		
			}	
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_CYSL>{num} {
						 surfaceArray[surfIndex].details.matParams[8]=atof(yytext);
						 fprintf(yyout, "%s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	

			<OBJDET_SCAT_DOUBLECAUCHY1D>"a_ysp" {
					 fprintf(yyout, " a_ysp= ");
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_AYSP;		
			}	
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_AYSP>{num} {
						 surfaceArray[surfIndex].details.matParams[9]=atof(yytext);
						 fprintf(yyout, "%s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	

			<OBJDET_SCAT_DOUBLECAUCHY1D>"c_ysp" {
					 fprintf(yyout, " a_sp= ");
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_CYSP;		
			}	
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_CYSP>{num} {
						 surfaceArray[surfIndex].details.matParams[10]=atof(yytext);
						 fprintf(yyout, "%s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	

			<OBJDET_SCAT_DOUBLECAUCHY1D>"a_ksp" {
					 fprintf(yyout, " a_ksp= ");
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_AKSP;		
			}	
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_AKSP>{num} {
						 surfaceArray[surfIndex].details.matParams[11]=atof(yytext);
						 fprintf(yyout, "%s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	

			<OBJDET_SCAT_DOUBLECAUCHY1D>"c_ksp" {
					 fprintf(yyout, " c_ksp= ");
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_CKSP;		
			}	
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_CKSP>{num} {
						 surfaceArray[surfIndex].details.matParams[12]=atof(yytext);
						 fprintf(yyout, "%s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	

			<OBJDET_SCAT_DOUBLECAUCHY1D>"a_ksl" {
					 fprintf(yyout, " a_ksl= ");
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_AKSL;		
			}	
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_AKSL>{num} {
						 surfaceArray[surfIndex].details.matParams[13]=atof(yytext);
						 fprintf(yyout, "%s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	

			<OBJDET_SCAT_DOUBLECAUCHY1D>"c_ksl" {
					 fprintf(yyout, " c_ksl= ");
					 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D_CKSL;		
			}	
			
				<OBJDET_SCAT_DOUBLECAUCHY1D_CKSL>{num} {
						 surfaceArray[surfIndex].details.matParams[14]=atof(yytext);
						 fprintf(yyout, "%s; ", yytext);
						 BEGIN OBJDET_SCAT_DOUBLECAUCHY1D;		
				}	

			<OBJDET_SCAT_DOUBLECAUCHY1D>"\n" {
					 fprintf(yyout, " a_sp= ");
					 BEGIN OBJDET;		
			}					

				
		<OBJDET_SCAT>"Lambert2D" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.scatterSide=ST_LAMBERT2D;
						fprintf(yyout, "face0 ST_LAMBERT2D;");
						break;
					case 1:
						surfaceArray[surfIndex].details.scatterFront=ST_LAMBERT2D;
						fprintf(yyout, "face1 ST_LAMBERT2D;");
						break;
					case 2:
						surfaceArray[surfIndex].details.scatterBack=ST_LAMBERT2D;
						fprintf(yyout, "face2 ST_LAMBERT2D;");
						break;
					case 3:
						surfaceArray[surfIndex].details.scatterBack=ST_LAMBERT2D;
						surfaceArray[surfIndex].details.scatterFront=ST_LAMBERT2D;
						surfaceArray[surfIndex].details.scatterSide=ST_LAMBERT2D;
						fprintf(yyout, "all faces ST_LAMBERT2D;");
						break;
				 }						
				 BEGIN OBJDET_SCAT_LAMBERT2D_TIR;
		}
		
				<OBJDET_SCAT_LAMBERT2D_TIR>{num} {
						 surfaceArray[surfIndex].details.matParams[0]=atof(yytext);
						 fprintf(yyout, " TIR= %s; ", yytext);
						 BEGIN OBJDET;		
				}	

			<OBJDET_SCAT_LAMBERT2D_TIR>"\n" {
					 surfaceArray[surfIndex].details.scatterFront=ST_UNKNOWNSCATTER;
					 fprintf(yyout, " face%i ST_UNKNOWNSCATTER;", faceIndex);
					 geomParseFlag_Scatter=false;
					 BEGIN OBJDET;
			}
		
				

		<OBJDET_SCAT>"\n" {
				 surfaceArray[surfIndex].details.scatterFront=ST_UNKNOWNSCATTER;
				 fprintf(yyout, " face%i ST_UNKNOWNSCATTER;", faceIndex);
				 geomParseFlag_Scatter=false;
				 BEGIN OBJDET;
		}

   <OBJDET>"Importance Area"{ws}*":"{ws}* {
			 fprintf(yyout, "Importance Area:");
			 surfaceArray[surfIndex].details.importanceArea=true;
			 BEGIN OBJDET_IMPAREA;
  }
  
			<OBJDET_IMPAREA>"Object" {
					 fprintf(yyout, " towards object;");
					 BEGIN OBJDET_IMPAREA_OBJECT;
			}

				<OBJDET_IMPAREA_OBJECT>{int} {
						 surfaceArray[surfIndex].details.importanceObjNr=atoi(yytext)-1-srcIndex;
						 fprintf(yyout, " %s", yytext);
						 BEGIN OBJDET;
				}

			<OBJDET_IMPAREA>"alphaXmin" {
					 fprintf(yyout, " alphaXmin;");
					 BEGIN OBJDET_IMPAREA_ALPHAXMIN;
			}

				<OBJDET_IMPAREA_ALPHAXMIN>{num} {
						 fprintf(yyout, " %s", yytext);
						 surfaceArray[surfIndex].details.importanceConeAlphaMin.x=atof(yytext)/360*2*PI;						 
						 geomParseFlag_ImpAreaXmin=true;
						 BEGIN OBJDET_IMPAREA;
				}

			<OBJDET_IMPAREA>"alphaXmax" {
					 fprintf(yyout, " alphaXmax;");
					 BEGIN OBJDET_IMPAREA_ALPHAXMAX;
			}

				<OBJDET_IMPAREA_ALPHAXMAX>{num} {
						 fprintf(yyout, " %s", yytext);
						 surfaceArray[surfIndex].details.importanceConeAlphaMax.x=atof(yytext)/360*2*PI;						 
						 geomParseFlag_ImpAreaXmax=true;
						 BEGIN OBJDET_IMPAREA;
				}

			<OBJDET_IMPAREA>"alphaYmin" {
					 fprintf(yyout, " alphaYmin;");
					 BEGIN OBJDET_IMPAREA_ALPHAYMIN;
			}

				<OBJDET_IMPAREA_ALPHAYMIN>{num} {
						 fprintf(yyout, " %s", yytext);
						 surfaceArray[surfIndex].details.importanceConeAlphaMin.y=atof(yytext)/360*2*PI;						 
						 geomParseFlag_ImpAreaYmin=true;
						 BEGIN OBJDET_IMPAREA;
				}

			<OBJDET_IMPAREA>"alphaYmax" {
					 fprintf(yyout, " alphaYmax;");
					 BEGIN OBJDET_IMPAREA_ALPHAYMAX;
			}

				<OBJDET_IMPAREA_ALPHAYMAX>{num} {
						 fprintf(yyout, " %s", yytext);
						 surfaceArray[surfIndex].details.importanceConeAlphaMax.y=atof(yytext)/360*2*PI;						 
						 geomParseFlag_ImpAreaYmax=true;
						 BEGIN OBJDET_IMPAREA;
				}  
		
			<OBJDET_IMPAREA>"\n" {
					 if (! (geomParseFlag_ImpAreaYmax&&geomParseFlag_ImpAreaYmin&&geomParseFlag_ImpAreaXmax&&geomParseFlag_ImpAreaXmin))
					 {
						LexerErr=true;
						std::cout << "error: importance area is not fully defined for object " << geomIndex << std::endl;
						fprintf(yyout, " angles for importance area were not fully defined!!");
					 }
					 BEGIN OBJDET;
			}
		
		
 /*---------------------------------------- coating ------------------------------*/
   <OBJDET>"Coating"{ws}*":"{ws}* {
			 fprintf(yyout, "Coating!");
			 geomParseFlag_Coating=true;
			 BEGIN OBJDET_COAT;
  }

  		<OBJDET_COAT>"t"{ws}*"=" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.coatingSide=CT_NUMCOEFFS;
						break;
					case 1:
						surfaceArray[surfIndex].details.coatingFront=CT_NUMCOEFFS;
						break;
					case 2:
						surfaceArray[surfIndex].details.coatingBack=CT_NUMCOEFFS;
						break;
					case 3:
						surfaceArray[surfIndex].details.coatingBack=CT_NUMCOEFFS;
						surfaceArray[surfIndex].details.coatingFront=CT_NUMCOEFFS;
						surfaceArray[surfIndex].details.coatingSide=CT_NUMCOEFFS;
					break;
				 }						
				 BEGIN OBJDET_COAT_COEFFT;
		}

  			<OBJDET_COAT_COEFFT>{num} {
					 switch (faceIndex)
					 {
						case 0:
							surfaceArray[surfIndex].details.coating_side_t=atof(yytext);
							break;
						case 1:
							surfaceArray[surfIndex].details.coating_front_t=atof(yytext);
							break;
						case 2:
							surfaceArray[surfIndex].details.coating_back_t=atof(yytext);
							break;
						case 3:
							surfaceArray[surfIndex].details.coating_back_t=atof(yytext);
							surfaceArray[surfIndex].details.coating_front_t=atof(yytext);
							surfaceArray[surfIndex].details.coating_side_t=atof(yytext);
						break;
					 }						
					 BEGIN OBJDET_COAT;
			}		
 
  		<OBJDET_COAT>"r"{ws}*"=" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.coatingSide=CT_NUMCOEFFS;
						break;
					case 1:
						surfaceArray[surfIndex].details.coatingFront=CT_NUMCOEFFS;
						break;
					case 2:
						surfaceArray[surfIndex].details.coatingBack=CT_NUMCOEFFS;
						break;
					case 3:
						surfaceArray[surfIndex].details.coatingBack=CT_NUMCOEFFS;
						surfaceArray[surfIndex].details.coatingFront=CT_NUMCOEFFS;
						surfaceArray[surfIndex].details.coatingSide=CT_NUMCOEFFS;
					break;
				 }						
				 BEGIN OBJDET_COAT_COEFFR;
		}

  			<OBJDET_COAT_COEFFR>{num} {
					 switch (faceIndex)
					 {
						case 0:
							surfaceArray[surfIndex].details.coating_side_r=atof(yytext);
							break;
						case 1:
							surfaceArray[surfIndex].details.coating_front_r=atof(yytext);
							break;
						case 2:
							surfaceArray[surfIndex].details.coating_back_r=atof(yytext);
							break;
						case 3:
							surfaceArray[surfIndex].details.coating_back_r=atof(yytext);
							surfaceArray[surfIndex].details.coating_front_r=atof(yytext);
							surfaceArray[surfIndex].details.coating_side_r=atof(yytext);
						break;
					 }						
					 BEGIN OBJDET_COAT;
			}		
  
  		<OBJDET_COAT>"a_r"{ws}*"=" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.coatingSide=CT_DISPNUMCOEFFS;
						break;
					case 1:
						surfaceArray[surfIndex].details.coatingFront=CT_DISPNUMCOEFFS;
						break;
					case 2:
						surfaceArray[surfIndex].details.coatingBack=CT_DISPNUMCOEFFS;
						break;
					case 3:
						surfaceArray[surfIndex].details.coatingBack=CT_DISPNUMCOEFFS;
						surfaceArray[surfIndex].details.coatingFront=CT_DISPNUMCOEFFS;
						surfaceArray[surfIndex].details.coatingSide=CT_DISPNUMCOEFFS;
					break;
				 }						
				 BEGIN OBJDET_COAT_COEFFAR;
		}

  			<OBJDET_COAT_COEFFAR>{num} {
					 switch (faceIndex)
					 {
						case 0:
							surfaceArray[surfIndex].details.coating_side_a_r=atof(yytext);
							break;
						case 1:
							surfaceArray[surfIndex].details.coating_front_a_r=atof(yytext);
							break;
						case 2:
							surfaceArray[surfIndex].details.coating_back_a_r=atof(yytext);
							break;
						case 3:
							surfaceArray[surfIndex].details.coating_back_a_r=atof(yytext);
							surfaceArray[surfIndex].details.coating_front_a_r=atof(yytext);
							surfaceArray[surfIndex].details.coating_side_a_r=atof(yytext);
						break;
					 }						
					 BEGIN OBJDET_COAT;
			}	
			
  		<OBJDET_COAT>"a_t"{ws}*"=" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.coatingSide=CT_DISPNUMCOEFFS;
						break;
					case 1:
						surfaceArray[surfIndex].details.coatingFront=CT_DISPNUMCOEFFS;
						break;
					case 2:
						surfaceArray[surfIndex].details.coatingBack=CT_DISPNUMCOEFFS;
						break;
					case 3:
						surfaceArray[surfIndex].details.coatingBack=CT_DISPNUMCOEFFS;
						surfaceArray[surfIndex].details.coatingFront=CT_DISPNUMCOEFFS;
						surfaceArray[surfIndex].details.coatingSide=CT_DISPNUMCOEFFS;
					break;
				 }						
				 BEGIN OBJDET_COAT_COEFFAT;
		}

  			<OBJDET_COAT_COEFFAT>{num} {
					 switch (faceIndex)
					 {
						case 0:
							surfaceArray[surfIndex].details.coating_side_a_t=atof(yytext);
							break;
						case 1:
							surfaceArray[surfIndex].details.coating_front_a_t=atof(yytext);
							break;
						case 2:
							surfaceArray[surfIndex].details.coating_back_a_t=atof(yytext);
							break;
						case 3:
							surfaceArray[surfIndex].details.coating_back_a_t=atof(yytext);
							surfaceArray[surfIndex].details.coating_front_a_t=atof(yytext);
							surfaceArray[surfIndex].details.coating_side_a_t=atof(yytext);
						break;
					 }						
					 BEGIN OBJDET_COAT;
			}				
			
  		<OBJDET_COAT>"c_r"{ws}*"=" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.coatingSide=CT_DISPNUMCOEFFS;
						break;
					case 1:
						surfaceArray[surfIndex].details.coatingFront=CT_DISPNUMCOEFFS;
						break;
					case 2:
						surfaceArray[surfIndex].details.coatingBack=CT_DISPNUMCOEFFS;
						break;
					case 3:
						surfaceArray[surfIndex].details.coatingBack=CT_DISPNUMCOEFFS;
						surfaceArray[surfIndex].details.coatingFront=CT_DISPNUMCOEFFS;
						surfaceArray[surfIndex].details.coatingSide=CT_DISPNUMCOEFFS;
					break;
				 }						
				 BEGIN OBJDET_COAT_COEFFCR;
		}

  			<OBJDET_COAT_COEFFCR>{num} {
					 switch (faceIndex)
					 {
						case 0:
							surfaceArray[surfIndex].details.coating_side_c_r=atof(yytext);
							break;
						case 1:
							surfaceArray[surfIndex].details.coating_front_c_r=atof(yytext);
							break;
						case 2:
							surfaceArray[surfIndex].details.coating_back_c_r=atof(yytext);
							break;
						case 3:
							surfaceArray[surfIndex].details.coating_back_c_r=atof(yytext);
							surfaceArray[surfIndex].details.coating_front_c_r=atof(yytext);
							surfaceArray[surfIndex].details.coating_side_c_r=atof(yytext);
						break;
					 }						
					 BEGIN OBJDET_COAT;
			}	
			
  		<OBJDET_COAT>"c_t"{ws}*"=" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.coatingSide=CT_DISPNUMCOEFFS;
						break;
					case 1:
						surfaceArray[surfIndex].details.coatingFront=CT_DISPNUMCOEFFS;
						break;
					case 2:
						surfaceArray[surfIndex].details.coatingBack=CT_DISPNUMCOEFFS;
						break;
					case 3:
						surfaceArray[surfIndex].details.coatingBack=CT_DISPNUMCOEFFS;
						surfaceArray[surfIndex].details.coatingFront=CT_DISPNUMCOEFFS;
						surfaceArray[surfIndex].details.coatingSide=CT_DISPNUMCOEFFS;
					break;
				 }						
				 BEGIN OBJDET_COAT_COEFFCT;
		}

  			<OBJDET_COAT_COEFFCT>{num} {
					 switch (faceIndex)
					 {
						case 0:
							surfaceArray[surfIndex].details.coating_side_c_t=atof(yytext);
							break;
						case 1:
							surfaceArray[surfIndex].details.coating_front_c_t=atof(yytext);
							break;
						case 2:
							surfaceArray[surfIndex].details.coating_back_c_t=atof(yytext);
							break;
						case 3:
							surfaceArray[surfIndex].details.coating_back_c_t=atof(yytext);
							surfaceArray[surfIndex].details.coating_front_c_t=atof(yytext);
							surfaceArray[surfIndex].details.coating_side_c_t=atof(yytext);
						break;
					 }						
					 BEGIN OBJDET_COAT;
			}					
							
  
  		<OBJDET_COAT>"Fresnel" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.coatingSide=CT_FRESNELCOEFFS;
						break;
					case 1:
						surfaceArray[surfIndex].details.coatingFront=CT_FRESNELCOEFFS;
						break;
					case 2:
						surfaceArray[surfIndex].details.coatingBack=CT_FRESNELCOEFFS;
						break;
					case 3:
						surfaceArray[surfIndex].details.coatingBack=CT_FRESNELCOEFFS;
						surfaceArray[surfIndex].details.coatingFront=CT_FRESNELCOEFFS;
						surfaceArray[surfIndex].details.coatingSide=CT_FRESNELCOEFFS;
					break;
				 }						
				 BEGIN OBJDET;
		}

  		<OBJDET_COAT>"(none)" {
				 switch (faceIndex)
				 {
					case 0:
						surfaceArray[surfIndex].details.coatingSide=CT_NOCOATING;
						break;
					case 1:
						surfaceArray[surfIndex].details.coatingFront=CT_NOCOATING;
						break;
					case 2:
						surfaceArray[surfIndex].details.coatingBack=CT_NOCOATING;
						break;
					case 3:
						surfaceArray[surfIndex].details.coatingBack=CT_NOCOATING;
						surfaceArray[surfIndex].details.coatingFront=CT_NOCOATING;
						surfaceArray[surfIndex].details.coatingSide=CT_NOCOATING;
					break;
						
				 }						
				 BEGIN OBJDET;
		}
		
		<OBJDET_COAT>"\n" {
				 BEGIN OBJDET;
		}

 /*------------------------- radius --------------------------------------*/
  <OBJDET>"Radius 1"{ws}*":" {
			 fprintf(yyout, "radius1:");
			 BEGIN OBJDET_RAD1;
  }

		<OBJDET_RAD1>{number} {
				 if (geomParseFlag_Rad1==true)
				 {
					surfaceArray[surfIndex].radius1.y=atof(yytext)*scale2mm;
					fprintf(yyout, " %s;", yytext);
				 }
				 else
				 {
					 surfaceArray[surfIndex].radius1.x=atof(yytext)*scale2mm;
					 surfaceArray[surfIndex].radius1.y=atof(yytext)*scale2mm;
					 geomParseFlag_Rad1=true;
					 fprintf(yyout, " %s;", yytext);
				 }
				 BEGIN OBJDET_RAD1;
		}
		
		<OBJDET_RAD1>"\n" {
				 if (geomParseFlag_Rad1==false)
				 {
					fprintf(yyout, "error: definition of radius1 is incomplete.");
					LexerErr=true;
				 }
				 BEGIN OBJDET;
		}
		

  <OBJDET>"Radius 2"{ws}*":" {
			 fprintf(yyout, "radius2:");
			 BEGIN OBJDET_RAD2;
  }

		<OBJDET_RAD2>{number} {
				 if (geomParseFlag_Rad2==true)
				 {
					surfaceArray[surfIndex].radius2.y=atof(yytext)*scale2mm;
					fprintf(yyout, " %s;", yytext);
				 }
				 else
				 {
					 surfaceArray[surfIndex].radius2.x=atof(yytext)*scale2mm;
					 surfaceArray[surfIndex].radius2.y=atof(yytext)*scale2mm;
					 geomParseFlag_Rad2=true;
					 fprintf(yyout, " %s;", yytext);
				 }
				 BEGIN OBJDET_RAD2;
		}

		<OBJDET_RAD2>"\n" {
				 if (geomParseFlag_Rad2==false)
				 {
					fprintf(yyout, "error: definition of radius1 is incomplete.");
					LexerErr=true;
				 }
				 BEGIN OBJDET;
		}
		

  <OBJDET>"Radius"{ws}*":" {
			 fprintf(yyout, "radius:");
			 BEGIN OBJDET_RAD;
  }

		<OBJDET_RAD>{number} {
				 surfaceArray[surfIndex].radius1.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].radius1.y=atof(yytext)*scale2mm;
				 geomParseFlag_Rad1=true;
				 geomParseFlag_Rad2=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

	  <OBJDET>"Coeff on r" {
			 fprintf(yyout, "coeff");
			 geomParseFlag_AsphCoeff=true;
			 BEGIN OBJDET_ASPH_COEFF;
	  }
  
			<OBJDET_ASPH_COEFF>{dig}+{ws}*":" {
					asphere_coeff_index=atoi(yytext);
					fprintf(yyout, " %s", yytext);
					BEGIN OBJDET_ASPH_PAR;
			}

				<OBJDET_ASPH_PAR>{number}+ {
						if (surfaceArray[surfIndex].type==OT_ODDASPH) 
						{
							surfaceArray[surfIndex].details.asphereParams[asphere_coeff_index-1]=atof(yytext)*scale2mm;
						}
						else
						{
							surfaceArray[surfIndex].details.asphereParams[(asphere_coeff_index-2)/2]=atof(yytext)*scale2mm;
						}
						fprintf(yyout, " %s;", yytext);
						BEGIN OBJDET;
				}


 /*---------------------------------- conic ----------------------------------*/

  <OBJDET>"Conic 1"{ws}*":" {
			 fprintf(yyout, "conic1:");
			 BEGIN OBJDET_CON1;
  }

		<OBJDET_CON1>{number} {
				 surfaceArray[surfIndex].conic1=atof(yytext)/scale2mm;
				 fprintf(yyout, " %s;", yytext);
				 geomParseFlag_Con1=true;
				 BEGIN OBJDET;
		}

  <OBJDET>"Conic 2"{ws}*":" {
			 fprintf(yyout, "conic2:");
			 BEGIN OBJDET_CON2;
  }

		<OBJDET_CON2>{number} {
				 surfaceArray[surfIndex].conic2=atof(yytext)/scale2mm;
				 fprintf(yyout, " %s;", yytext);
				 geomParseFlag_Con2=true;
				 BEGIN OBJDET;
		}		

  <OBJDET>"Conic"{ws}*":" {
			 fprintf(yyout, "conic:");
			 BEGIN OBJDET_CON;
  }

		<OBJDET_CON>{number} {
				 surfaceArray[surfIndex].conic1=atof(yytext)/scale2mm;
				 surfaceArray[surfIndex].conic2=atof(yytext)/scale2mm;
				 geomParseFlag_Con1=true;
				 geomParseFlag_Con2=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

 /*----------------------------- aperture -------------------------------*/

  <OBJDET>"X Half Width"{ws}*":" {
			 fprintf(yyout, "x-half:");
			 BEGIN OBJDET_XHALFWIDTH;
  }

  <OBJDET>"X Half-Width"{ws}*":" {
			 fprintf(yyout, "x-half:");
			 BEGIN OBJDET_XHALFWIDTH;
  }

		<OBJDET_XHALFWIDTH>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.x=atof(yytext)*scale2mm;
				 geomParseFlag_Apert1X=true;
				 geomParseFlag_Apert2X=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

  <OBJDET>"Radial Height"{ws}*":" {
			 fprintf(yyout, "y-half:");
			 BEGIN OBJDET_YHALFWIDTH;
  }

  <OBJDET>"Y Half Width"{ws}*":" {
			 fprintf(yyout, "y-half:");
			 BEGIN OBJDET_YHALFWIDTH;
  }

		<OBJDET_YHALFWIDTH>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.y=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.y=atof(yytext)*scale2mm;
				 fprintf(yyout, " %s;", yytext);
				 geomParseFlag_Apert1Y=true;
				 geomParseFlag_Apert2Y=true;
				 BEGIN OBJDET;
		}

		
  <OBJDET>"Edge 1"{ws}*":" {
			 fprintf(yyout, "edge1:");
			 BEGIN OBJDET_EDGE1;
  }

		<OBJDET_EDGE1>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth1.y=atof(yytext)*scale2mm;
				 geomParseFlag_Apert1X=true;
				 geomParseFlag_Apert1Y=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

  <OBJDET>"Edge 2"{ws}*":" {
			 fprintf(yyout, "edge2:");
			 BEGIN OBJDET_EDGE2;
  }

		<OBJDET_EDGE2>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth2.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.y=atof(yytext)*scale2mm;
				 geomParseFlag_Apert2X=true;
				 geomParseFlag_Apert2Y=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

  <OBJDET>"Edge"{ws}*":" {
			 fprintf(yyout, "edge:");
			 BEGIN OBJDET_EDGE;
  }

		<OBJDET_EDGE>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth1.y=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.y=atof(yytext)*scale2mm;
				 geomParseFlag_Apert1X=true;
				 geomParseFlag_Apert1Y=true;
				 geomParseFlag_Apert2X=true;
				 geomParseFlag_Apert2Y=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

  <OBJDET>"Max X"{ws}*":" {
			 fprintf(yyout, "max x:");
			 BEGIN OBJDET_MAXAPRT_X;
  }

		<OBJDET_MAXAPRT_X>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.x=atof(yytext)*scale2mm;
				 geomParseFlag_Apert1X=true;
				 geomParseFlag_Apert2X=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

  <OBJDET>"Max Y"{ws}*":" {
			 fprintf(yyout, "max y:");
			 BEGIN OBJDET_MAXAPRT_Y;
  }

		<OBJDET_MAXAPRT_Y>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.y=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.y=atof(yytext)*scale2mm;
				 geomParseFlag_Apert1Y=true;
				 geomParseFlag_Apert2Y=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

		
  <OBJDET>"Max Aper"{ws}*":" {
			 fprintf(yyout, "max aper:");
			 BEGIN OBJDET_MAXAPRT;
  }

		<OBJDET_MAXAPRT>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth1.y=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.y=atof(yytext)*scale2mm;
				 geomParseFlag_Apert1X=true;
				 geomParseFlag_Apert1Y=true;
				 geomParseFlag_Apert2X=true;
				 geomParseFlag_Apert2Y=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}
		
  <OBJDET>"Min Aper"{ws}*":" {
			 fprintf(yyout, "min aper:");
			 BEGIN OBJDET_MINAPRTX;
  }

		<OBJDET_MINAPRTX>{number} {
				 surfaceArray[surfIndex].details.obscurationHalfWidth.x=atof(yytext)*scale2mm;
				 geomParseFlag_MinApert1X=true;
				 fprintf(yyout, " x=%s;", yytext);
				 BEGIN OBJDET_MINAPRTY;
		}

		<OBJDET_MINAPRTY>{number} {
				 surfaceArray[surfIndex].details.obscurationHalfWidth.y=atof(yytext)*scale2mm;
				 geomParseFlag_MinApert1Y=true;
				 fprintf(yyout, " y=%s;", yytext);
				 BEGIN OBJDET;
		}
		
		<OBJDET_MINAPRTY>"\n" {
				 surfaceArray[surfIndex].details.obscurationHalfWidth.y=surfaceArray[surfIndex].details.obscurationHalfWidth.x;
				 geomParseFlag_MinApert1Y=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}


  <OBJDET>"Front R"{ws}*":" {
			 fprintf(yyout, "front radius:");
			 BEGIN OBJDET_FRONTAPRT;
  }

		<OBJDET_FRONTAPRT>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth1.y=atof(yytext)*scale2mm;
				 geomParseFlag_Apert1X=true;
				 geomParseFlag_Apert1Y=true;				 
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}
		
  <OBJDET>"Back R"{ws}*":" {
			 fprintf(yyout, "back radius:");
			 BEGIN OBJDET_BACKAPRT;
  }

		<OBJDET_BACKAPRT>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth2.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.y=atof(yytext)*scale2mm;
				 geomParseFlag_Apert2X=true;
				 geomParseFlag_Apert2Y=true;				 
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}	
		
  <OBJDET>"Radial Aperture"{ws}*":" {
			 fprintf(yyout, "radial radius:");
			 BEGIN OBJDET_RADAPRT;
  }

		<OBJDET_RADAPRT>{number} {
				 surfaceArray[surfIndex].details.apertureHalfWidth1.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth1.y=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.x=atof(yytext)*scale2mm;
				 surfaceArray[surfIndex].details.apertureHalfWidth2.y=atof(yytext)*scale2mm;
				 geomParseFlag_Apert1X=true;
				 geomParseFlag_Apert1Y=true;
				 geomParseFlag_Apert2X=true;
				 geomParseFlag_Apert2Y=true;				 
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}	

  <OBJDET>"Is Rectangle?"{ws}*":" {
			 fprintf(yyout, "is rectangle? :");
			 BEGIN OBJDET_APRTTYPE;
  }
		
		<OBJDET_APRTTYPE>{dig} {
				 int help=atoi(yytext);
				 if (help == 1)
				 {
					surfaceArray[surfIndex].details.aperture=AT_RECT;
				 }
				 else
				 {
					surfaceArray[surfIndex].details.aperture=AT_ELLIPT;
				 }
				 fprintf(yyout, " %s;", yytext);
				 geomParseFlag_ApertType=true;
				 BEGIN OBJDET;
		}

 /*-------------------------------- ideal lense ------------------------------------*/

  <OBJDET>"Lambda0(um)"{ws}*":" {
			 if (surfaceArray[surfIndex].type!=OT_IDEALLENSE)
			 {
				fprintf(yyout, "warning: centre wavelength outside ideal lense definition!!");
				std::cout << "warning in yylex(): centre wavelength outside ideal lense definition at surface index:" << surfIndex << std::endl;
			 }				
			 else
				fprintf(yyout, "centre wavlength:");
			 BEGIN OBJDET_LAMBDA0;
  }

		<OBJDET_LAMBDA0>{num} {
				 surfaceArray[surfIndex].details.idealLense_lambda0=atof(yytext);
				 geomParseFlag_IdealLenseLambda0=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

		<OBJDET_LAMBDA0>"\n" {
				 fprintf(yyout, " no centre wavelnegth definition found!!");
				 surfaceArray[surfIndex].details.idealLense_lambda0=0;
				 BEGIN OBJDET;
		}

  <OBJDET>"Focal length"{ws}*":" {
			 if (surfaceArray[surfIndex].type!=OT_IDEALLENSE)
			 {
				fprintf(yyout, "warning: centre focal length outside ideal lense definition!!");
				std::cout << "warning in yylex(): centre wavelength outside ideal lense definition at surface index:" << surfIndex << std::endl;
			 }
			 else
				fprintf(yyout, "centre focal length:");
			 BEGIN OBJDET_F0;
  }

		<OBJDET_F0>{num} {
				 surfaceArray[surfIndex].details.idealLense_f0=atof(yytext);
				 geomParseFlag_IdealLenseF0=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

		<OBJDET_F0>"\n" {
				 fprintf(yyout, " no centre focal length definition found!!");
				 surfaceArray[surfIndex].details.idealLense_f0=0;
				 BEGIN OBJDET;
		}

  <OBJDET>"Chromacy constant"{ws}*":" {
			 if (surfaceArray[surfIndex].type!=OT_IDEALLENSE)
			 {
				fprintf(yyout, "error: chromacy constant outside ideal lense definition!!");
				std::cout << "warning in yylex(): chromacy constant outside ideal lense definition at surface index:" << surfIndex << std::endl;
			 }				
			 else
				fprintf(yyout, "chromacy constant:");
			 BEGIN OBJDET_A;
  }

		<OBJDET_A>{num} {
				 surfaceArray[surfIndex].details.idealLense_A=atof(yytext);
				 geomParseFlag_IdealLenseA=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}		

		<OBJDET_A>"\n" {
				 fprintf(yyout, " no chromacy constant definition found!!");
				 surfaceArray[surfIndex].details.idealLense_A=0;
				 BEGIN OBJDET;
		}

 /*-------------------------------- diffraction grating ------------------------------------*/

   <OBJDET>"Lines/µm"{ws}*":" {
			 if (surfaceArray[surfIndex].type!=OT_DGRATING)
			 {
				fprintf(yyout, "warning: lines per µm outside grating definition!!");
				std::cout << "warning in yylex(): lines per µm outside grating definition at surface index:" << surfIndex << std::endl;
			 }
			 else
				fprintf(yyout, "lines per µm:");
			 BEGIN OBJDET_LINES;
  }

		<OBJDET_LINES>"file" {
				 surfaceArray[surfIndex].details.gratingLinesFromFile=true;
				 geomParseFlag_DiffLines=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}	

		<OBJDET_LINES>{number} {
				 surfaceArray[surfIndex].details.gratingLinesFromFile=false;
				 surfaceArray[surfIndex].details.lines=1/atof(yytext)*scale2um;
				 geomParseFlag_DiffLines=true;
				 fprintf(yyout, " %f;", surfaceArray[surfIndex].details.lines);
				 BEGIN OBJDET;
		}	

  <OBJDET>"Diff Order"{ws}*":" {
			 if (surfaceArray[surfIndex].type!=OT_DGRATING)
			 {
				fprintf(yyout, "warning: diffraction order outside grating definition!!");
				std::cout << "warning in yylex(): diffraction order outside grating definition at surface index:" << surfIndex << std::endl;
			 }
			 else
			 {
				fprintf(yyout, "diffraction order:");
				diffOrderIndex=0;
			 }
			 BEGIN OBJDET_DIFFORDER;
  }

		<OBJDET_DIFFORDER>"file" {
				 surfaceArray[surfIndex].details.gratingOrdersFromFile=true;
				 geomParseFlag_DiffOrder=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}			

		<OBJDET_DIFFORDER>{number} {
				 if (diffOrderIndex < MAX_NR_DIFFORDERS)
				 {
				    surfaceArray[surfIndex].details.gratingOrdersFromFile=false;
					surfaceArray[surfIndex].details.diffOrder[diffOrderIndex]=atoi(yytext);
					diffOrderIndex++;
					fprintf(yyout, " %i;", surfaceArray[surfIndex].details.diffOrder[diffOrderIndex-1]);
				 }
				 else
				 {
					fprintf(yyout, "ERROR!! too many diffraction orders");
				 }
				 geomParseFlag_DiffOrder=true;
				 BEGIN OBJDET_DIFFORDER;
		}			

		<OBJDET_DIFFORDER>"\n" {
				 surfaceArray[surfIndex].details.nrDiffOrders=diffOrderIndex;
				 BEGIN OBJDET;
		}	

  <OBJDET>"Efficiencies"{ws}*":" {
			 if (surfaceArray[surfIndex].type!=OT_DGRATING)
			 {
				fprintf(yyout, "warning: diffraction efficiency outside grating definition!!");
				std::cout << "warning in yylex(): diffraction efficiency outside grating definition at surface index:" << surfIndex << std::endl;
			 }
			 else
			 {
				fprintf(yyout, "efficiencies:");
				diffEffIndex=0;
			 }
			 BEGIN OBJDET_EFFS;
  }		
  
		<OBJDET_EFFS>"file" {
				 surfaceArray[surfIndex].details.gratingEffsFromFile=true;
				 fprintf(yyout, " %s;", yytext);
				 geomParseFlag_DiffEffs=true;
				 BEGIN OBJDET;
		}			

		<OBJDET_EFFS>{number} {
				 geomParseFlag_DiffEffs=true;
				 if (diffEffIndex < MAX_NR_DIFFORDERS)
				 {
				    surfaceArray[surfIndex].details.gratingEffsFromFile=false;
					surfaceArray[surfIndex].details.diffEff[diffEffIndex]=atof(yytext);
					diffEffIndex++;
					fprintf(yyout, " %s;", yytext);
					
					BEGIN OBJDET_EFFS;
				 }
				 else
				 {
					fprintf(yyout, "ERROR!! too many diffraction efficiencies");
					BEGIN OBJDET;
				 }			 
		}			

		<OBJDET_EFFS>"\n" {
				 if (surfaceArray[surfIndex].details.nrDiffOrders!=diffEffIndex)
					fprintf(yyout, "ERROR!!! different number of efficiencies than diffraction orders!!");
				 BEGIN OBJDET;
		}	
  
 /*-------------------------------- cosine normal ------------------------------------*/			

   <OBJDET>"Groove Period(um)"{ws}*":" {
			 fprintf(yyout, "groove period:");
			 BEGIN OBJDET_COSINENORMAL_PERIOD;
  }

	   <OBJDET_COSINENORMAL_PERIOD>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.cosNormPeriod=atof(yytext)/1000*scale2mm;
				 geomParseFlag_CosNormPeriod=true;
				 BEGIN OBJDET;
	  }

	   <OBJDET_COSINENORMAL_PERIOD>"\n" {
				 fprintf(yyout, " definition not found!! ", yytext);
				 surfaceArray[surfIndex].details.cosNormPeriod=0;
				 geomParseFlag_CosNormPeriod=false;
				 BEGIN OBJDET;
	  }

   <OBJDET>"Groove Amplitude(um)"{ws}*":" {
			 fprintf(yyout, "groove amplitude:");
			 BEGIN OBJDET_COSINENORMAL_AMPL;
  }

	   <OBJDET_COSINENORMAL_AMPL>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.cosNormAmpl=atof(yytext)/1000*scale2mm;
				 geomParseFlag_CosNormAmpl=true;
				 BEGIN OBJDET;
	  }

	   <OBJDET_COSINENORMAL_AMPL>"\n" {
				 fprintf(yyout, " definition not found!! ", yytext);
				 surfaceArray[surfIndex].details.cosNormAmpl=0;
				 geomParseFlag_CosNormAmpl=false;
				 BEGIN OBJDET;
	  }

   <OBJDET>"Iteration Accuracy(um)"{ws}*":" {
			 fprintf(yyout, "iterationaccuracy:");
			 BEGIN OBJDET_COSINENORMAL_ITACCU;
  }

	   <OBJDET_COSINENORMAL_ITACCU>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.iterationAccuracy=atof(yytext)/1000*scale2mm;
				 BEGIN OBJDET;
	  }

	   <OBJDET_COSINENORMAL_ITACCU>"\n" {
				 fprintf(yyout, " definition not found!! ", yytext);
				 surfaceArray[surfIndex].details.iterationAccuracy=DOUBLE_MAX;
				 BEGIN OBJDET;
	  }


   <OBJDET>"Groove Axis"{ws}*":" {
			 fprintf(yyout, "groove axis:");
			 BEGIN OBJDET_COSINENORMAL_AXISX;
  }

	   <OBJDET_COSINENORMAL_AXISX>{num} {
				 fprintf(yyout, " %s;", yytext);
				 surfaceArray[surfIndex].details.cosNormAxis.x=atof(yytext);
				 BEGIN OBJDET_COSINENORMAL_AXISY;
	  }

	   <OBJDET_COSINENORMAL_AXISX>"\n" {
				 fprintf(yyout, " definition not found!! ", yytext);
				 surfaceArray[surfIndex].details.cosNormAxis=make_double3(0,0,0);
				 geomParseFlag_CosNormAxis=false;
				 BEGIN OBJDET;
	  }

		   <OBJDET_COSINENORMAL_AXISY>{num} {
					 fprintf(yyout, " %s;", yytext);
					 surfaceArray[surfIndex].details.cosNormAxis.y=atof(yytext);
					 BEGIN OBJDET_COSINENORMAL_AXISZ;
		  }
	  
		   <OBJDET_COSINENORMAL_AXISY>"\n" {
					 fprintf(yyout, " definition not found!! ", yytext);
					 surfaceArray[surfIndex].details.cosNormAxis=make_double3(0,0,0);
					 geomParseFlag_CosNormAxis=false;
					 BEGIN OBJDET;
		  }		
		  
			   <OBJDET_COSINENORMAL_AXISZ>{num} {
						 fprintf(yyout, " %s;", yytext);
						 surfaceArray[surfIndex].details.cosNormAxis.z=atof(yytext);
						 geomParseFlag_CosNormAxis=true;
						 BEGIN OBJDET;
			  }		  
 /*-------------------------------- thickness ------------------------------------*/
 
   <OBJDET>"Thickness"{ws}*":" {
			 fprintf(yyout, "thickness:");
			 BEGIN OBJDET_THK;
  }

   <OBJDET>"Thick"{ws}*":" {
			 fprintf(yyout, "thickness:");
			 BEGIN OBJDET_THK;
  }

		<OBJDET_THK>{number} {
				 surfaceArray[surfIndex].thickness=atof(yytext)*scale2mm;
				 fprintf(yyout, " %s;", yytext);
				 geomParseFlag_Thickness=true;
				 BEGIN OBJDET;
		}	
 

   <OBJDET>"Z Length"{ws}*":" {
			 fprintf(yyout, "z-length:");
			 BEGIN OBJDET_ZLEN;
  }

		<OBJDET_ZLEN>{number} {
				 surfaceArray[surfIndex].thickness=atof(yytext)*scale2mm;
				 geomParseFlag_Thickness=true;
				 fprintf(yyout, " %s;", yytext);
				 BEGIN OBJDET;
		}

.|\n {}

%%

// ____________________________________________________________

//int main()
//    {
//		yyout = fopen( "e:\\mauch\\prescriptionLexOut.txt", "w" ) ;
//		yyin = fopen( "e:\\mauch\\prescription.txt", "r" ) ;
    
//		while(yylex() != 0)
//			;
//		fclose(yyout);
//    }

parserError parseMicroSimGratingData(ParseGratingResultStruct** parseResultsGratingPtrPtr, FILE *hfile)
{
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILES_PATH, "gratingLexOut.txt");
	yyout = fopen( filepath, "w" ) ;
	gratingResultStructPtr=new ParseGratingResultStruct();
	yyin = hfile;
    /* do the actual parsing */
	while(yylex() != 0)
		;
	fclose(yyout);

	sprintf(gratingResultStructPtr->errMsg, "no error");
	*parseResultsGratingPtrPtr=gratingResultStructPtr;
	return PARSER_NO_ERR;
}

parserError parseZemaxGlassCatalog(parseGlassResultStruct** parseResultsGlassPtrPtr, FILE *hfile, char *glassName)
{

	return PARSER_NO_ERR;
}	

parserError parseZemaxPrescr(parseResultStruct** parseResultsGeometry,  FILE *hfile, simMode mode)
{
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILES_PATH, "prescriptionNonSeqLexOut.txt");
	yyout = fopen( filepath, "w" ) ;
	yyin = hfile;

	l_mode=mode; // make simulation known to the parser ( sequential mode does not create all the surfaces that would be created for nonsequential mode. Side faces of lenses are not sequential for example...

	parseResultStruct *l_parseResultsPtr;
	l_parseResultsPtr = (parseResultStruct*) malloc(1*sizeof(parseResultStruct));
	
	// init the error message
	sprintf(l_parseResultsPtr->errMsg, "%s", "no error");
	
	/* init index variables for the current parsing */
	surfDetIndex = -1; // init to -1 because of weird incrementing inside parser
	surfIndex = -1; // init to -1 because of weird incrementing inside parser
	geomIndex = 0;	
    
    /* do the actual parsing */
	while(yylex() != 0)
		;

	// check the last object that has been parsed. ( all other objects have been checked inside yylex8() )
	if ( !checkObjectDefinition(&(surfaceArray[surfIndex])) )
	{
		fprintf( yyout, "error: object %i was not fully defined!!!", (surfIndex+1));
		std::cout << "error in yylex(): object " << (surfIndex+1) << " was not fully defined" << std::endl;
		LexerErr=true;
	}
	if (LexerErr)
	{
		std::cout << "error in parseZemaxPrescr: yylex() returned an error" << std::endl;
		fclose(yyout);
		free(surfaceArray);
		
		return PARSER_ERR;
	}
	
	/* allocate memory to store the result of the detected geometries. Not that in general this number will be different to the detected Zemax surfaces ( Coordinate breaks, standard lenses.. ) */
	l_parseResultsPtr->geometryParams = new GeometryParseParamStruct[geomIndex];
	l_parseResultsPtr->geomNumber=geomIndex;
	/* allocate memory to store the result of the detected sources */
	l_parseResultsPtr->sourceParams = new FieldParseParamStruct[srcIndex];
	l_parseResultsPtr->sourceNumber=srcIndex;
	/* allocate memory to store the result of the detected detectors */
	l_parseResultsPtr->detectorParams = new DetectorParseParamStruct[detIndex];
	l_parseResultsPtr->detectorNumber=detIndex;
	/* allocate memory to store the result of the detected sources */
	l_parseResultsPtr->pupilParams = new PupilParseParamStruct[srcIndex];
	l_parseResultsPtr->pupilNumber=pupilIndex;
	
	/* indicate simulation method */
	l_parseResultsPtr->simType=simType;
	
	/* init global coordinate system */
	globalRoot=make_double3(0.0,0.0,0.0);
	globalNormal=make_double3(0.0,0.0,1.0);
	/* init shift vector. This is needed to realize decentering of surfaces and coordinate breaks */
	double3 shiftVec=make_double3(0.0,0.0,0.0);	
	/* declare axis of diffraction for grating objects */
	double3 diffAxis;

	/* define loop counters */
	long long int iSurf;
	long long int iGeom=0;
	long long int iSrc=0;
	long long int iDet=0;
	long long int iPupil=0;
	/* loop through surfaceArray and place geometries in global coordinate system */
	for (iSurf=0; iSurf<=surfIndex ;iSurf++)
	{
		ZemaxParamStruct test=surfaceArray[iSurf];
		if (iGeom<=geomIndex)
		{
			/* copy the geometry params */
			l_parseResultsPtr->geometryParams[iGeom].normal=make_double3(0.0,0.0,1.0);
			rotateRay(&l_parseResultsPtr->geometryParams[iGeom].normal,surfaceArray[iSurf].details.tilt1);
			l_parseResultsPtr->geometryParams[iGeom].root=surfaceArray[iSurf].details.root;//globalRoot;
			l_parseResultsPtr->geometryParams[iGeom].tilt=surfaceArray[iSurf].details.tilt1;
			l_parseResultsPtr->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
			l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth1;
			l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth2=surfaceArray[iSurf].details.apertureHalfWidth2;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=surfaceArray[iSurf].glass;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.filterMax=surfaceArray[iSurf].details.filterMax;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.filterMin=surfaceArray[iSurf].details.filterMin;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.tilt=surfaceArray[iSurf].details.tilt1;
//			if (surfaceArray[iSurf].glass==MT_UNKNOWNMATERIAL)
//			{
//				std::cout << "warning in parseZemaxPrescr: object number " << iSurf+1 << " has an unknown material." << std::endl;
//			}
			l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType=surfaceArray[iSurf].details.scatterFront;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingType=surfaceArray[iSurf].details.coatingFront;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
			l_parseResultsPtr->geometryParams[iGeom].thickness=surfaceArray[iSurf].thickness;
			l_parseResultsPtr->geometryParams[iGeom].conic1=surfaceArray[iSurf].conic1;
			l_parseResultsPtr->geometryParams[iGeom].radius1=surfaceArray[iSurf].radius1;
			//l_parseResultsPtr->geometryParams[iGeom].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
			//l_parseResultsPtr->geometryParams[iGeom].rotNormal2=surfaceArray[iSurf].details.tilt2.z;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_r=surfaceArray[iSurf].details.coating_front_r;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_t=surfaceArray[iSurf].details.coating_front_t;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_a_r=surfaceArray[iSurf].details.coating_front_a_r;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_a_t=surfaceArray[iSurf].details.coating_front_a_t;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_c_r=surfaceArray[iSurf].details.coating_front_c_r;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_c_t=surfaceArray[iSurf].details.coating_front_c_t;		
			l_parseResultsPtr->geometryParams[iGeom].obscurationHalfWidth=surfaceArray[iSurf].details.obscurationHalfWidth;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.importanceObjNr=surfaceArray[iSurf].details.importanceObjNr;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.importanceConeAlphaMin=surfaceArray[iSurf].details.importanceConeAlphaMin;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.importanceConeAlphaMax=surfaceArray[iSurf].details.importanceConeAlphaMax;
			l_parseResultsPtr->geometryParams[iGeom].materialParams.importanceArea=surfaceArray[iSurf].details.importanceArea;
			memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
			memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
			memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );

			switch(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType)
			{
				case ST_TORRSPARR1D:
				case ST_TORRSPARR2D:
					l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
					rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis),surfaceArray[iSurf].details.tilt1);
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, MAX_NR_MATPARAMS*sizeof(double) );
					break;
				case ST_DISPDOUBLECAUCHY1D:
				case ST_DOUBLECAUCHY1D:
					l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis=make_double3(surfaceArray[iSurf].details.matParams[4],surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6]);
					rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis),surfaceArray[iSurf].details.tilt1);
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, MAX_NR_MATPARAMS*sizeof(double) );
					break;		
				case ST_LAMBERT2D:
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, MAX_NR_MATPARAMS*sizeof(double) );
					break;						
				default:
					break;
			}

			switch(surfaceArray[iSurf].type)	
			{
				case OT_ILLPUPIL:
					l_parseResultsPtr->pupilParams[iPupil].type=PUP_ILLU; 
					l_parseResultsPtr->pupilParams[iPupil].apertureHalfWidth=surfaceArray[iSurf].details.apertureHalfWidth1;
					l_parseResultsPtr->pupilParams[iPupil].root=surfaceArray[iSurf].details.root;
					l_parseResultsPtr->pupilParams[iPupil].tilt=surfaceArray[iSurf].details.tilt1;
					l_parseResultsPtr->pupilParams[iPupil].apertureType=surfaceArray[iSurf].details.aperture;
					l_parseResultsPtr->pupilParams[iPupil].pupilID=iPupil;
					
					// increment pupil counter
					iPupil=iPupil+1;
					break;
								
				case OT_IDEALLENSE:
					l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=MT_IDEALLENSE; // ideal lenses can only have material type ideal lense
					l_parseResultsPtr->geometryParams[iGeom].materialParams.idealLense_f0=surfaceArray[iSurf].details.idealLense_f0;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.idealLense_A=surfaceArray[iSurf].details.idealLense_A;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.idealLense_lambda0=surfaceArray[iSurf].details.idealLense_lambda0;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.root=surfaceArray[iSurf].details.root;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.root=surfaceArray[iSurf].details.root;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.normal=make_double3(0.0,0.0,1.0);
					rotateRay(&l_parseResultsPtr->geometryParams[iGeom].materialParams.normal,surfaceArray[iSurf].details.tilt1);
					l_parseResultsPtr->geometryParams[iGeom].materialParams.apertureHalfWidth=surfaceArray[iSurf].details.apertureHalfWidth1;
					
					l_parseResultsPtr->geometryParams[iGeom].type=GEOM_IDEALLENSE;
					
					// increment geometry counter
					iGeom=iGeom+1;
					break;
					
				case OT_SRC:
				    l_parseResultsPtr->sourceParams[iSrc].type=SRC;
					l_parseResultsPtr->sourceParams[iSrc].coherence=surfaceArray[iSurf].details.srcCoherence;
					l_parseResultsPtr->sourceParams[iSrc].normal=make_double3(0.0,0.0,1.0);
					rotateRay(&l_parseResultsPtr->sourceParams[iSrc].normal,surfaceArray[iSurf].details.tilt1);
					l_parseResultsPtr->sourceParams[iSrc].root=surfaceArray[iSurf].details.root;//globalRoot;
					l_parseResultsPtr->sourceParams[iSrc].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth1;
					l_parseResultsPtr->sourceParams[iSrc].apertureHalfWidth2=surfaceArray[iSurf].details.apertureHalfWidth2;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.matType=surfaceArray[iSurf].glass;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.scatterType=surfaceArray[iSurf].details.scatterFront;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.coatingType=surfaceArray[iSurf].details.coatingFront;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
					l_parseResultsPtr->sourceParams[iSrc].importanceObjNr=surfaceArray[iSurf].details.importanceObjNr;
					l_parseResultsPtr->sourceParams[iSrc].importanceConeAlphaMin=surfaceArray[iSurf].details.importanceConeAlphaMin;
					l_parseResultsPtr->sourceParams[iSrc].importanceConeAlphaMax=surfaceArray[iSurf].details.importanceConeAlphaMax;
					l_parseResultsPtr->sourceParams[iSrc].importanceArea=surfaceArray[iSurf].details.importanceArea;
					
					memcpy ( l_parseResultsPtr->sourceParams[iSrc].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->sourceParams[iSrc].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );
					//l_parseResultsPtr->sourceParams[iSrc].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
				
					//l_parseResultsPtr->geometryParams[iGeom].type=SRC;
					l_parseResultsPtr->sourceParams[iSrc].width=surfaceArray[iSurf].details.rayFieldWidth;
					l_parseResultsPtr->sourceParams[iSrc].height=surfaceArray[iSurf].details.rayFieldHeight;
					l_parseResultsPtr->sourceParams[iSrc].widthLayout=surfaceArray[iSurf].details.rayFieldWidthLayout;
					l_parseResultsPtr->sourceParams[iSrc].heightLayout=surfaceArray[iSurf].details.rayFieldHeightLayout;
					l_parseResultsPtr->sourceParams[iSrc].power=surfaceArray[iSurf].details.power;
					l_parseResultsPtr->sourceParams[iSrc].rayDirDistr=surfaceArray[iSurf].details.rayDirDistr;
					l_parseResultsPtr->sourceParams[iSrc].rayPosDistr=surfaceArray[iSurf].details.rayPosDistr;
					l_parseResultsPtr->sourceParams[iSrc].lambda=surfaceArray[iSurf].details.lambda;
					l_parseResultsPtr->sourceParams[iSrc].rayDirection=createObliqueVec(surfaceArray[iSurf].details.rayDirectionTilt);//make_double3(0,0,1); // init ray direction along z-axis
					l_parseResultsPtr->sourceParams[iSrc].alphaMax=surfaceArray[iSurf].details.alphaMax;
					l_parseResultsPtr->sourceParams[iSrc].alphaMin=surfaceArray[iSurf].details.alphaMin;
					l_parseResultsPtr->sourceParams[iSrc].nrRayDirections=surfaceArray[iSurf].details.nrRayDirections;
					//rotateRay(&(l_parseResultsPtr->sourceParams[iSrc].rayDirection),surfaceArray[iSurf].details.tilt1);// rotate ray direction as orientation of source
					//rotateRay(&(l_parseResultsPtr->sourceParams[iSrc].rayDirection),surfaceArray[iSurf].details.rayDirectionTilt);// rotate ray direction as orientation of source
					l_parseResultsPtr->sourceParams[iSrc].tilt=surfaceArray[iSurf].details.tilt1;

					iSrc=iSrc+1;
					break;

				case OT_DIFFSRC_RAYAIM:
				case OT_DIFFSRC_FREEFORM:
				case OT_DIFFSRC_HOLO:
				case OT_DIFFSRC:
					l_parseResultsPtr->sourceParams[iSrc].type=GEOM_UNKNOWN;
					if (surfaceArray[iSurf].type == OT_DIFFSRC)
						l_parseResultsPtr->sourceParams[iSrc].type=DIFFSRC;
					if (surfaceArray[iSurf].type == OT_DIFFSRC_FREEFORM)
						l_parseResultsPtr->sourceParams[iSrc].type=DIFFSRC_FREEFORM;
					if (surfaceArray[iSurf].type == OT_DIFFSRC_RAYAIM)
						l_parseResultsPtr->sourceParams[iSrc].type=DIFFSRC_RAYAIM;	
					if (surfaceArray[iSurf].type == OT_DIFFSRC_HOLO)
						l_parseResultsPtr->sourceParams[iSrc].type=DIFFSRC_HOLO;	
					l_parseResultsPtr->sourceParams[iSrc].coherence=surfaceArray[iSurf].details.srcCoherence;
					l_parseResultsPtr->sourceParams[iSrc].normal=make_double3(0.0,0.0,1.0);
					rotateRay(&l_parseResultsPtr->sourceParams[iSrc].normal,surfaceArray[iSurf].details.tilt1);
					l_parseResultsPtr->sourceParams[iSrc].root=surfaceArray[iSurf].details.root;//globalRoot;
					l_parseResultsPtr->sourceParams[iSrc].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth1;
					l_parseResultsPtr->sourceParams[iSrc].apertureHalfWidth2=surfaceArray[iSurf].details.apertureHalfWidth2;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.matType=surfaceArray[iSurf].glass;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.scatterType=surfaceArray[iSurf].details.scatterFront;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.coatingType=surfaceArray[iSurf].details.coatingFront;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
					l_parseResultsPtr->sourceParams[iSrc].importanceObjNr=surfaceArray[iSurf].details.importanceObjNr;
					l_parseResultsPtr->sourceParams[iSrc].importanceConeAlphaMin=surfaceArray[iSurf].details.importanceConeAlphaMin;
					l_parseResultsPtr->sourceParams[iSrc].importanceConeAlphaMax=surfaceArray[iSurf].details.importanceConeAlphaMax;
					l_parseResultsPtr->sourceParams[iSrc].importanceArea=surfaceArray[iSurf].details.importanceArea;
					memcpy ( l_parseResultsPtr->sourceParams[iSrc].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->sourceParams[iSrc].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );
					//l_parseResultsPtr->sourceParams[iSrc].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
				
					l_parseResultsPtr->sourceParams[iSrc].width=surfaceArray[iSurf].details.rayFieldWidth;
					l_parseResultsPtr->sourceParams[iSrc].height=surfaceArray[iSurf].details.rayFieldHeight;
					l_parseResultsPtr->sourceParams[iSrc].widthLayout=surfaceArray[iSurf].details.rayFieldWidthLayout;
					l_parseResultsPtr->sourceParams[iSrc].heightLayout=surfaceArray[iSurf].details.rayFieldHeightLayout;
					l_parseResultsPtr->sourceParams[iSrc].power=surfaceArray[iSurf].details.power;
					l_parseResultsPtr->sourceParams[iSrc].rayDirDistr=surfaceArray[iSurf].details.rayDirDistr;
					l_parseResultsPtr->sourceParams[iSrc].rayPosDistr=surfaceArray[iSurf].details.rayPosDistr;
					l_parseResultsPtr->sourceParams[iSrc].lambda=surfaceArray[iSurf].details.lambda;
					l_parseResultsPtr->sourceParams[iSrc].rayDirection=createObliqueVec(surfaceArray[iSurf].details.rayDirectionTilt);//make_double3(0,0,1); // init ray direction along z-axis
					l_parseResultsPtr->sourceParams[iSrc].alphaMax=surfaceArray[iSurf].details.alphaMax;
					l_parseResultsPtr->sourceParams[iSrc].alphaMin=surfaceArray[iSurf].details.alphaMin;
					l_parseResultsPtr->sourceParams[iSrc].nrRayDirections=surfaceArray[iSurf].details.nrRayDirections;
					l_parseResultsPtr->sourceParams[iSrc].epsilon=surfaceArray[iSurf].details.epsilon;
					//rotateRay(&(l_parseResultsPtr->sourceParams[iSrc].rayDirection),surfaceArray[iSurf].details.tilt1);// rotate ray direction as orientation of source
					//rotateRay(&(l_parseResultsPtr->sourceParams[iSrc].rayDirection),surfaceArray[iSurf].details.rayDirectionTilt);// rotate ray direction as orientation of source
					l_parseResultsPtr->sourceParams[iSrc].tilt=surfaceArray[iSurf].details.tilt1;

					iSrc=iSrc+1;
					break;

				case OT_PATHTRACINGSRC:
				    l_parseResultsPtr->sourceParams[iSrc].type=SRC;
					l_parseResultsPtr->sourceParams[iSrc].coherence=surfaceArray[iSurf].details.srcCoherence;
					l_parseResultsPtr->sourceParams[iSrc].normal=make_double3(0.0,0.0,1.0);
					rotateRay(&l_parseResultsPtr->sourceParams[iSrc].normal,surfaceArray[iSurf].details.tilt1);
					l_parseResultsPtr->sourceParams[iSrc].root=surfaceArray[iSurf].details.root;//globalRoot;
					l_parseResultsPtr->sourceParams[iSrc].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth1;
					l_parseResultsPtr->sourceParams[iSrc].apertureHalfWidth2=surfaceArray[iSurf].details.apertureHalfWidth2;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.matType=surfaceArray[iSurf].glass;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.scatterType=surfaceArray[iSurf].details.scatterFront;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.coatingType=surfaceArray[iSurf].details.coatingFront;
					l_parseResultsPtr->sourceParams[iSrc].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
					l_parseResultsPtr->sourceParams[iSrc].importanceObjNr=surfaceArray[iSurf].details.importanceObjNr;
					l_parseResultsPtr->sourceParams[iSrc].importanceConeAlphaMin=surfaceArray[iSurf].details.importanceConeAlphaMin;
					l_parseResultsPtr->sourceParams[iSrc].importanceConeAlphaMax=surfaceArray[iSurf].details.importanceConeAlphaMax;
					l_parseResultsPtr->sourceParams[iSrc].importanceArea=surfaceArray[iSurf].details.importanceArea;
					
					memcpy ( l_parseResultsPtr->sourceParams[iSrc].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->sourceParams[iSrc].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );
					//l_parseResultsPtr->sourceParams[iSrc].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
				
					//l_parseResultsPtr->geometryParams[iGeom].type=SRC;
					l_parseResultsPtr->sourceParams[iSrc].width=surfaceArray[iSurf].details.rayFieldWidth;
					l_parseResultsPtr->sourceParams[iSrc].height=surfaceArray[iSurf].details.rayFieldHeight;
					l_parseResultsPtr->sourceParams[iSrc].widthLayout=surfaceArray[iSurf].details.rayFieldWidthLayout;
					l_parseResultsPtr->sourceParams[iSrc].heightLayout=surfaceArray[iSurf].details.rayFieldHeightLayout;
					l_parseResultsPtr->sourceParams[iSrc].power=surfaceArray[iSurf].details.power;
					l_parseResultsPtr->sourceParams[iSrc].rayDirDistr=surfaceArray[iSurf].details.rayDirDistr;
					l_parseResultsPtr->sourceParams[iSrc].rayPosDistr=surfaceArray[iSurf].details.rayPosDistr;
					l_parseResultsPtr->sourceParams[iSrc].lambda=surfaceArray[iSurf].details.lambda;
					l_parseResultsPtr->sourceParams[iSrc].rayDirection=createObliqueVec(surfaceArray[iSurf].details.rayDirectionTilt);//make_double3(0,0,1); // init ray direction along z-axis
					l_parseResultsPtr->sourceParams[iSrc].alphaMax=surfaceArray[iSurf].details.alphaMax;
					l_parseResultsPtr->sourceParams[iSrc].alphaMin=surfaceArray[iSurf].details.alphaMin;
					l_parseResultsPtr->sourceParams[iSrc].nrRayDirections=surfaceArray[iSurf].details.nrRayDirections;
					//rotateRay(&(l_parseResultsPtr->sourceParams[iSrc].rayDirection),surfaceArray[iSurf].details.tilt1);// rotate ray direction as orientation of source
					//rotateRay(&(l_parseResultsPtr->sourceParams[iSrc].rayDirection),surfaceArray[iSurf].details.rayDirectionTilt);// rotate ray direction as orientation of source
					l_parseResultsPtr->sourceParams[iSrc].tilt=surfaceArray[iSurf].details.tilt1;
					
					l_parseResultsPtr->geometryParams[iGeom].type=GEOM_PLANESURF;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=MT_PATHTRACESRC;			
					l_parseResultsPtr->geometryParams[iGeom].materialParams.importanceConeAlphaMin=surfaceArray[iSurf].details.importanceConeAlphaMin;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.importanceConeAlphaMax=surfaceArray[iSurf].details.importanceConeAlphaMax;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.flux=surfaceArray[iSurf].details.power;

					// increment geometry counter
					iGeom=iGeom+1;
					// increment source counter
					iSrc=iSrc+1;
					break;								

				case OT_STANDARD:
					if ( (surfaceArray[iSurf].radius1.x==0) && (surfaceArray[iSurf].radius1.y==0) )
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_PLANESURF;
					}
					else
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_SPHERICALSURF;
					}
					if ( surfaceArray[iSurf].details.detector )
					{
						l_parseResultsPtr->detectorParams[iDet].detPixel=surfaceArray[iSurf].details.detPixel;
						l_parseResultsPtr->detectorParams[iDet].detPixel_PhaseSpace=surfaceArray[iSurf].details.detPixel_PhaseSpace;
						l_parseResultsPtr->detectorParams[iDet].detectorType=surfaceArray[iSurf].details.detectorType;
						l_parseResultsPtr->detectorParams[iDet].nrRaysPerPixel=surfaceArray[iSurf].details.nrRaysPerPixel;
						l_parseResultsPtr->detectorParams[iDet].rayDirDistr=surfaceArray[iSurf].details.rayDirDistr;
						l_parseResultsPtr->detectorParams[iDet].rayPosDistr=surfaceArray[iSurf].details.rayPosDistr;
						l_parseResultsPtr->detectorParams[iDet].root=surfaceArray[iSurf].details.root;
						//l_parseResultsPtr->detectorParams[iDet].rotNormal=surfaceArray[iSurf].details.tilt1.z;
						l_parseResultsPtr->detectorParams[iDet].apertureHalfWidth=surfaceArray[iSurf].details.apertureHalfWidth1;
						l_parseResultsPtr->detectorParams[iDet].normal=make_double3(0.0,0.0,1.0);
						rotateRay(&l_parseResultsPtr->detectorParams[iDet].normal,surfaceArray[iSurf].details.tilt1);
						l_parseResultsPtr->detectorParams[iDet].geomID=iGeom+1;
						l_parseResultsPtr->detectorParams[iDet].tilt=surfaceArray[iSurf].details.tilt1;
						l_parseResultsPtr->detectorParams[iDet].importanceArea=surfaceArray[iSurf].details.importanceArea;
						l_parseResultsPtr->detectorParams[iDet].importanceConeAlphaMax=surfaceArray[iSurf].details.importanceConeAlphaMax;
						l_parseResultsPtr->detectorParams[iDet].importanceConeAlphaMin=surfaceArray[iSurf].details.importanceConeAlphaMin;
						l_parseResultsPtr->detectorParams[iDet].importanceObjNr=surfaceArray[iSurf].details.importanceObjNr;
						l_parseResultsPtr->detectorParams[iDet].dirHalfWidth=surfaceArray[iSurf].details.dirHalfWidth;
						iDet=iDet+1;
					}						
					// increment geometry counter
					iGeom=iGeom+1;
					break;

				case OT_DGRATING:
					if ( (surfaceArray[iSurf].radius1.x==0) && (surfaceArray[iSurf].radius1.y==0) )
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_PLANESURF;
					}
					else
					{
						// not implemented yet
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_SPHERICALSURF;
					}
					l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=MT_LINGRAT1D;
					//l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingType=CT_LINGRAT1D;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType=surfaceArray[iSurf].details.scatterFront;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.gratingConstant=surfaceArray[iSurf].details.lines;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.gratingOrdersFromFile=surfaceArray[iSurf].details.gratingOrdersFromFile;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.gratingLinesFromFile=surfaceArray[iSurf].details.gratingLinesFromFile;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.gratingEffsFromFile=surfaceArray[iSurf].details.gratingEffsFromFile;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.nrDiffOrders=surfaceArray[iSurf].details.nrDiffOrders;
					memcpy ( &(l_parseResultsPtr->geometryParams[iGeom].materialParams.diffOrder[0]), &(surfaceArray[iSurf].details.diffOrder[0]), MAX_NR_DIFFORDERS*sizeof(short) );
					memcpy ( &(l_parseResultsPtr->geometryParams[iGeom].materialParams.diffEff[0]), &(surfaceArray[iSurf].details.diffEff[0]), MAX_NR_DIFFORDERS*sizeof(double) );
					// rotate diffraction axis
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingAxis=make_double3(0,1,0);
					rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingAxis),surfaceArray[iSurf].details.tilt1);
					// increment geometry counter
					iGeom=iGeom+1;
					// in nonsequential mode we also have to consider back and the side of the grating
					if (l_mode==SIM_GEOMRAYS_NONSEQ)
					{
						/* copy the geometry params of the second surface*/
						l_parseResultsPtr->geometryParams[iGeom].normal=make_double3(0.0,0.0,1.0);
						rotateRay(&l_parseResultsPtr->geometryParams[iGeom].normal,surfaceArray[iSurf].details.tilt1);
						l_parseResultsPtr->geometryParams[iGeom].tilt=surfaceArray[iSurf].details.tilt2;
						//l_parseResultsPtr->geometryParams[iGeom].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
						l_parseResultsPtr->geometryParams[iGeom].root=surfaceArray[iSurf].details.root+surfaceArray[iSurf].thickness*l_parseResultsPtr->geometryParams[iGeom-1].normal;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_r=surfaceArray[iSurf].details.coating_back_r;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_t=surfaceArray[iSurf].details.coating_back_t;
						
						if ( (surfaceArray[iSurf].radius2.x==0) && (surfaceArray[iSurf].radius2.y==0) )
						{
							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_PLANESURF;
						}
						else
						{
							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_SPHERICALSURF;
						}
						l_parseResultsPtr->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
						l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth2;
						l_parseResultsPtr->geometryParams[iGeom].radius1=surfaceArray[iSurf].radius2;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=surfaceArray[iSurf].glass;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingType=surfaceArray[iSurf].details.coatingBack;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType=surfaceArray[iSurf].details.scatterBack;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
						rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis),surfaceArray[iSurf].details.tilt1);
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, 8*sizeof(double) );
						l_parseResultsPtr->geometryParams[iGeom].conic1=surfaceArray[iSurf].conic2;
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );					
						
						// increment geometry counter
						iGeom=iGeom+1;
						
						/* copy the geometry params of the cylinder pipe */
						l_parseResultsPtr->geometryParams[iGeom].normal=make_double3(0.0,0.0,1.0);
						rotateRay(&l_parseResultsPtr->geometryParams[iGeom].normal,surfaceArray[iSurf].details.tilt1);
						l_parseResultsPtr->geometryParams[iGeom].tilt=surfaceArray[iSurf].details.tilt1;
						//l_parseResultsPtr->geometryParams[iGeom].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
						//l_parseResultsPtr->geometryParams[iGeom].rotNormal2=surfaceArray[iSurf].details.tilt1.z;
						l_parseResultsPtr->geometryParams[iGeom].root=surfaceArray[iSurf].details.root+surfaceArray[iSurf].thickness*l_parseResultsPtr->geometryParams[iGeom-1].normal;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_r=surfaceArray[iSurf].details.coating_side_r;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_t=surfaceArray[iSurf].details.coating_side_t;
						
						// check wether we have a cylinder or a cone
						if ( (surfaceArray[iSurf].details.apertureHalfWidth1.x==surfaceArray[iSurf].details.apertureHalfWidth2.x) && (surfaceArray[iSurf].details.apertureHalfWidth2.y==surfaceArray[iSurf].details.apertureHalfWidth1.y) )
						{
							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CYLPIPE;
						}
						else
						{
							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CONEPIPE;
						}
						l_parseResultsPtr->geometryParams[iGeom].thickness=surfaceArray[iSurf].thickness;
						l_parseResultsPtr->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
						l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth1;
						l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth2=surfaceArray[iSurf].details.apertureHalfWidth2;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=surfaceArray[iSurf].glass;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingType=surfaceArray[iSurf].details.coatingSide;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType=surfaceArray[iSurf].details.scatterSide;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
						rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis),surfaceArray[iSurf].details.tilt1);
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, 8*sizeof(double) );
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );					
						
						// increment geometry counter
						iGeom=iGeom+1;
					}
					break;

				case OT_STLENSE:
					if ( (surfaceArray[iSurf].radius1.x==0) && (surfaceArray[iSurf].radius1.y==0) )
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_PLANESURF;
					}
					else
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_SPHERICALSURF;
					}

					// increment geometry counter
					iGeom=iGeom+1;

					/* copy the geometry params of the second surface*/
					l_parseResultsPtr->geometryParams[iGeom].normal=make_double3(0.0,0.0,1.0);
					rotateRay(&l_parseResultsPtr->geometryParams[iGeom].normal,surfaceArray[iSurf].details.tilt1);
					l_parseResultsPtr->geometryParams[iGeom].tilt=surfaceArray[iSurf].details.tilt2;
					//l_parseResultsPtr->geometryParams[iGeom].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
//					//l_parseResultsPtr->geometryParams[iGeom].rotNormal2=surfaceArray[iSurf].details.tilt1.z;
					l_parseResultsPtr->geometryParams[iGeom].root=surfaceArray[iSurf].details.root+surfaceArray[iSurf].thickness*l_parseResultsPtr->geometryParams[iGeom-1].normal;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_r=surfaceArray[iSurf].details.coating_back_r;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_t=surfaceArray[iSurf].details.coating_back_t;
					
					if ( (surfaceArray[iSurf].radius2.x==0) && (surfaceArray[iSurf].radius2.y==0) )
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_PLANESURF;
					}
					else
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_SPHERICALSURF;
					}
					l_parseResultsPtr->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
					l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth2;
					l_parseResultsPtr->geometryParams[iGeom].radius1=surfaceArray[iSurf].radius2;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=surfaceArray[iSurf].glass;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingType=surfaceArray[iSurf].details.coatingBack;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType=surfaceArray[iSurf].details.scatterBack;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
					rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis),surfaceArray[iSurf].details.tilt1);
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, 8*sizeof(double) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );
					
					l_parseResultsPtr->geometryParams[iGeom].conic1=surfaceArray[iSurf].conic2;
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
					
					// increment geometry counter
					iGeom=iGeom+1;
					// in nonsequential mode we also have to consider the side of the lense
					if (l_mode==SIM_GEOMRAYS_NONSEQ)
					{
						// check wether the thickness is big enough to make up a physical lense with the given apertures and radii
						// calc height of first surface
						
						double maxAprt1= surfaceArray[iSurf].details.apertureHalfWidth1.x;
						if (maxAprt1<surfaceArray[iSurf].details.apertureHalfWidth1.y)
							maxAprt1=surfaceArray[iSurf].details.apertureHalfWidth1.y;
						double maxAprt2= surfaceArray[iSurf].details.apertureHalfWidth2.x;
						if (maxAprt2<surfaceArray[iSurf].details.apertureHalfWidth2.y)
							maxAprt2=surfaceArray[iSurf].details.apertureHalfWidth2.y;
						double h1Sqr=surfaceArray[iSurf].radius1.x*surfaceArray[iSurf].radius1.x-maxAprt1*maxAprt1;
						// a radius of zero actually means a radius of infinity...
						if (surfaceArray[iSurf].radius1.x==0)
							h1Sqr=DOUBLE_MAX;
						double h2Sqr=surfaceArray[iSurf].radius2.x*surfaceArray[iSurf].radius2.x-maxAprt2*maxAprt2;
						if (surfaceArray[iSurf].radius2.x==0)
							h2Sqr=DOUBLE_MAX;

						if ( (h1Sqr<0)||(h2Sqr<0) )
						{
							std::cout << "error in parseZemaxPrescr at object " << (surfIndex+1) << ": radii of curvature are too small for given aperture" << std::endl;
							return PARSER_ERR;
						}
						double test=-surfaceArray[iSurf].radius2.x-sqrt(h2Sqr);
						
						double deltaH1=surfaceArray[iSurf].radius1.x-sqrt(h1Sqr);
						if (deltaH1 < 0)
							deltaH1=0;
						double deltaH2=-surfaceArray[iSurf].radius2.x-sqrt(h2Sqr);
						if (deltaH2 < 0)
							deltaH2=0;
						if (surfaceArray[iSurf].thickness < deltaH1+deltaH2)
						{
							std::cout << "error in parseZemaxPrescr at object " << (surfIndex+1) << ": thickness of lense is too low for given radii of curvature" << std::endl;
							return PARSER_ERR;
						}
					
						/* copy the geometry params of the cylinder pipe */
						l_parseResultsPtr->geometryParams[iGeom].normal=make_double3(0.0,0.0,1.0);
						rotateRay(&l_parseResultsPtr->geometryParams[iGeom].normal,surfaceArray[iSurf].details.tilt1);
						l_parseResultsPtr->geometryParams[iGeom].tilt=surfaceArray[iSurf].details.tilt1;
						//l_parseResultsPtr->geometryParams[iGeom].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
						//l_parseResultsPtr->geometryParams[iGeom].rotNormal2=surfaceArray[iSurf].details.tilt1.z;
						l_parseResultsPtr->geometryParams[iGeom].root=surfaceArray[iSurf].details.root+surfaceArray[iSurf].thickness*l_parseResultsPtr->geometryParams[iGeom-1].normal;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_r=surfaceArray[iSurf].details.coating_side_r;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_t=surfaceArray[iSurf].details.coating_side_t;
				
						// check wether we have a cylinder or a cone
						if ( (surfaceArray[iSurf].details.apertureHalfWidth1.x==surfaceArray[iSurf].details.apertureHalfWidth2.x) && (surfaceArray[iSurf].details.apertureHalfWidth2.y==surfaceArray[iSurf].details.apertureHalfWidth1.y) )
						{
							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CYLPIPE;
						}
						else
						{
							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CONEPIPE;
						}
						l_parseResultsPtr->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
						l_parseResultsPtr->geometryParams[iGeom].thickness=surfaceArray[iSurf].thickness-(sqrt(h1Sqr)+sqrt(h1Sqr));
						l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth1;
						l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth2=surfaceArray[iSurf].details.apertureHalfWidth2;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=surfaceArray[iSurf].glass;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingType=surfaceArray[iSurf].details.coatingSide;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType=surfaceArray[iSurf].details.scatterSide;
						l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
						l_parseResultsPtr->geometryParams[iGeom].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
						rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis),surfaceArray[iSurf].details.tilt1);
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, 8*sizeof(double) );
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );						
						// increment geometry counter
						iGeom=iGeom+1;
					}
					
					break;					

				case OT_BICONLENSE:
					// the front surface of a biconic lense can be made of a plane surface, a spherical surface, a cylindrical surface or a real biconic surface
					if ( (surfaceArray[iSurf].radius1.x==0) && (surfaceArray[iSurf].radius1.y==0) )
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_PLANESURF;
					else
					{
						if ( surfaceArray[iSurf].radius1.x==surfaceArray[iSurf].radius1.y )
							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_SPHERICALSURF;
						else
						{
							if ( (surfaceArray[iSurf].radius1.x==0) || (surfaceArray[iSurf].radius1.y==0) )
							{
								l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CYLLENSESURF;
								// our CylLense surface always has the radius along one direction, i.e. x. If we have the radius along y direction, we rotate the surface 90° around the z-axis and exchange the aperture half widths
								if ( surfaceArray[iSurf].radius1.x==0 )
								{
									l_parseResultsPtr->geometryParams[iGeom].radius1.x=surfaceArray[iSurf].radius1.y;
									l_parseResultsPtr->geometryParams[iGeom].radius1.y=surfaceArray[iSurf].radius1.x;
									l_parseResultsPtr->geometryParams[iGeom].tilt.z=l_parseResultsPtr->geometryParams[iGeom].tilt.z+PI/2;
									l_parseResultsPtr->geometryParams[iGeom].tilt.x=-surfaceArray[iSurf].details.tilt1.y;
									l_parseResultsPtr->geometryParams[iGeom].tilt.y=-surfaceArray[iSurf].details.tilt1.x;
								}
							}	
							else
							{
								std::cout << "error in parseZemaxPrescr: biconic lense with two different curvature radii that are not equal to zero is not implemented yet..." << std::endl;
								fclose(yyout);
								free(surfaceArray);
								
								return PARSER_ERR;
							}
							
						}
					}

					// increment geometry counter
					iGeom=iGeom+1;

					/* copy the geometry params of the second surface*/
					l_parseResultsPtr->geometryParams[iGeom].normal=make_double3(0.0,0.0,1.0);
					rotateRay(&l_parseResultsPtr->geometryParams[iGeom].normal,surfaceArray[iSurf].details.tilt1);
					l_parseResultsPtr->geometryParams[iGeom].tilt=surfaceArray[iSurf].details.tilt2;
					//l_parseResultsPtr->geometryParams[iGeom].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
//					//l_parseResultsPtr->geometryParams[iGeom].rotNormal2=surfaceArray[iSurf].details.tilt1.z;
					l_parseResultsPtr->geometryParams[iGeom].root=surfaceArray[iSurf].details.root+surfaceArray[iSurf].thickness*l_parseResultsPtr->geometryParams[iGeom-1].normal;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_r=surfaceArray[iSurf].details.coating_back_r;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_t=surfaceArray[iSurf].details.coating_back_t;
					
					l_parseResultsPtr->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
					l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth2;
					l_parseResultsPtr->geometryParams[iGeom].radius1=surfaceArray[iSurf].radius2;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=surfaceArray[iSurf].glass;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingType=surfaceArray[iSurf].details.coatingBack;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType=surfaceArray[iSurf].details.scatterBack;
					l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
					rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis),surfaceArray[iSurf].details.tilt1);
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, 8*sizeof(double) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );
					
					l_parseResultsPtr->geometryParams[iGeom].conic1=surfaceArray[iSurf].conic2;
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );

					// the front surface of a biconic lense can be made of a plane surface, a spherical surface, a cylindrical surface or a real biconic surface
					if ( (surfaceArray[iSurf].radius2.x==0) && (surfaceArray[iSurf].radius2.y==0) )
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_PLANESURF;
					else
					{
						if ( surfaceArray[iSurf].radius2.x==surfaceArray[iSurf].radius2.y )
							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_SPHERICALSURF;
						else
						{
							if ( (surfaceArray[iSurf].radius2.x==0) || (surfaceArray[iSurf].radius2.y==0) )
							{
								l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CYLLENSESURF;
								// our CylLense surface always has the radius along one direction, i.e. x. If we have the radius along y direction, we rotate the surface 90° around the z-axis and exchange the aperture half widths
								if ( surfaceArray[iSurf].radius2.x==0 )
								{
									l_parseResultsPtr->geometryParams[iGeom].radius1.x=surfaceArray[iSurf].radius2.y;
									l_parseResultsPtr->geometryParams[iGeom].radius1.y=surfaceArray[iSurf].radius2.x;
									l_parseResultsPtr->geometryParams[iGeom].tilt.z=l_parseResultsPtr->geometryParams[iGeom].tilt.z+PI/2;
									l_parseResultsPtr->geometryParams[iGeom].tilt.x=-surfaceArray[iSurf].details.tilt2.y;
									l_parseResultsPtr->geometryParams[iGeom].tilt.y=-surfaceArray[iSurf].details.tilt2.x;
								}
							}
							else
							{
								std::cout << "error in parseZemaxPrescr: biconic lense with two different curvature radii that are not equal to zero is not implemented yet..." << std::endl;
								fclose(yyout);
								free(surfaceArray);
								
								return PARSER_ERR;
							}
							
						}
					}
					
					// increment geometry counter
					iGeom=iGeom+1;
					// in nonsequential mode we also have to consider the side of the lense
//					if (l_mode==SIM_GEOMRAYS_NONSEQ)
//					{
//						/* copy the geometry params of the cylinder pipe */
//						l_parseResultsPtr->geometryParams[iGeom].normal=make_double3(0.0,0.0,1.0);
//						rotateRay(&l_parseResultsPtr->geometryParams[iGeom].normal,surfaceArray[iSurf].details.tilt1);
//						l_parseResultsPtr->geometryParams[iGeom].tilt=surfaceArray[iSurf].details.tilt1;
//						l_parseResultsPtr->geometryParams[iGeom].rotNormal1=surfaceArray[iSurf].details.tilt1.z;
//						l_parseResultsPtr->geometryParams[iGeom].rotNormal2=surfaceArray[iSurf].details.tilt1.z;
//						l_parseResultsPtr->geometryParams[iGeom].root=surfaceArray[iSurf].details.root+surfaceArray[iSurf].thickness*l_parseResultsPtr->geometryParams[iGeom-1].normal;
//						l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_r=surfaceArray[iSurf].details.coating_side_r;
//						l_parseResultsPtr->geometryParams[iGeom].materialParams.coating_t=surfaceArray[iSurf].details.coating_side_t;
//				
//						// check wether we have a cylinder or a cone
//						if ( (surfaceArray[iSurf].details.apertureHalfWidth1.x==surfaceArray[iSurf].details.apertureHalfWidth2.x) && (surfaceArray[iSurf].details.apertureHalfWidth2.y==surfaceArray[iSurf].details.apertureHalfWidth1.y) )
//						{
//							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CYLPIPE;
//						}
//						else
//						{
//							l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CONEPIPE;
//						}
//						l_parseResultsPtr->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
//						l_parseResultsPtr->geometryParams[iGeom].thickness=surfaceArray[iSurf].thickness;
//						l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth1=surfaceArray[iSurf].details.apertureHalfWidth1;
//						l_parseResultsPtr->geometryParams[iGeom].apertureHalfWidth2=surfaceArray[iSurf].details.apertureHalfWidth2;
//						l_parseResultsPtr->geometryParams[iGeom].materialParams.matType=surfaceArray[iSurf].glass;
//						l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingType=surfaceArray[iSurf].details.coatingSide;
//						l_parseResultsPtr->geometryParams[iGeom].materialParams.scatterType=surfaceArray[iSurf].details.scatterSide;
//						l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
//						l_parseResultsPtr->geometryParams[iGeom].materialParams.nRefr=surfaceArray[iSurf].details.nRefr;
//						rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.scatteringAxis),surfaceArray[iSurf].details.tilt1);
//						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, 8*sizeof(double) );
//						memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
//						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.glassName, surfaceArray[iSurf].glassName, GEOM_CMT_LENGTH*sizeof(char) );
//						memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.immersionName, surfaceArray[iSurf].details.immersionName, GEOM_CMT_LENGTH*sizeof(char) );						
						
						// increment geometry counter
//						iGeom=iGeom+1;
//					}
					break;				

				case OT_CYLPIPE:					
					if (surfaceArray[iSurf].details.apertureHalfWidth1 == surfaceArray[iSurf].details.apertureHalfWidth2)
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CYLPIPE;
					}
					else
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_CONEPIPE;
					}
							
					// increment geometry counter
					iGeom=iGeom+1;
					break;			

				case OT_ASPHSURF:					
					if (surfaceArray[iSurf].details.apertureHalfWidth1 == surfaceArray[iSurf].details.apertureHalfWidth2)
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_ASPHERICALSURF;
					}
					else
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_ASPHERICALSURF;
					}
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].asphereParams, surfaceArray[iSurf].details.asphereParams, MAX_NR_ASPHPAR*sizeof(double) );
					// rotate scattering axis 
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
					rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingAxis),surfaceArray[iSurf].details.tilt1);
							
					// increment geometry counter
					iGeom=iGeom+1;
					break;					

				case OT_APTSTOP:					
					if ( (surfaceArray[iSurf].radius1.x==0) && (surfaceArray[iSurf].radius1.y==0) )
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_APTSTOP;
					}
					else
					{
						std::cout << "error in parseZemaxDescr(): aperture stops on non planar surfaces not implemented yet. error at surface :" << iSurf << std::endl;
						iGeom=iGeom+1;
						return PARSER_ERR;
					}
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.params, 8*sizeof(double) );
					// rotate scattering axis 
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
					rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingAxis),surfaceArray[iSurf].details.tilt1);
					// copy material params
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, 8*sizeof(double) );
							
					// increment geometry counter
					iGeom=iGeom+1;
					break;	
					
				case OT_COSINENORMAL:					
					if ( (surfaceArray[iSurf].radius1.x==0) && (surfaceArray[iSurf].radius1.y==0) )
					{
						l_parseResultsPtr->geometryParams[iGeom].type=GEOM_COSINENORMAL;
					}
					else
					{
						std::cout << "error in parseZemaxDescr(): cosine normals on non planar surfaces not implemented yet. error at surface :" << iSurf << std::endl;
						iGeom=iGeom+1;
						return PARSER_ERR;
					}
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
					memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.params, 8*sizeof(double) );
					// rotate scattering axis 
					l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingAxis=make_double3(surfaceArray[iSurf].details.matParams[5],surfaceArray[iSurf].details.matParams[6],surfaceArray[iSurf].details.matParams[7]);
					rotateRay(&(l_parseResultsPtr->geometryParams[iGeom].materialParams.coatingAxis),surfaceArray[iSurf].details.tilt1);
					l_parseResultsPtr->geometryParams[iGeom].cosNormAxis=surfaceArray[iSurf].details.cosNormAxis;
					rotateRay(&l_parseResultsPtr->geometryParams[iGeom].cosNormAxis,surfaceArray[iSurf].details.tilt1);
					l_parseResultsPtr->geometryParams[iGeom].cosNormAmpl=surfaceArray[iSurf].details.cosNormAmpl;
					l_parseResultsPtr->geometryParams[iGeom].cosNormPeriod=surfaceArray[iSurf].details.cosNormPeriod;
					l_parseResultsPtr->geometryParams[iGeom].iterationAccuracy=surfaceArray[iSurf].details.iterationAccuracy;;
							
					// increment geometry counter
					iGeom=iGeom+1;
					break;		

				default:
					// copy material params
					//memcpy ( l_parseResultsPtr->geometryParams[iGeom].materialParams.varParams, surfaceArray[iSurf].details.matParams, 8*sizeof(double) );
					std::cout << "error in parseZemaxDescr(): unknown surface type at surface:" << iSurf << std::endl;
					iGeom=iGeom+1;
					return PARSER_ERR;
					// implement some error mechanism
					break;

			} // end switch geometry type

		} // end if (iGeom<=geomIndex)
		else
		{
			std::cout << "error in parseZemaxDescr(): number of geometries that are created from prescription exceeds expected value" << std::endl;
			return PARSER_ERR;
			// implement some error formalism
		}
		
	} // end for
	fclose(yyout);
	free(surfaceArray);
	*parseResultsGeometry=l_parseResultsPtr;

	return PARSER_NO_ERR;
}	
	
parserError initSurfaceStruct(ZemaxParamStruct* surface)
{
	surface->type=OT_UNKNOWNOTYPE;
	sprintf(surface->comment,"");
	surface->radius1.x=0;
	surface->radius1.y=0;
	surface->radius2.x=0;
	surface->radius2.y=0;
	surface->thickness=0;
	surface->glass=MT_UNKNOWNMATERIAL;
	surface->diameter=0;
	surface->conic1=0;
	surface->conic2=0;
	sprintf(surface->comment, "%s", "NO COMMENT");
	surface->details.dec1=make_double3(0,0,0);
	surface->details.dec2=make_double3(0,0,0);
	surface->details.tilt1=make_double3(0,0,0);
	surface->details.tilt2=make_double3(0,0,0);
	surface->details.order1=TILTDEC;
	surface->details.order2=TILTDEC;
	surface->details.aperture=AT_UNKNOWNATYPE;
	surface->details.apertureHalfWidth1=make_double2(0,0);
	surface->details.apertureHalfWidth2=make_double2(0,0);
	surface->details.obscurationHalfWidth=make_double2(0,0);
	surface->details.scatterFront=ST_NOSCATTER;
	surface->details.scatterBack=ST_NOSCATTER;
	surface->details.scatterSide=ST_NOSCATTER;
	surface->details.coatingFront=CT_NOCOATING;
	surface->details.coatingBack=CT_NOCOATING;
	surface->details.coatingSide=CT_NOCOATING;
	surface->details.coating_side_r=0;
	surface->details.coating_side_t=0;
	surface->details.coating_front_r=0;
	surface->details.coating_front_t=0;
	surface->details.coating_back_r=0;
	surface->details.coating_back_t=0;
	surface->details.gratingOrdersFromFile=false;
	surface->details.gratingLinesFromFile=false;
	surface->details.gratingEffsFromFile=false;
	surface->details.nrDiffOrders=0;
	surface->details.nRefr=make_double2(0,0);
	surface->details.rayDirectionTilt=make_double2(0,0);
	surface->details.rayDirDistr=RAYDIR_UNKNOWN;
	surface->details.rayPosDistr=RAYPOS_UNKNOWN;
	surface->details.power=0;
	surface->details.rayFieldWidth=0;
	surface->details.rayFieldHeight=0;
	surface->details.lambda=0;
	surface->details.alphaMin=make_double2(0,0);
	surface->details.alphaMax=make_double2(0,0);
	sprintf(surface->details.immersionName, "%s", "STANDARD");
	surface->details.detector=false;
	surface->details.detPixel=make_ulong2(0,0);
	surface->details.detectorType=DET_UNKNOWN;
	surface->details.idealLense_A=0;
	surface->details.idealLense_f0=0;
	surface->details.idealLense_lambda0=0;
	surface->details.srcCoherence=0;
	surface->details.importanceObjNr=-1;
	surface->details.importanceConeAlphaMax=make_double2(0,0);
	surface->details.importanceConeAlphaMin=make_double2(0,0);
	surface->details.nrRayDirections=make_ulong2(1,1);
	surface->details.importanceArea=false;
	surface->details.rayFieldHeightLayout=0;
	surface->details.rayFieldWidthLayout=0;
	surface->details.sweep=false;
	surface->details.coating_back_c_t=0;
	surface->details.coating_back_c_r=0;
	surface->details.coating_front_c_t=0;
	surface->details.coating_front_c_r=0;
	surface->details.coating_side_c_t=0;
	surface->details.coating_side_c_r=0;
	surface->details.coating_side_a_t=0;
	surface->details.coating_side_a_r=0;
	surface->details.coating_front_a_t=0;
	surface->details.coating_front_a_r=0;
	surface->details.coating_back_a_r=0;
	surface->details.coating_back_a_t=0;
	surface->details.coating_side_a_t=0;
	surface->details.coating_side_a_r=0;
	surface->details.dirHalfWidth=make_double2(0,0);
	surface->details.iterationAccuracy=DOUBLE_MAX;
	for (short i=0;i<MAX_NR_MATPARAMS;i++)
	{
		surface->details.matParams[i]=0;
	}
	for (short i=0;i<MAX_NR_ASPHPAR;i++)
	{
		surface->details.asphereParams[i]=0;
	}
	for (short i=0;i<MAX_NR_DIFFORDERS;i++)
	{
		surface->details.diffOrder[i]=0;
		surface->details.diffEff[i]=0;
	}
	for (short i=0;i<MAX_NR_MATPARAMS;i++)
	{
		surface->details.params[i]=0;
		surface->details.matParams[i]=0;
	}
	
	return PARSER_NO_ERR;
}

void initGeomParseFlags(void)
{
	geomParseFlag_ObjName=false;
	geomParseFlag_ObjType=false;
	geomParseFlag_DetPixX=false;
	geomParseFlag_DetPixY=false;
	geomParseFlag_DetType=false;
	geomParseFlag_SrcRayNrHeight=false;
	geomParseFlag_SrcRayNrWidth=false;
	geomParseFlag_SrcPower=false;
	geomParseFlag_SrcDirDist=false;
	geomParseFlag_SrcPosDist=false;
	geomParseFlag_SrcLambda=false;
	geomParseFlag_RefObj=false;
	geomParseFlag_Tilt=false;
	geomParseFlag_PosMtrxZ=false;
	geomParseFlag_PosMtrxY=false;
	geomParseFlag_PosMtrxX=false;
	geomParseFlag_Mat=false;
	geomParseFlag_Immerse=false;
	geomParseFlag_Face=false;
	geomParseFlag_Scatter=false;
	geomParseFlag_Coating=false;
	geomParseFlag_Rad1=false;
	geomParseFlag_Rad2=false;
	geomParseFlag_AsphCoeff=false;
	geomParseFlag_Con2=false;
	geomParseFlag_Con1=false;
    geomParseFlag_Apert1X=false;
    geomParseFlag_Apert1Y=false;
    geomParseFlag_Apert2X=false;
    geomParseFlag_Apert2Y=false;
    geomParseFlag_ApertType=false;
    geomParseFlag_IdealLenseLambda0=false;
    geomParseFlag_IdealLenseF0=false;
    geomParseFlag_IdealLenseA=false;
    geomParseFlag_DiffLines=false;
    geomParseFlag_DiffOrder=false;
    geomParseFlag_DiffEffs=false;
    geomParseFlag_Thickness=false;
    geomParseFlag_SrcCoherence=false;
    geomParseFlag_MinApert1X=false;
    geomParseFlag_MinApert1Y=false;	
    geomParseFlag_CosNormPeriod=false;
    geomParseFlag_CosNormAmpl=false;
    geomParseFlag_CosNormAxis=false;
    geomParseFlag_FilterMax=false;
    geomParseFlag_FilterMin=false;
    geomParseFlag_NrRaysPerPix=false;
}


bool checkObjectDefinition(ZemaxParamStruct *objectPtr)
{
	if (!geomParseFlag_ObjType)
		return false;
	switch (objectPtr->type)
	{
		case OT_DET:
			if ( !( geomParseFlag_Tilt && geomParseFlag_PosMtrxZ && geomParseFlag_Mat
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_NrRaysPerPix ) )
				{
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;
					if( (!geomParseFlag_NrRaysPerPix) )
						std::cout << "error in checkObjectDefinition(): definition of number of rays per pixel is missing" << std::endl;
						
					return false;
				}
				break;
				
		case OT_ILLPUPIL:
			if ( !( geomParseFlag_Tilt && geomParseFlag_PosMtrxZ 
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					 ) )
				{
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;
						
					return false;
				}
				break;				
		case OT_ASPHSURF:
			if ( !( geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ && geomParseFlag_Mat
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y) )
				{
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;
						
					return false;
				}	
				break;				
			
		case OT_APTSTOP:
		case OT_STANDARD:
			if ( !( geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ && geomParseFlag_Mat
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y) )
				{
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;
						
					return false;
				}
			if (objectPtr->details.detector)
			{
				if ( !(geomParseFlag_DetPixX && geomParseFlag_DetPixY && geomParseFlag_DetType) )
				{
					if( (!geomParseFlag_DetPixX) ||(!geomParseFlag_DetPixY))
						std::cout << "error in checkObjectDefinition(): detector pixel is not defined correctly" << std::endl;
					if( (!geomParseFlag_DetType) )
						std::cout << "error in checkObjectDefinition(): detector type is not defined correctly" << std::endl;
				
					return false;
				}
			}
			break;
//		case OT_APTSTOP:
//			if ( !( geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ && geomParseFlag_Mat
//					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y) )
//				{
//					return false;
//				}
//			break;
			
		case OT_DGRATING:
			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_Apert2X && geomParseFlag_Apert2Y && geomParseFlag_Mat
					&& geomParseFlag_DiffLines && geomParseFlag_DiffOrder && geomParseFlag_DiffEffs && geomParseFlag_Thickness) )
				{				
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture1 is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert2X) ||(!geomParseFlag_Apert2Y))
						std::cout << "error in checkObjectDefinition(): aperture2 is not defined correctly" << std::endl;
					if( (!geomParseFlag_DiffLines) )
						std::cout << "error in checkObjectDefinition(): diffraction lines are not defined correctly" << std::endl;
					if( (!geomParseFlag_DiffOrder) )
						std::cout << "error in checkObjectDefinition(): diffraction orders are not defined correctly" << std::endl;
					if( (!geomParseFlag_DiffEffs) )
						std::cout << "error in checkObjectDefinition(): diffraction efficiencies are not defined correctly" << std::endl;
					if( (!geomParseFlag_Thickness) )
						std::cout << "error in checkObjectDefinition(): thickness is not defined correctly" << std::endl;
				
					return false;
				}
			break;	
		case OT_SRC:
			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_SrcRayNrHeight && geomParseFlag_SrcRayNrWidth && geomParseFlag_SrcPower
					&& geomParseFlag_SrcDirDist && geomParseFlag_SrcPosDist && geomParseFlag_SrcLambda) )
				{
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcRayNrHeight) || (!geomParseFlag_SrcRayNrWidth))
						std::cout << "error in checkObjectDefinition(): ray number is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcPower) )
						std::cout << "error in checkObjectDefinition(): source power is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcDirDist) )
						std::cout << "error in checkObjectDefinition(): direction distribution is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcPosDist) )
						std::cout << "error in checkObjectDefinition(): position distribution is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcLambda) )
						std::cout << "error in checkObjectDefinition(): source wavelength is not defined correctly" << std::endl;
				
					return false;
				}
			break;
		case OT_DIFFSRC_FREEFORM:
		case OT_DIFFSRC_HOLO:
		case OT_DIFFSRC_RAYAIM:
		case OT_DIFFSRC:
			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ && geomParseFlag_SrcDiffNr
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_SrcRayNrHeight && geomParseFlag_SrcRayNrWidth && geomParseFlag_SrcPower
					&& geomParseFlag_SrcDirDist && geomParseFlag_SrcPosDist && geomParseFlag_SrcLambda) )
				{
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcRayNrHeight) || (!geomParseFlag_SrcRayNrWidth))
						std::cout << "error in checkObjectDefinition(): ray number is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcPower) )
						std::cout << "error in checkObjectDefinition(): source power is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcDirDist) )
						std::cout << "error in checkObjectDefinition(): direction distribution is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcPosDist) )
						std::cout << "error in checkObjectDefinition(): position distribution is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcLambda) )
						std::cout << "error in checkObjectDefinition(): source wavelength is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcDiffNr) )
						std::cout << "error in checkObjectDefinition(): differential ray number is not defined correctly" << std::endl;
				
					return false;
				}
			break;
			
		case OT_PATHTRACINGSRC:
			if ( !(geomParseFlag_Tilt && geomParseFlag_PosMtrxZ 
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_SrcRayNrHeight && geomParseFlag_SrcRayNrWidth && geomParseFlag_SrcPower
					&& geomParseFlag_SrcDirDist && geomParseFlag_SrcPosDist && geomParseFlag_SrcLambda) )
				{
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcRayNrHeight) || (!geomParseFlag_SrcRayNrWidth))
						std::cout << "error in checkObjectDefinition(): ray number is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcPower) )
						std::cout << "error in checkObjectDefinition(): source power is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcDirDist) )
						std::cout << "error in checkObjectDefinition(): direction distribution is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcPosDist) )
						std::cout << "error in checkObjectDefinition(): position distribution is not defined correctly" << std::endl;
					if( (!geomParseFlag_SrcLambda) )
						std::cout << "error in checkObjectDefinition(): source wavelength is not defined correctly" << std::endl;
				
					return false;
				}
				break;			
		
		case OT_IDEALLENSE:
			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_IdealLenseLambda0 && geomParseFlag_IdealLenseF0 && geomParseFlag_IdealLenseA) )
				{
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;				
					if( (!geomParseFlag_IdealLenseLambda0) )
						std::cout << "error in checkObjectDefinition(): centre wavelength of ideal lense is not defined correctly" << std::endl;
					if( (!geomParseFlag_IdealLenseF0) )
						std::cout << "error in checkObjectDefinition(): centre focal length of ideal lense is not defined correctly" << std::endl;
					if( (!geomParseFlag_IdealLenseA) )
						std::cout << "error in checkObjectDefinition(): chromaticity constant of focal length of ideal lense is not defined correctly" << std::endl;
				
					return false;
				}
			break;
		case OT_BICONLENSE:
		case OT_STLENSE:
			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_Apert2X && geomParseFlag_Apert2Y && geomParseFlag_Thickness && geomParseFlag_Mat
					&& geomParseFlag_Rad1 && geomParseFlag_Rad2 ) )
				{
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture is not defined correctly" << std::endl;				
					if( (!geomParseFlag_Apert2X) ||(!geomParseFlag_Apert2Y))
						std::cout << "error in checkObjectDefinition(): aperture2 is not defined correctly" << std::endl;
					if( (!geomParseFlag_Rad1) )
						std::cout << "error in checkObjectDefinition(): radius is not defined correctly" << std::endl;
					if( (!geomParseFlag_Rad2) )
						std::cout << "error in checkObjectDefinition(): radius is not defined correctly" << std::endl;
						
					return false;
				}
			break;
//		case OT_BICONLENSE:
//			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ
//					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
//					&& geomParseFlag_Apert2X && geomParseFlag_Apert2Y && geomParseFlag_Thickness && geomParseFlag_Mat
//					&& geomParseFlag_Rad1 && geomParseFlag_Rad2 ) )
//				{
//					return false;
//				}
//			break;

		case OT_CONEPIPE:
		case OT_CYLPIPE:
			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_Apert2X && geomParseFlag_Apert2Y && geomParseFlag_Thickness && geomParseFlag_Mat) )
				{
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y))
						std::cout << "error in checkObjectDefinition(): aperture1 is not defined correctly" << std::endl;				
					if( (!geomParseFlag_Apert2X) ||(!geomParseFlag_Apert2Y))
						std::cout << "error in checkObjectDefinition(): aperture2 is not defined correctly" << std::endl;				
					if( (!geomParseFlag_Thickness) )
						std::cout << "error in checkObjectDefinition(): thickness is not defined correctly" << std::endl;				

					return false;
				}
			break;
//		case OT_CONEPIPE:
//			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ
//					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
//					&& geomParseFlag_Apert2X && geomParseFlag_Apert2Y && geomParseFlag_Thickness && geomParseFlag_Mat) )
//				{
//					return false;
//				}
//			break;

		case OT_COSINENORMAL:
			if ( !(geomParseFlag_RefObj && geomParseFlag_Tilt && geomParseFlag_PosMtrxZ
					&& geomParseFlag_PosMtrxY && geomParseFlag_PosMtrxX && geomParseFlag_Apert1X && geomParseFlag_Apert1Y
					&& geomParseFlag_Mat && geomParseFlag_CosNormPeriod && geomParseFlag_CosNormAmpl && geomParseFlag_CosNormAxis) )
				{
					if(!geomParseFlag_RefObj)
						std::cout << "error in checkObjectDefinition(): reference object is not defined correctly" << std::endl;
					if(!geomParseFlag_Tilt)
						std::cout << "error in checkObjectDefinition(): tilt is not defined correctly" << std::endl;
					if( (!geomParseFlag_PosMtrxZ) || (!geomParseFlag_PosMtrxY) || (!geomParseFlag_PosMtrxX) )
						std::cout << "error in checkObjectDefinition(): position is not defined correctly" << std::endl;
					if(!geomParseFlag_Mat)
						std::cout << "error in checkObjectDefinition(): material is not defined correctly" << std::endl;
					if( (!geomParseFlag_Apert1X) ||(!geomParseFlag_Apert1Y) )
						std::cout << "error in checkObjectDefinition(): aperture1 is not defined correctly" << std::endl;				
					if( (!geomParseFlag_CosNormPeriod) )
						std::cout << "error in checkObjectDefinition(): period is not defined correctly" << std::endl;				
					if( (!geomParseFlag_CosNormAmpl) )
						std::cout << "error in checkObjectDefinition(): amplitude is not defined correctly" << std::endl;				
					if( (!geomParseFlag_CosNormAxis) )
						std::cout << "error in checkObjectDefinition(): axis is not defined correctly" << std::endl;				
				
					return false;
				}
			break;			
			
		default:
			std::cout << "error in checkObjectDefinition(): unknown geometry type" << std::endl;
			return false;
			break;
	}				
					
	return true;
}