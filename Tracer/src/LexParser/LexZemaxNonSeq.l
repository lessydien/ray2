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

parseDOEResultStruct *doeResultStructPtr;
int doeCoeffIndex=0;
int doeIndex=0;
int lexDOENr=0;

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
bool doeErr;

double scale2m=0.001;
double scale2mm=1;
double scale2um=1000;

%}

%s GLASS
%s GRATING

%s DOEDET

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
		  
"CC" {
		  fprintf( yyout, "found beginning of glass library \n");
		  BEGIN	GLASS;
		  }
		  
"GRATING DATA" {
		  BEGIN GRATINGDET;
		  }

"DOE" {
		  doeCoeffIndex=0;
		  doeIndex=0;
		  fprintf( yyout, "found DOE file marker");
		  doeErr=false;
		  BEGIN DOEDET;
		 }
		 		  
".*" {
		fprintf( yyout, "surprised finding %s \n", yytext);
	 }

 /* -----------------------DOE Parser------------------------------*/

<DOEDET>{number} {
			fprintf( yyout, " %f;", atof(yytext));
			if (doeIndex==lexDOENr)
			{
				doeResultStructPtr->coeffArray[doeCoeffIndex]=atof(yytext);
				doeCoeffIndex++;
			}
			BEGIN DOEDET;
			}

<DOEDET>"\n" {
			doeIndex++;
			BEGIN DOEDET;
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

parserError parseDOEFile(parseDOEResultStruct** parseResultsDOEPtrPtr, FILE *hfile, int DOEnr)
{
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "DOELexOut.txt");
	yyout = fopen( filepath, "w" ) ;
	lexDOENr=DOEnr;
	fprintf( yyout, "starting to parse DOE file \n");
	doeResultStructPtr=new parseDOEResultStruct();
	doeErr=true;
	yyin = hfile;
    /* do the actual parsing */
//	while(yylex() != 0)
//		;
	yylex();
	fclose(yyout);
	if (doeErr)
	{
		std::cout << "error in parseDOEFile" << std::endl;
		return PARSER_ERR;
	}
	doeResultStructPtr->coeffLength=doeCoeffIndex;
	sprintf(doeResultStructPtr->errMsg, "no error");
	*parseResultsDOEPtrPtr=doeResultStructPtr;
	return PARSER_NO_ERR;
}	

parserError parseMicroSimGratingData(ParseGratingResultStruct** parseResultsGratingPtrPtr, FILE *hfile)
{
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "gratingLexOut.txt");
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

parserError parseZemaxGlassCatalogOld(parseGlassResultStruct** parseResultsGlassPtrPtr, FILE *hfile, char *glassName)
{
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "glassLexOut.txt");
	yyout = fopen( filepath, "w" ) ;
	fprintf( yyout, "starting to parse glass catalog \n");
	glassResultStructPtr=new parseGlassResultStruct();
	sprintf(search_glassName, "%s", glassName);
	yyin = hfile;
	//yyin = fopen( "e:\\mauch\\glass.AGF", "r" );
    /* do the actual parsing */
//	while(yylex() != 0)
//		;
	yylex();
	fclose(yyout);
	if (!glassFound)
	{
		std::cout << "error in parseZemaxGlassCatalog: glass name -%s was not found" << (*glassName) << std::endl;
		return PARSER_ERR;
	}
	
	sprintf(glassResultStructPtr->errMsg, "no error");
	*parseResultsGlassPtrPtr=glassResultStructPtr;
	return PARSER_NO_ERR;
}	

parserError parseZemaxGlassCatalog(parseGlassResultStruct** parseResultsGlassPtrPtr, FILE *hfile, const char *glassName)
{
	char filepath[512];
	sprintf(filepath, "%s" PATH_SEPARATOR "%s", OUTPUT_FILEPATH, "glassLexOut.txt");
	yyout = fopen( filepath, "w" ) ;
	fprintf( yyout, "starting to parse glass catalog \n");
	glassResultStructPtr=new parseGlassResultStruct();
	sprintf(search_glassName, "%s", glassName);
	yyin = hfile;
	glassFound=false;
	//yyin = fopen( "e:\\mauch\\glass.AGF", "r" );
    /* do the actual parsing */
//	while(yylex() != 0)
//		;
	yylex();
	fclose(yyout);
	if (!glassFound)
	{
		std::cout << "error in parseZemaxGlassCatalog: glass name -%s was not found" << (*glassName) << std::endl;
		return PARSER_ERR;
	}
	
	sprintf(glassResultStructPtr->errMsg, "no error");
	*parseResultsGlassPtrPtr=glassResultStructPtr;
	return PARSER_NO_ERR;
}	

