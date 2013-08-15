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
#include "FlexZemax.h"

char yylval[20];

extern FILE *yyin;

extern FILE *yyout;

char testchar[20];
double testdouble;

ZemaxParamStruct *surfaceArray;
/* index counting the zemax surfaces. init to -1 even though we start with index 0 later */
long int surfIndex = -1;
/* index counting the surface details. init to -1 even though we start with index 0 later */
long int surfDetIndex = -1;
/* index counting the rayTrace geometries ( remember that e.g. coordinate breaks are Zemax surfaces but no raytRace geometries */
long int geomIndex = 0;

double3 globalRoot;
double3 globalNormal;

long int surfLength=0;

/* flag to indicate the last surface in parsing the surface summary */
int lastSurfSumFlag=0;


%}

string  \"[^\n"]+\"



%s SURFSUM
%s SURFSUM_TYPE
%s SURFSUM_CMT
%s SURFSUM_RAD
%s SURFSUM_TKS
%s SURFSUM_GLS
%s SURFSUM_DIA
%s SURFSUM_CON
%s SURFSUM_CRDBRK
%s SURFSUM_END

%s LENSNOTES
%s LENSNOTES_SURF

%s SURFDETAIL
%s SURFDETAIL_SURF
%s SURFDETAIL_DECX
%s SURFDETAIL_DECY
%s SURFDETAIL_APTR
%s SURFDETAIL_APTR_X
%s SURFDETAIL_APTR_Y
%s SURFDETAIL_TILTX
%s SURFDETAIL_TILTY
%s SURFDETAIL_TILTZ
%s SURFDETAIL_TILT_2STAGE_DEC1X
%s SURFDETAIL_TILT_2STAGE_DEC1Y
%s SURFDETAIL_TILT_2STAGE_TILT1X
%s SURFDETAIL_TILT_2STAGE_TILT1Y
%s SURFDETAIL_TILT_2STAGE_TILT1Z
%s SURFDETAIL_TILT_2STAGE_1ORDER
%s SURFDETAIL_TILT_2STAGE
%s SURFDETAIL_TILT_2STAGE_DEC2X
%s SURFDETAIL_TILT_2STAGE_DEC2Y
%s SURFDETAIL_TILT_2STAGE_TILT2X
%s SURFDETAIL_TILT_2STAGE_TILT2Y
%s SURFDETAIL_TILT_2STAGE_TILT2Z
%s SURFDETAIL_TILT_2STAGE_2ORDER
%s SURFDETAIL_ORDER



ws      [ \t]+

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
glass    "MIRROR"|"N-BK7"

%%
"SURFACE DATA SUMMARY:" {
           BEGIN SURFSUM;
          }
          
"LENS NOTES:" {
		   BEGIN LENSNOTES;
		  }      
		  
"SURFACE DATA DETAIL:" {
			BEGIN SURFDETAIL;
		  }

 /* ---------------------------------------------------------------*/
 
 <SURFSUM>{surfnum} {
			fprintf( yyout, "surf nr: %s;", yytext); 
			BEGIN SURFSUM_TYPE;
 }

 <SURFSUM>{surflast} {
			fprintf( yyout, "last surf:"); 
			lastSurfSumFlag=1;
			BEGIN SURFSUM_TYPE;
 }

 /*------------------------------------------------------------------------*/
 
 <SURFSUM_TYPE>"STANDARD" {
			 fprintf( yyout, "type: standard;");
			 surfIndex=surfIndex+1;
			 geomIndex=geomIndex+1;
			 surfaceArray[surfIndex].type=STANDARD;
			 BEGIN SURFSUM_CMT;
 }

 <SURFSUM_TYPE>"COORDBRK" {
			 fprintf( yyout, "type: coordinate break;");
			 surfIndex=surfIndex+1;
			 surfaceArray[surfIndex].type=CRDBRK;
			 BEGIN SURFSUM_CRDBRK;
 }
 
  <SURFSUM_TYPE>"DGRATING" {
			 fprintf( yyout, "type: grating;");
    		 surfIndex=surfIndex+1;
			 geomIndex=geomIndex+1;
			 surfaceArray[surfIndex].type=DGRATING;
			 BEGIN SURFSUM_CMT;
 }
 
   <SURFSUM_TYPE>{name} {
			 fprintf( yyout, "type: unknown;");
			 surfaceArray[surfIndex].type=UNKNOWNOTYPE;
			 // implement some error mechnaism rather than continue parsing !!
			 surfIndex=surfIndex+1;
			 BEGIN SURFSUM_CMT;
 }
  
 /*------------------------------------------------------------------------*/

 <SURFSUM_CRDBRK>{num} {
			 fprintf( yyout, " thickness: %s", yytext);
			 surfaceArray[surfIndex].thickness=atof(yytext);
			 BEGIN SURFSUM_END;
 }

 /*------------------------------------------------------------------------*/

 <SURFSUM_CMT>{num} {
 			 fprintf( yyout, " no name; radius %s;", yytext);
 			 surfaceArray[surfIndex].radius=atof(yytext);
			 BEGIN SURFSUM_TKS; /* if we don't find a valid name, assign no name and interprete current text as radius. Therefore proceed to thickness */
 }
 
 <SURFSUM_CMT>"Infinity" {
 			 fprintf( yyout, " no name; radius %s;", yytext);
 			 surfaceArray[surfIndex].radius=99;
			 BEGIN SURFSUM_TKS; /* if we don't find a valid name, assign no name and interprete current text as radius. Therefore proceed to thickness */
 }

 <SURFSUM_CMT>"- " {
 			 fprintf( yyout, " no name; radius %s;", yytext);
			 BEGIN SURFSUM_TKS; /* if we don't find a valid name, assign no name and interprete current text as radius. Therefore proceed to thickness */
 }
 
 <SURFSUM_CMT>{name} {
 			 fprintf( yyout, " name: %s;", yytext);
 			 sprintf(surfaceArray[surfIndex].comment, yytext);
			 BEGIN SURFSUM_RAD;
}

 
 /*--------------------------------------------------------------------------*/
 
 <SURFSUM_RAD>{num} {
 			 fprintf( yyout, " rad: %s;", yytext);
 			 surfaceArray[surfIndex].radius=atof(yytext);
			 BEGIN SURFSUM_TKS;
 } 
 
 <SURFSUM_RAD>"Infinity" {
 			 fprintf( yyout, " radius: %s index: %i;", yytext, surfIndex);
 			 surfaceArray[surfIndex].radius=ZEMAX_INF;
			 BEGIN SURFSUM_TKS;
 } 
 
 <SURFSUM_RAD>"- " {
 			 fprintf( yyout, " radius %s;", yytext);
			 BEGIN SURFSUM_TKS; 
 }
 
 /*--------------------------------------------------------------------------*/
 
 <SURFSUM_TKS>{num} {
 			 fprintf( yyout, " thickness: %s;", yytext);
 			 surfaceArray[surfIndex].thickness=atof(yytext);
			 BEGIN SURFSUM_GLS; 
 }
 
 /*---------------------------------------------------------------------------*/
 
 <SURFSUM_GLS>"MIRROR" {
 			 fprintf( yyout, " glass: %s;", yytext);
 			 surfaceArray[surfIndex].glass=MIRROR;
			 BEGIN SURFSUM_DIA; 
 } 
 
 <SURFSUM_GLS>"N-BK7" {
 			 fprintf( yyout, " glass: %s;", yytext);
 			 surfaceArray[surfIndex].glass=NBK7;
			 BEGIN SURFSUM_DIA; 
 } 
 
 <SURFSUM_GLS>{num} {
 			 fprintf( yyout, " no glass; diameter: %s;", yytext);
 			 surfaceArray[surfIndex].diameter=atof(yytext);
			 BEGIN SURFSUM_CON; /* if we find a number here, we probably don't have any glass and read the diameter instead. So proceed to conic directly */ 
 } 
 
 /*-------------------------------------------------------------------------------*/
 
 <SURFSUM_DIA>{num} {
 			 fprintf( yyout, " diameter: %s;", yytext);
 			 surfaceArray[surfIndex].diameter=atof(yytext);
			 BEGIN SURFSUM_CON; 
 }
 
 /*---------------------------------------------------------------------------------*/
 
 <SURFSUM_CON>{num} {
 			 fprintf( yyout, " con: %s", yytext);
 			 surfaceArray[surfIndex].conic=atof(yytext);
			 BEGIN SURFSUM_END; 
 }
 
 /*------------------------------------------------------------------------*/

 <SURFSUM_END>"\n" {
			if (lastSurfSumFlag==0)
			{
				fprintf( yyout, "\n");
				BEGIN SURFSUM;
			}
			else
			{
				BEGIN 0;
				fprintf( yyout, "\n\n");
			}
 }	

  /*-------------------------------------------------------------------------*/
  /*                                                                         */
  /*					Lense notes											 */
  /*																		 */
  /*-------------------------------------------------------------------------*/
  
  <LENSNOTES>"Surfaces" {
			BEGIN LENSNOTES_SURF;
  }
  
  
  <LENSNOTES_SURF>{int} {
			fprintf( yyout, "%i \n", atoi(yytext));
			surfLength=atoi(yytext);
			surfLength=surfLength+1;
			surfaceArray = new ZemaxParamStruct[surfLength];
			long int i;
			for (i=0; i<surfLength; i++)
			{
				initSurfaceStruct(&surfaceArray[i]);
			}
			BEGIN 0;
  }

  /*-------------------------------------------------------------------------*/
  /*                                                                         */
  /*					surface details										 */
  /*																		 */
  /*-------------------------------------------------------------------------*/

  <SURFDETAIL>"Surface" {
			/* make sure we have no more surface details than we had surfaces before */
			if ( !(surfDetIndex<=surfIndex) )
			{
				fprintf( yyout, "\n surfDetIndex %i bigger than surfIndex %s", surfDetIndex, surfIndex);
				// some error formalism to be implemented !!	
			}
			surfDetIndex=surfDetIndex+1;
			fprintf( yyout, "\n surface %i", surfDetIndex);
			BEGIN SURFDETAIL_SURF;
  }
  <SURFDETAIL>"Surface IMA" {
			/* make sure we have no more surface details than we had surfaces before */
			if ( !(surfDetIndex<=surfIndex) )
			{
				fprintf( yyout, "\n surfDetIndex %i bigger than surfIndex %s", surfDetIndex, surfIndex);
				// some error formalism to be implemented !!	
			}
			surfDetIndex=surfDetIndex+1;
			fprintf( yyout, "\n last surface %i", surfDetIndex);
			BEGIN 0;
  }
		<SURFDETAIL_SURF>"STANDARD" {
				/* make sure the surface numbering of the summary and the details is consistent */
				if ( !(surfaceArray[surfDetIndex].type!=STANDARD) )
				{
					// some error formalism to be implemented
				}
				fprintf( yyout, " type: %s", yytext);
				BEGIN SURFDETAIL;
		}

		<SURFDETAIL_SURF>"COORDBRK" {
				/* make sure the surface numbering of the summary and the details is consistent */
				if ( !(surfaceArray[surfDetIndex].type!=CRDBRK) )
				{
					// some error formalism to be implemented
				}
				fprintf( yyout, " type: %s", yytext);
				BEGIN SURFDETAIL;
		}
  
		<SURFDETAIL_SURF>"DGRATING" {
				/* make sure the surface numbering of the summary and the details is consistent */
				if ( !(surfaceArray[surfDetIndex].type!=DGRATING) )
				{
					// some error formalism to be implemented
				}
				fprintf( yyout, " type: %s", yytext);
				BEGIN SURFDETAIL;
		}

		  <SURFDETAIL_SURF>"Surface" {
					/* make sure we have no more surface details than we had surfaces before */
					if ( !(surfDetIndex<=surfIndex) )
					{
						fprintf( yyout, "\n surfDetIndex %i bigger than surfIndex %s", surfDetIndex, surfIndex);					
						// some error formalism to be implemented !!	
					}
					surfDetIndex=surfDetIndex+1;
					fprintf( yyout, "\n surface %i", surfDetIndex);
					BEGIN SURFDETAIL_SURF;
		  }
    
  
  <SURFDETAIL>"Decenter X" {
			BEGIN SURFDETAIL_DECX;
  }

	  <SURFDETAIL_DECX>{number} {
			surfaceArray[surfDetIndex].details.dec1.x=atof(yytext);
			fprintf( yyout, " decenterX: %s", yytext);
			BEGIN SURFDETAIL;
		}

  <SURFDETAIL>"Decenter Y" {
			BEGIN SURFDETAIL_DECY;
  }

	  <SURFDETAIL_DECY>{number} {
			surfaceArray[surfDetIndex].details.dec1.y=atof(yytext);
			fprintf( yyout, " decenterY: %s", yytext);
			BEGIN SURFDETAIL;
		}
  
  <SURFDETAIL>"Aperture" {
			fprintf( yyout, " Aperture!! :");
			BEGIN SURFDETAIL_APTR;
  }

	  <SURFDETAIL_APTR>"Rectangular Aperture" {
			surfaceArray[surfDetIndex].details.aperture=RECT;
			fprintf( yyout, " rectangular;");
			BEGIN SURFDETAIL_APTR;
	  }
	  
  	  <SURFDETAIL_APTR>"Elliptical Aperture" {
			surfaceArray[surfDetIndex].details.aperture=ELLIPT;
			fprintf( yyout, " elliptical;");
			BEGIN SURFDETAIL_APTR;
	  }

	  <SURFDETAIL_APTR>"Rectangular Obscuration" {
			surfaceArray[surfDetIndex].details.aperture=RECTOBSC;
			fprintf( yyout, " obscuration!! :");
			BEGIN SURFDETAIL_APTR;
	  }

	  <SURFDETAIL_APTR>"Elliptical Obscuration" {
			surfaceArray[surfDetIndex].details.aperture=ELLIPTOBSC;
			fprintf( yyout, " elliptical; ");
			BEGIN SURFDETAIL_APTR;
	  }

   	  <SURFDETAIL_APTR>"X Half Width" {
			surfaceArray[surfDetIndex].details.apertureHalfWidth.x=atof(yytext);
			BEGIN SURFDETAIL_APTR_X;
	  }

   		  <SURFDETAIL_APTR_X>{num} {
				surfaceArray[surfDetIndex].details.apertureHalfWidth.x=atof(yytext);
				fprintf( yyout, " x-half width: %s", yytext);
				BEGIN SURFDETAIL_APTR;
		  }

   	  <SURFDETAIL_APTR>"Y Half Width" {
			surfaceArray[surfDetIndex].details.apertureHalfWidth.y=atof(yytext);
			BEGIN SURFDETAIL_APTR_Y;
	  }
	  
   		  <SURFDETAIL_APTR_Y>{num} {
				surfaceArray[surfDetIndex].details.apertureHalfWidth.y=atof(yytext);
				fprintf( yyout, " y-half width: %s", yytext);
				BEGIN SURFDETAIL;
		  }

		  <SURFDETAIL_APTR>"Surface" {
					/* make sure we have no more surface details than we had surfaces before */
					if ( !(surfDetIndex<=surfIndex) )
					{
				fprintf( yyout, "\n surfDetIndex %i bigger than surfIndex %s", surfDetIndex, surfIndex);					
						// some error formalism to be implemented !!	
					}
					surfDetIndex=surfDetIndex+1;
					fprintf( yyout, "\n surface %i", surfDetIndex);
					BEGIN SURFDETAIL_SURF;
		  }

  <SURFDETAIL>"Tilt About X" {
			BEGIN SURFDETAIL_TILTX;
  }
  
		<SURFDETAIL_TILTX>{number} {
			surfaceArray[surfDetIndex].details.tilt1.x=atof(yytext);
			fprintf( yyout, " tiltX: %s ;", yytext);
			BEGIN SURFDETAIL;
		}
  
  <SURFDETAIL>"Tilt About Y" {
			BEGIN SURFDETAIL_TILTY;
  }

		<SURFDETAIL_TILTY>{number} {
			fprintf( yyout, " tiltY: %s ;", yytext);
			surfaceArray[surfDetIndex].details.tilt1.y=atof(yytext);
			BEGIN SURFDETAIL;
		}
  
  <SURFDETAIL>"Tilt About Z" {
			BEGIN SURFDETAIL_TILTZ;
  }
  
		<SURFDETAIL_TILTZ>{number} {
			fprintf( yyout, " tiltZ: %s ;", yytext);
			surfaceArray[surfDetIndex].details.tilt1.z=atof(yytext);
			BEGIN SURFDETAIL;
		}

  <SURFDETAIL>"Tilt/Decenter" {
			fprintf( yyout, " Tilt/Decenter:");
			BEGIN SURFDETAIL_TILT_2STAGE;
  }
  
		<SURFDETAIL_TILT_2STAGE>"Before surface" {
			fprintf( yyout, " before surf:");
			BEGIN SURFDETAIL_TILT_2STAGE_DEC1X;
		}
		
			<SURFDETAIL_TILT_2STAGE_DEC1X>{num} {
				surfaceArray[surfDetIndex].details.dec1.x=atof(yytext);
				fprintf( yyout, " dec-x: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_DEC1Y;
			}
		
			<SURFDETAIL_TILT_2STAGE_DEC1Y>{num} {
				surfaceArray[surfDetIndex].details.dec1.y=atof(yytext);
				fprintf( yyout, " dec-y: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_TILT1X;
			}

			<SURFDETAIL_TILT_2STAGE_TILT1X>{num} {
				surfaceArray[surfDetIndex].details.tilt1.x=atof(yytext);
				fprintf( yyout, " tilt-x: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_TILT1Y;
			}

			<SURFDETAIL_TILT_2STAGE_TILT1Y>{num} {
				surfaceArray[surfDetIndex].details.tilt1.y=atof(yytext);
				fprintf( yyout, " tilt-y: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_TILT1Z;
			}
			
			<SURFDETAIL_TILT_2STAGE_TILT1Z>{num} {
				surfaceArray[surfDetIndex].details.tilt1.z=atof(yytext);
				fprintf( yyout, " tilt-z: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_1ORDER;
			}
			
			<SURFDETAIL_TILT_2STAGE_1ORDER>"Tilt, Decenter" {
				surfaceArray[surfDetIndex].details.order1=TILTDEC;
				fprintf( yyout, " order: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE;
			}

			<SURFDETAIL_TILT_2STAGE_1ORDER>"Decenter, Tilt" {
				surfaceArray[surfDetIndex].details.order1=DECTILT;
				fprintf( yyout, " order: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE;
			}

		<SURFDETAIL_TILT_2STAGE>"After surface" {
			fprintf( yyout, " after surface:");
			BEGIN SURFDETAIL_TILT_2STAGE_DEC2X;
		}
		
			<SURFDETAIL_TILT_2STAGE_DEC2X>{num} {
				surfaceArray[surfDetIndex].details.dec2.x=atof(yytext);
				fprintf( yyout, " dec-x: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_DEC2Y;
			}
		
			<SURFDETAIL_TILT_2STAGE_DEC2Y>{num} {
				surfaceArray[surfDetIndex].details.dec2.y=atof(yytext);
				fprintf( yyout, " dec-y: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_TILT2X;
			}

			<SURFDETAIL_TILT_2STAGE_TILT2X>{num} {
				surfaceArray[surfDetIndex].details.tilt2.x=atof(yytext);
				fprintf( yyout, " tilt-x: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_TILT2Y;
			}

			<SURFDETAIL_TILT_2STAGE_TILT2Y>{num} {
				surfaceArray[surfDetIndex].details.tilt2.y=atof(yytext);
				fprintf( yyout, " tilt-y: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_TILT2Z;
			}
			
			<SURFDETAIL_TILT_2STAGE_TILT2Z>{num} {
				surfaceArray[surfDetIndex].details.tilt2.z=atof(yytext);
				fprintf( yyout, " tilt-z: %s", yytext);
				BEGIN SURFDETAIL_TILT_2STAGE_2ORDER;
			}
			
			<SURFDETAIL_TILT_2STAGE_2ORDER>"Tilt, Decenter" {
				surfaceArray[surfDetIndex].details.order2=TILTDEC;
				fprintf( yyout, " order: %s", yytext);
				BEGIN SURFDETAIL;
			}

			<SURFDETAIL_TILT_2STAGE_2ORDER>"Decenter, Tilt" {
				surfaceArray[surfDetIndex].details.order2=DECTILT;
				fprintf( yyout, " order: %s", yytext);
				BEGIN SURFDETAIL;
			}

<SURFDETAIL>"Diffract Order" {
			// ignore this command but make sure we don't take it for an "order" command !!
			BEGIN SURFDETAIL;
  }
   
  <SURFDETAIL>"Order" {
			BEGIN SURFDETAIL_ORDER;
  }
  
		<SURFDETAIL_ORDER>"Decenter then tilt" {
			surfaceArray[surfDetIndex].details.order1=DECTILT;
			fprintf( yyout, " order: %s", yytext);
			BEGIN SURFDETAIL;
		}

		<SURFDETAIL_ORDER>"Tilt then decenter" {
			surfaceArray[surfDetIndex].details.order1=TILTDEC;
			fprintf( yyout, " order: %s", yytext);
			BEGIN SURFDETAIL;
		}

  /*----------------------------------------------------------------------------*/

 


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

parserError parseZemaxPrescr(parseResultStruct* parseResults, FILE *hfile)
{
	yyout = fopen( "e:\\mauch\\prescriptionLexOut.txt", "w" ) ;
	yyin = hfile;
	
	/* init index variables for the current parsing */
	surfDetIndex = -1; // init to -1 because we weird incrementing inside parser
	surfIndex = -1; // init to -1 because we weird incrementing inside parser
	geomIndex = 0;	
    
    /* do the actual parsing */
	while(yylex() != 0)
		;
	fclose(yyout);
	
	/* allocate memory to store the result of the detected geometries. Not that in general this number will be smaller than the detected Zemax surfaces ( Coordinate breaks... ) */
	parseResults->geometryParams = new GeometryParamStruct[geomIndex];
	parseResults->geomNumber=geomIndex;

	/* init global coordinate system */
	globalRoot=make_double3(0.0,0.0,0.0);
	globalNormal=make_double3(0.0,0.0,1.0);
	/* init shift vector. This is needed to realize decentering of surfaces and coordinate breaks */
	double3 shiftVec=make_double3(0.0,0.0,0.0);	

	/* define loop counters */
	long int iSurf;
	long int iGeom=0;
	/* loop through surfaceArray and place geometries in global coordinate system */
	for (iSurf=0; iSurf<=surfIndex ;iSurf++)
	{
		ZemaxParamStruct test=surfaceArray[iSurf];
		if (iGeom<=geomIndex)
		{
			switch(surfaceArray[iSurf].type)	
			{
				case CRDBRK:
					/* take care of the decenter of the coordinate break */
					doDecenter(surfaceArray[iSurf], BEFORESURF);
					break;

				case STANDARD:
					if (surfaceArray[iSurf].radius==ZEMAX_INF)
					{
						/* take care of the decenter before the surface */
						doDecenter(surfaceArray[iSurf], BEFORESURF);

						/* copy the geometry params */
						parseResults->geometryParams[iGeom].normal=globalNormal;
						parseResults->geometryParams[iGeom].root=globalRoot;
						parseResults->geometryParams[iGeom].type=GEOM_PLANESURF;
						parseResults->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
						parseResults->geometryParams[iGeom].apertureHalfWidth=surfaceArray[iSurf].details.apertureHalfWidth;
						parseResults->geometryParams[iGeom].radius=surfaceArray[iSurf].radius;
						parseResults->geometryParams[iGeom].glass=surfaceArray[iSurf].glass;
						parseResults->geometryParams[iGeom].diameter=surfaceArray[iSurf].diameter;
						parseResults->geometryParams[iGeom].conic=surfaceArray[iSurf].conic;
						memcpy ( parseResults->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
					}
					else
					{
						/* take care of the decenter before the surface */
						doDecenter(surfaceArray[iSurf], BEFORESURF);

						/* copy the geometry params */
						parseResults->geometryParams[iGeom].normal=globalNormal;
						parseResults->geometryParams[iGeom].root=globalRoot;
						parseResults->geometryParams[iGeom].type=GEOM_SPHERICALSURF;
						parseResults->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
						parseResults->geometryParams[iGeom].apertureHalfWidth=surfaceArray[iSurf].details.apertureHalfWidth;
						parseResults->geometryParams[iGeom].radius=surfaceArray[iSurf].radius;
						parseResults->geometryParams[iGeom].glass=surfaceArray[iSurf].glass;
						parseResults->geometryParams[iGeom].diameter=surfaceArray[iSurf].diameter;
						parseResults->geometryParams[iGeom].conic=surfaceArray[iSurf].conic;
						memcpy ( parseResults->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
					}					
					/* take care of the decenter after the surface */
					doDecenter(surfaceArray[iSurf], AFTERSURF);
					// increment geometry counter
					iGeom=iGeom+1;
					break;


				case DGRATING:
					/* take care of the decenter before the surface */
					doDecenter(surfaceArray[iSurf], BEFORESURF);

					/* copy the geometry params */
					parseResults->geometryParams[iGeom].normal=globalNormal;
					parseResults->geometryParams[iGeom].root=globalRoot;
					parseResults->geometryParams[iGeom].type=GEOM_GRATING;
					parseResults->geometryParams[iGeom].aperture=surfaceArray[iSurf].details.aperture;
					parseResults->geometryParams[iGeom].apertureHalfWidth=surfaceArray[iSurf].details.apertureHalfWidth;
					parseResults->geometryParams[iGeom].radius=surfaceArray[iSurf].radius;
					parseResults->geometryParams[iGeom].glass=surfaceArray[iSurf].glass;
					parseResults->geometryParams[iGeom].diameter=surfaceArray[iSurf].diameter;
					parseResults->geometryParams[iGeom].conic=surfaceArray[iSurf].conic;
					memcpy ( parseResults->geometryParams[iGeom].comment, surfaceArray[iSurf].comment, GEOM_CMT_LENGTH*sizeof(char) );
							
					/* take care of the decenter after the surface */
					doDecenter(surfaceArray[iSurf], AFTERSURF);
					// increment geometry counter
					iGeom=iGeom+1;
					break;					


				default:
					// implement some error mechanism
					break;

			} // end switch
			/* update global coordinate system */
			globalRoot=globalRoot+surfaceArray[iSurf].thickness*globalNormal;
		} // end if (iGeom<=geomIndex)
		else
		{
			// implement some error formalism
		}
		
	} // end for

	return PARSER_NO_ERR;
}	
	
parserError initSurfaceStruct(ZemaxParamStruct* surface)
{
	surface->type=UNKNOWNOTYPE;
	sprintf(surface->comment,"");
	surface->radius=0;
	surface->thickness=0;
	surface->glass=UNKNOWNGLASS;
	surface->diameter=0;
	surface->conic=0;
	sprintf(surface->comment, "%s", "NO COMMENT");
	surface->details.dec1=make_double3(0,0,0);
	surface->details.dec2=make_double3(0,0,0);
	surface->details.tilt1=make_double3(0,0,0);
	surface->details.tilt2=make_double3(0,0,0);
	surface->details.order1=TILTDEC;
	surface->details.order2=TILTDEC;
	surface->details.aperture=UNKNOWNATYPE;
	surface->details.apertureHalfWidth=make_double2(0,0);
	
	return PARSER_NO_ERR;
}

void doDecenter(ZemaxParamStruct surfacePar, DecenterType type)
{
	ZemaxParamStruct test = surfacePar;
 	if (type==BEFORESURF)
	{
		if (surfacePar.details.order1==DECTILT)
		{
			/* rotate the decenter vector */
			double3 decTmp=surfacePar.details.dec1;
			rotateRay(&decTmp,surfacePar.details.tilt1);	
			/* do the decenter */
			globalRoot=globalRoot+decTmp;
			/* rotate normal */
			rotateRay(&globalNormal,surfacePar.details.tilt1);	
		}
		else
		{
			/* do the decenter */
			globalRoot=globalRoot+surfacePar.details.dec1;
			/* rotate normal */
			rotateRay(&globalNormal,surfacePar.details.tilt1);	
		}
	}
	else
	{
		if (surfacePar.details.order1==DECTILT)
		{
			/* rotate the decenter vector */
			double3 decTmp=surfacePar.details.dec2;
			rotateRay(&decTmp,surfacePar.details.tilt2);	
			/* do the decenter */
			globalRoot=globalRoot+decTmp;
			/* rotate normal */
			rotateRay(&globalNormal,surfacePar.details.tilt2);	
		}
		else
		{
			/* do the decenter */
			globalRoot=globalRoot+surfacePar.details.dec2;
			/* rotate normal */
			rotateRay(&globalNormal,surfacePar.details.tilt2);	
		}
	}
}