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

#include<iostream>
using namespace std;
int tasknr = -1;

int mylineno = 0;
int cnt =0;
int cntProperties =0;    /* für DB */
int cntPropertySets =0;   /* für DB */
int propcnt2;            /* zählt bei der Def. von PropSet die Properties hoch */
int finalindex;
int cntsets = 0;         /* zählt die PropertySets innerhalb der Object Definition*/
char dummy[200];
char propname[200];
char propsetname[200];
char mename[200];
 int proptype;              /* 1 = double, 2=int, 3=text */

#define MAXPROPERTIES 1000
#define MAXPROPERTYSETS 500
#define MAXPROPSPERSET 100

string dbproperties[MAXPROPERTIES];
string dbpropertysets[MAXPROPERTYSETS];
int dbpropsinsets[MAXPROPERTYSETS][MAXPROPSPERSET];
int dbpropertytypes[MAXPROPERTYSETS];

%}

string  \"[^\n"]+\"



%s PROPERTY
%s PROPERTY2
%s PROPERTYSET
%s PROPERTYSET2
%s MEASUREMENTENTITY
%s MEASUREMENTENTITY2
%s MEASUREMENTENTITY3
%s MEASUREMENTENTITY4
%s OBJECT
%s OBJECT2
%s OBJECT3
%s OBJECT4
%s OBJECT5
%s TASK
%s TASK2
%s ENVIROMENT


ws      [ \t]+

alpha   [A-Za-z]
dig     [0-9]
name    ({alpha}|{dig}|\$)({alpha}|{dig}|[_.\-/$])*
typename "STRING"{name}
num1    [-+]?{dig}+\.?([eE][-+]?{dig}+)?
num2    [-+]?{dig}*\.{dig}+([eE][-+]?{dig}+)?
number  {num1}|{num2}
numberunit  {number}|{number}mm|{number}mu

%%
"define PropertySet" {
           BEGIN PROPERTYSET;
          }

"define Property" {
           BEGIN PROPERTY;
          }

"define MeasurementEntity" {
           BEGIN MEASUREMENTENTITY;
          }

"define Object" {
           BEGIN OBJECT;
          }

"define Task" {
           BEGIN TASK;
          }

"define Enviroment" {
          cout << "\n\nclass Enviroment : public MeasurementEntity { " << endl;
          strcpy(mename, "Enviroment");
          cout << "  public:" << endl;
          cout << "     " << mename << "() : MeasurementEntity(\"Enviroment\")" << endl << "     {" << endl; 

          cnt = 0;
          BEGIN MEASUREMENTENTITY2;
         }

  /* ---------------------------------------------------------------*/
<PROPERTY>{name} {
          cout << "class " << YYText() << " : public Property { " << endl;
          strcpy(propname, YYText());
          dbproperties[cntProperties++] = propname;
          BEGIN PROPERTY2;
}

<PROPERTY2>"propertyDouble" {
          cout << "  public:" << endl;
          cout << "    " << propname << "() : Property(propertyDouble, \"" 
               << &propname[4] << "\") {}; "  << endl;
          dbpropertytypes[cntProperties-1] = 1;
          cout << "};" << endl << endl;
          BEGIN 0;
}
<PROPERTY2>"propertyInt" {
          cout << "  public:" << endl;
          cout << "    " << propname << "() : Property(propertyInt, \"" 
               << &propname[4] << "\"){}; "  << endl;
          cout << "};" << endl << endl;
          dbpropertytypes[cntProperties-1] = 2;
          BEGIN 0;
}

<PROPERTY2>"property2DUchar" {
          cout << "  public:" << endl;
          cout << "    " << propname << "() : Property(property2DUchar, \"" 
               << &propname[4] << "\") {}; "  << endl;
          cout << "};" << endl << endl;
          dbpropertytypes[cntProperties-1] = 3;
          BEGIN 0;
}

<PROPERTY2>"property2DDouble" {
          cout << "  public:" << endl;
          cout << "    " << propname << "() : Property(property2DDouble, \"" 
               << &propname[4] << "\") {}; "  << endl;
          cout << "};" << endl << endl;
          dbpropertytypes[cntProperties-1] = 3;
          BEGIN 0;
}

<PROPERTY2>"property1DDouble" {
          cout << "  public:" << endl;
          cout << "    " << propname << "() : Property(property1DDouble, \"" 
               << &propname[4] << "\" {}; "  << endl;
          cout << "};" << endl << endl;
          dbpropertytypes[cntProperties-1] = 3;
          BEGIN 0;
}

<PROPERTY2>"property3DDouble" {
          cout << "  public:" << endl;
          cout << "    " << propname << "() : Property(property3DDouble, \"" 
               << &propname[4] << "\") {}; "  << endl;
          cout << "};" << endl << endl;
          dbpropertytypes[cntProperties-1] = 3;
          BEGIN 0;
}

<PROPERTY2>"propertyString" {
          cout << "  public:" << endl;
          cout << "    " << propname << "() : Property(propertyString, \"" 
               << &propname[4] << "\") {}; "  << endl;
          cout << "};" << endl << endl;
          dbpropertytypes[cntProperties-1] = 3;
          BEGIN 0;
}

  /* ---------------------------------------------------------------*/
<PROPERTYSET>{name} {
          cout << "class " << YYText() << " : public PropertySet { " << endl;
          strcpy(propname, YYText());
          cout << "  public:" << endl;
          cout << "     " << propname << "() : PropertySet(\"" << 
                   YYText() << "\") " << endl << "     {" << endl; 

          cnt = 0;
          dbpropertysets[cntPropertySets++] = propname;
          propcnt2 = 0;
          BEGIN PROPERTYSET2;
}

<PROPERTYSET2>"{" 
<PROPERTYSET2>"}" { 
          cout << "      }" << endl << endl;
          cout << "     ~" << propname << "() {}; " << endl << "};" << endl << endl; 
          BEGIN 0;}

<PROPERTYSET2>{name}  {
          cout << "          mProperties[" << cnt++ << "] = new "<< YYText() << "();" << endl;
          cout << "          mCntProperties++;" << endl;
          /* ------- Hier muss man jetzt noch die Datenbank füllen, also index des 
             Properties einfügen */
 
        for(int t=0; t < cntProperties; t++)
          if(YYText() == dbproperties[t])
            { 
              dbpropsinsets[cntPropertySets-1][propcnt2++] = t;
              break;
            }
          }




  /* ---------------------------------------------------------------*/
<MEASUREMENTENTITY>{name} {
          cout << "class " << YYText() << " : public MeasurementEntity { " << endl;
          strcpy(mename, YYText());
          cout << "  public:" << endl;
          cout << "     " << mename << "() : MeasurementEntity(\"" << 
                   YYText() << "\")" << endl << "     {" << endl; 

          cnt = 0;
          BEGIN MEASUREMENTENTITY2;
}

<MEASUREMENTENTITY2>"{" 
<MEASUREMENTENTITY3>"}" { 
          cout << "      };" << endl << endl;
          cout << "     ~" << mename << "() {}; " << endl << "};" << endl << endl; 
          BEGIN 0;}

<MEASUREMENTENTITY2>{name}  {

          cout <<     "        mPropertySet = new " << YYText() << " (); "<< endl;
          BEGIN MEASUREMENTENTITY3;
          strcpy(propsetname, YYText());
          }

	   /*  Ab hier wirds kompliziert !!! weil man die Properties kennen muss 
               Eigentlich müsste man hier jetzt bei ieder Property folgendes machen:
               1. Ist die Property überhaupt im PropertySet definiert ? Falls nein
                  dann Fehler 
               2. Welches Property ist das innerhalb des PropertySets -> i 
               3. mValueMinAllowed und maxAllowed [i] entsprechend setzen */

<MEASUREMENTENTITY3>Prop{name}  {

          strcpy(propname, YYText());

	  /* Step 1 & 2: Search in the property Database for the propname */
          int pos = -1;
          for(int t=0;t < cntProperties; t++)
          {
//           cout << propname <<  "  " << dbproperties[t] << endl;
            if(propname == dbproperties[t])
                {
                   pos = t;
                   break;
                }
          }
          if(pos == -1) { cerr << "PARSING ERROR 0 at " << mename << endl; exit(0);}

          /* pos gibt jetzt die Definition an, bei der in der dbproperty das
             entsprechende Property abgelegt ist. Also muss man jetzt in der
             passenden propertySet definition (also dbpropertyset) nachschauen,
             welchen index das property im propertyset hat.
             Eins nach dem anderen ! Also erst mal noch den propertySet suchen: */

          int posps = -1;
          for(int t=0;t < cntPropertySets; t++)
          {
  //          cout << propsetname << "  " << dbpropertysets[t] << endl;
            if(propsetname == dbpropertysets[t])
             {
                posps = t;
                break;
             }
          }          
          if(posps == -1) { cerr << "PARSING ERROR1 at " << mename << endl; exit(0);}

          /* Schön. und nun können wir endlich in dbpropsinsets nachschauen, welchen
             Index wir für ValueMinAllowed zu verwenden haben.          */

          finalindex = -1;
          for(int t=0;t < MAXPROPSPERSET; t++)
            {
//            cout << dbpropsinsets[posps][t] << "  " << pos <<  "  " << posps << endl;
            if(dbpropsinsets[posps][t] == pos)
            {
              finalindex = t;
              break;
            }
           }
          if(finalindex == -1) { cerr << "PARSING ERROR2 at " << mename << endl; exit(0);}
          BEGIN MEASUREMENTENTITY3;
          }

<MEASUREMENTENTITY3>{numberunit}  {
          /* Hier muss jetzt noch der Einheiten Kram rein */
          cout <<     "        mValueMinAllowed[" << finalindex << "] = " << YYText() 
           << ";" << endl;
          BEGIN MEASUREMENTENTITY4;
         }

<MEASUREMENTENTITY4>{numberunit}  {
          /* Hier muss jetzt noch der Einheiten Kram rein */
          cout <<     "        mValueMaxAllowed[" << finalindex << "] = " << YYText() 
               << ";" << endl;
          BEGIN MEASUREMENTENTITY4;
         }
<MEASUREMENTENTITY4>"IN"  {
          cout <<     "        mDirectProperty[" << finalindex << "] = 1" 
               << ";" << endl;
          BEGIN MEASUREMENTENTITY3;
}
<MEASUREMENTENTITY4>OUT  {
          cout <<     "        mDirectProperty[" << finalindex << "] = 2" 
               << ";" << endl;
          BEGIN MEASUREMENTENTITY3;
}
<MEASUREMENTENTITY4>"M"  {
          cout <<     "        mDirectProperty[" << finalindex << "] = 3" 
               << ";" << endl;
          BEGIN MEASUREMENTENTITY3;
}

  /* ----------------------------------------------------------------------/
  /* Jetzt kommt das selbe (nur dummerweise für N PropertySets auch nochmal für Obj.*/
<OBJECT>{name} {
          cout << "class " << YYText() << " : public Object { " << endl;
          strcpy(mename, YYText());
          cout << "  public:" << endl;
          cout << "     " << mename << "() : Object(\"" << 
                   YYText() << "\")" << endl << "     {" << endl; 

          cnt = 0;
          cntsets = -1;
          BEGIN OBJECT3;
}

<OBJECT2>"{" 
<OBJECT3>"}" { 
          cout << "      };" << endl << endl;
          cout << "     ~" << mename << "() {delete [] mPropertySets;}; " << endl 
             << "};" << endl << endl; 
          BEGIN 0;
          }


<OBJECT3>PropSet{name}  {

          cout << endl <<    "      mPropertySets[" << ++cntsets << "] = new " 
              << YYText()  << "();" << endl;
          strcpy(propsetname, YYText());
          BEGIN OBJECT3;
          }

<OBJECT3>Prop{name}  {
          strcpy(propname, YYText());
//          cout << "Step 1 for " << YYText() << endl;

	  /* Step 1 & 2: Search in the property Database for the propname */
          int pos = -1;
          for(int t=0;t < cntProperties; t++)
            {
//             cout << "test " << propname << "  " << dbproperties[t] << endl;
            if(propname == dbproperties[t])
                pos = t;
            }
//           cout << propname << " done" << pos << endl;
          if(pos == -1) { cerr << "PARSING ERROR at " << mename << endl; exit(0);}
          proptype = dbpropertytypes[pos];

          /* pos gibt jetzt die Definition an, bei der in der dbproperty das
             entsprechende Property abgelegt ist. Also muss man jetzt in der
             passenden propertySet definition (also dbpropertyset) nachschauen,
             welchen index das property im propertyset hat.
             Aber Eins nach dem anderen ! Also erst mal noch den propertySet suchen: */

          int posps = -1;
          for(int t=0;t < cntPropertySets; t++)
            if(propsetname == dbpropertysets[t])
                posps = t;
          if(posps == -1) { cerr << "PARSING ERROR 2 at " << mename << endl; exit(0);}

          /* Schön. und nun können wir endlich in dbpropsinsets nachschauen, welchen
             Index wir für ValueMinAllowed zu verwenden haben.          */
//          cout << "** " << propname << "  " << posps << " " << dbpropertysets[posps] << "  " << propsetname << endl;

          finalindex = -1;
          for(int t=0;t < MAXPROPSPERSET; t++)
            if(dbpropsinsets[posps][t] == pos)
            {
              finalindex = t;
              break;
            }
          if(finalindex == -1) { cerr << "PARSING ERROR 3 at " << mename << endl; exit(0);}
          BEGIN OBJECT4;
          }

<OBJECT4>{numberunit}  {
          /* Hier muss jetzt noch der Einheiten Kram rein */
          cout <<     "      mPropertySets[" << cntsets << "]->getProperty(" 
               << finalindex << ")->set";
          switch(proptype) {
           case 1: cout << "Double(" << YYText() << ");" << endl;
                   break;
           case 2: cout << "Int(" << YYText() << ");" << endl;
                   break;
          }
           BEGIN OBJECT3;
     }

<OBJECT4>{typename} {
          cout <<     "      mPropertySets[" << cntsets << "]->getProperty(" 
               << finalindex << ")->setString(\"" << YYText() << "\");" << endl;;
         BEGIN OBJECT3;
          }


  /* ----------------------------------------------------------------------*/
<TASK>{name}  {
          cout << "\nclass " << YYText() << " : public Task { " << endl;
          strcpy(mename, YYText());
          cout << "  public:" << endl;
          cout << "     " << mename << "() : Task(\"" << 
                   YYText() << "\", \"\")" << endl << "     {" << endl; 

          cnt = 0;
          cntsets = -1;
          BEGIN TASK2;
}

<TASK2>OBJ{name}  {
          cout  <<    "      mObject = new " 
              << YYText()  << "();" << endl;
          }

<TASK2>ME{name}  {
          cout <<    "      mMeasurementEntities[" << cnt++ << "] = new " 
              << YYText()  << "();" << endl;
          }
<TASK2>"}" { 
          tasknr++;
          cout << "      };" << endl << endl;
          cout << "     ~" << mename << "() { }; " << endl 
             << "};" << endl << endl; 
          cout << endl << "Task* taskGenerator"<<tasknr <<"()\n{\n return new "<< mename 
               << "();\n}" << endl;
          
          BEGIN 0;
          }


  /* ----------------------------------------------------------------------*/
.|\n {}
%%

// ____________________________________________________________

int main( int /* argc */, char** /* argv */ )
    {
    /* ---- INIT DB ------------ */
    for(int t=0; t < MAXPROPERTYSETS; t++)
       for(int u=0; u < MAXPROPSPERSET; u++)
          dbpropsinsets[t][u] = -1;

    cout << "#include\"definitions.h\"" << endl;
    cout << "#define AUTOMATICH" << endl;
    FlexLexer* lexer = new yyFlexLexer;
    while(lexer->yylex() != 0)
        ;

  // ---------- Now, we output a final function that returns an array
  // ---------- of pointer to tasks and the number of tasks

  cout << "\nint generateTasks(Task* alltasks[]) \n { \n ";
  for(int t=0; t <= tasknr; t++)
     cout << "alltasks["<<t <<"] = taskGenerator" << t << "();" << endl;
  cout << "\n return " << tasknr+1 << ";\n } " << endl;
  
#if 0
    /* now print out the DB */
    for(int t=0;t < cntProperties; t++)   
      cout << "Prop " << t << dbproperties[t] << endl;

    for(int t=0;t < cntPropertySets; t++)   
    {
      cout << t << dbpropertysets[t] << "  ";
      int u=0;
      while(dbpropsinsets[t][u] != -1)
        cout << dbpropsinsets[t][u++] << " ";
      cout << endl;
    }
#endif
    return 0;
    }
