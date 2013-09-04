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

#include "myUtil.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream> 

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include<windows.h>
#    include<mmsystem.h>
#else /*Apple and Linux both use this */
#    include<sys/time.h>
#    include <unistd.h>
#    include <dirent.h>
#endif

void myUtilReportError(const char* message)
{
  std::cout << "OptiX Error: " << message << std::endl;
}

void myUtilHandleError(RTcontext context, RTresult code, const char* file, int line)
{
  myUtilHandleErrorNoExit( context, code, file, line );
  exit(1);
}

void myUtilHandleErrorNoExit(RTcontext context, RTresult code, const char* file, int line)
{
  const char* message;
  char s[2048];
  rtContextGetErrorString(context, code, &message);
  sprintf(s, "%s\n(%s:%d)", message, file, line);
  s[2047]='\0';
  myUtilReportError( s );
}

void myUtilHandleErrorNoContext(RTresult code, const char* file, int line)
{
  char s[2048];
  sprintf(s, "Code: %d\n(%s:%d)", code, file, line);
  s[2047]='\0';
  myUtilReportError( s );
 // exit(1);
}