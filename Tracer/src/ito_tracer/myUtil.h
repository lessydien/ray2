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

#ifndef MYUTIL_H
	#define MYUTIL_H

#include <optix.h>

  /************************************
   **
   **    Error checking helpers 
   **
   ***********************************/

  void myUtilReportError(const char* message);
  void myUtilHandleError(RTcontext context, RTresult code, const char* file, int line);
  void myUtilHandleErrorNoExit(RTcontext context, RTresult code, const char* file, int line);
  void myUtilHandleErrorNoContext(RTresult code, const char* file, int line);


/* assumes current scope has Context variable named 'context' */
#define RT_CHECK_ERROR( func )                                      \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      myUtilHandleError( context, code, __FILE__, __LINE__ );       \
  } while(0)

inline bool RT_CHECK_ERROR_NOEXIT( RTresult code, RTcontext context )
{
	if ( code != RT_SUCCESS )
	{
		myUtilHandleErrorNoExit( context, code, __FILE__, __LINE__ );
		return false;
	}
	return true;
}

/* assumes current scope has Context pointer variable named 'context' */
#define RT_CHECK_ERROR2( func )                                    \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      myUtilHandleError( *context, code, __FILE__, __LINE__ );      \
  } while(0)
 
/* assumes current scope has Context variable named 'context' */
#define RT_CHECK_ERROR_RETURN( func )                              \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS ) {                                     \
      myUtilHandleErrorNoExit( context, code, __FILE__, __LINE__ ); \
      return code;                                                 \
    }                                                              \
  } while(0)

/* assumes that there is no context, just print to stderr */
#define RT_CHECK_ERROR_NO_CONTEXT( func )                          \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      myUtilHandleErrorNoContext(code, __FILE__, __LINE__ );        \
  } while(0)

#endif