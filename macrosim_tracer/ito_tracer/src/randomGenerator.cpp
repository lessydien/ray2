/**\file randomGenerator.cpp
* \brief collection of functions for generating random numbers. The functions have been moved to the header and have been declared inline so they can be used on the GPU as well
* 
*           
* \author Mauch
*/

/**************************   mother.cpp   ************************************
* Author:        Agner Fog
* Date created:  1999
* Last modified: 2008-11-16
* Project:       randomc.h
* Platform:      This implementation uses 64-bit integers for intermediate calculations.
*                Works only on compilers that support 64-bit integers.
* Description:
* Random Number generator of type 'Mother-Of-All generator'.
*
* This is a multiply-with-carry type of random number generator
* invented by George Marsaglia.  The algorithm is:             
* S = 2111111111*X[n-4] + 1492*X[n-3] + 1776*X[n-2] + 5115*X[n-1] + C
* X[n] = S modulo 2^32
* C = floor(S / 2^32)
*
* Further documentation:
* The file ran-instructions.pdf contains further documentation and 
* instructions.
*
* Copyright 1999-2008 by Agner Fog. 
* GNU General Public License http://www.gnu.org/licenses/gpl.html
******************************************************************************/


