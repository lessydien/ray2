/**\file randomGenerator.h
* \brief 
* 
*           
* \author Mauch
*/

/*****************************   randomc.h   **********************************
* Author:        Agner Fog
* Date created:  1997
* Last modified: 2008-11-16
* Project:       randomc.h
* Source URL:    www.agner.org/random
*
* Description:
* This header file contains class declarations and other definitions for the 
* randomc class library of uniform random number generators in C++ language.
*
* Overview of classes:
* ====================
*
* class CRandomMersenne:
* Random number generator of type Mersenne twister.
* Source file mersenne.cpp
*
* class CRandomMother:
* Random number generator of type Mother-of-All (Multiply with carry).
* Source file mother.cpp
*
* class CRandomSFMT:
* Random number generator of type SIMD-oriented Fast Mersenne Twister.
* The class definition is not included here because it is not
* portable to all platforms. See sfmt.h and sfmt.cpp for details.
*
* Member functions (methods):
* ===========================
*
* All these classes have identical member functions:
*
* Constructor(int seed):
* The seed can be any integer. The time may be used as seed.
* Executing a program twice with the same seed will give the same sequence 
* of random numbers. A different seed will give a different sequence.
*
* void RandomInit(int seed);
* Re-initializes the random number generator with a new seed.
*
* void RandomInitByArray(int const seeds[], int NumSeeds);
* In CRandomMersenne and CRandomSFMT only: Use this function if you want 
* to initialize with a seed with more than 32 bits. All bits in the seeds[]
* array will influence the sequence of random numbers generated. NumSeeds 
* is the number of entries in the seeds[] array.
*
* double Random();
* Gives a floating point random number in the interval 0 <= x < 1.
* The resolution is 32 bits in CRandomMother and CRandomMersenne, and
* 52 bits in CRandomSFMT.
*
* int IRandom(int min, int max);
* Gives an integer random number in the interval min <= x <= max.
* (max-min < MAXINT).
* The precision is 2^-32 (defined as the difference in frequency between 
* possible output values). The frequencies are exact if max-min+1 is a
* power of 2.
*
* int IRandomX(int min, int max);
* Same as IRandom, but exact. In CRandomMersenne and CRandomSFMT only.
* The frequencies of all output values are exactly the same for an 
* infinitely long long sequence. (Only relevant for extremely long long sequences).
*
* uint32_t BRandom();
* Gives 32 random bits. 
*
*
* Example:
* ========
* The file EX-RAN.CPP contains an example of how to generate random numbers.
*
*
* Library version:
* ================
* Optimized versions of these random number generators are provided as function
* libraries in randoma.zip. These function libraries are coded in assembly
* language and support only x86 platforms, including 32-bit and 64-bit
* Windows, Linux, BSD, Mac OS-X (Intel based). Use randoma.h from randoma.zip
*
*
* Non-uniform random number generators:
* =====================================
* Random number generators with various non-uniform distributions are 
* available in stocc.zip (www.agner.org/random).
*
*
* Further documentation:
* ======================
* The file ran-instructions.pdf contains further documentation and 
* instructions for these random number generators.
*
* Copyright 1997-2008 by Agner Fog. 
* GNU General Public License http://www.gnu.org/licenses/gpl.html
*******************************************************************************/

#ifndef RANDOMC_H
#define RANDOMC_H

// Define integer types with known size: int32_t, uint32_t, int64_t, uint64_t.
// If this doesn't work then insert compiler-specific definitions here:
#if defined(__GNUC__)
  // Compilers supporting C99 or C++0x have inttypes.h defining these integer types
  #include <inttypes.h>
  #define INT64_SUPPORTED // Remove this if the compiler doesn't support 64-bit integers
#elif defined(_WIN16) || defined(__MSDOS__) || defined(_MSDOS) 
   // 16 bit systems use long long int for 32 bit integer
  typedef   signed long long int int32_t;
  typedef unsigned long long int uint32_t;
#elif defined(_MSC_VER)
  // Microsoft have their own definition
  typedef   signed __int32  int32_t;
  typedef unsigned __int32 uint32_t;
  typedef   signed __int64  int64_t;
  typedef unsigned __int64 uint64_t;
  #define INT64_SUPPORTED // Remove this if the compiler doesn't support 64-bit integers
#else
  // This works with most compilers
  typedef signed int          int32_t;
  typedef unsigned int       uint32_t;
  typedef long long           int64_t;
  typedef unsigned long long uint64_t;
  #define INT64_SUPPORTED // Remove this if the compiler doesn't support 64-bit integers
#endif

#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE
#include "GlobalConstants.h"
#include "randomGenerator.h"


// Output random bits
inline RT_HOSTDEVICE uint32_t BRandom(uint32_t *x) {
  uint64_t sum;
  //uint32_t x[5];                      // History buffer
  sum = (uint64_t)2111111111UL * (uint64_t)x[3] +
     (uint64_t)1492 * (uint64_t)(x[2]) +
     (uint64_t)1776 * (uint64_t)(x[1]) +
     (uint64_t)5115 * (uint64_t)(x[0]) +
     (uint64_t)x[4];
  x[3] = x[2];  x[2] = x[1];  x[1] = x[0];
  x[4] = (uint32_t)(sum >> 32);                  // Carry
  x[0] = (uint32_t)sum;                          // Low 32 bits of sum
  return x[0];
} 


// returns a random number between 0 and 1:
inline RT_HOSTDEVICE double Random(uint32_t *x) {
   return (double)BRandom(x) * (1./(65536.*65536.));
}


// returns integer random number in desired interval:
inline RT_HOSTDEVICE int IRandom(int min, int max, uint32_t *x) {
   // Output random integer in the interval min <= x <= max
   // Relative error on frequencies < 2^-32
   if (max <= min) {
      if (max == min) return min; else return 0x80000000;
   }
   // Assume 64 bit integers supported. Use multiply and shift method
   uint32_t interval;                  // Length of interval
   uint64_t longran;                   // Random bits * interval
   uint32_t iran;                      // Longran / 2^32

   interval = (uint32_t)(max - min + 1);
   longran  = (uint64_t)BRandom(x) * interval;
   iran = (uint32_t)(longran >> 32);
   // Convert back to signed and return result
   return (int32_t)iran + min;
}


// this function initializes the random number generator:
inline RT_HOSTDEVICE void RandomInit (int seed, uint32_t *x) {
  //uint32_t x[5];                      // History buffer
  int i;
  uint32_t s = seed;
  // make random numbers and put them into the buffer
  for (i = 0; i < 5; i++) {
    s = s * 29943829 - 1;
    x[i] = s;
  }
  // randomize some more
  for (i=0; i<19; i++) BRandom(x);
}

// this function returns gaussian distributed random variable
// see W. H Press, Numerical Recipes in C++ 2nd edition, pp. 293 for reference
inline RT_HOSTDEVICE double RandomGauss(uint32_t *x) {
	double fac, rsq, v1, v2;
	// pick coordinates from the unit circle via rejection method
	//double v1,v2;
	//long long index=0;
	//do
	//{
	//	v1=2.0*Random(x)-1.0;
	//	v2=2.0*Random(x)-1.0;
	//	rsq=v1*v1+v2*v2;
	//	index++;
	//	if (index>1000000)
	//		break;
	//} while ( (rsq >= 1.0) || (rsq == 0.0) );
	
	// pick coordinates from the unit circle via direct method (should be faster than rejection method. At least onGPU)
	double theta=2*PI*Random(x);
	double r=sqrt(Random(x));
	v1=r*cos(theta);
	v2=r*sin(theta);
	rsq=v1*v1+v2*v2;

	fac=sqrt(-2.0*logf(rsq)/rsq);
	return v1*fac;

}

// this function modifed by yang
//inline RT_HOSTDEVICE double RandomCos(uint32_t *x) {
//	double fac, rsq, v1, v2;
	// pick coordinates from the unit circle via rejection method
	//double v1,v2;
	//long long index=0;
	//do
	//{
	//	v1=2.0*Random(x)-1.0;
	//	v2=2.0*Random(x)-1.0;
	//	rsq=v1*v1+v2*v2;
	//	index++;
	//	if (index>1000000)
	//		break;
	//} while ( (rsq >= 1.0) || (rsq == 0.0) );
	
	// pick coordinates from the unit circle via direct method (should be faster than rejection method. At least onGPU)
//	double theta=2*PI*Random(x);
//	double r=sqrt(Random(x));
//	v1=r*cos(theta);
//	v2=r*sin(theta);
//	rsq=v1*v1+v2*v2;

//	fac=sqrt(-2.0*logf(rsq)/rsq);
//	return v1*fac;

//}

inline RT_HOSTDEVICE unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}

#endif // RANDOMC_H
