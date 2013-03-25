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

/* $ITO: complex.c 1.11 1998/04/27 10:29:40 haible Exp $ */
/* M.Fleischer 1/95 */

/*
 * Alle Funktionen mit Polen sind so definiert, dass die Schnittstellen vom
 * Ursprung weg in gerader Linie ins Unendliche gehen. Ist der Urspung selber
 * ein Pol, so geht die Schnittstelle nach -oo.
 */
/*Definitionen und Funktionalität für komplexe Zahlen*/

#include "complex.h"
//#include "mlib.h"
#include <math.h>
//#include "malloc.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/****** mlib/Complex ******
*
*  NAME
*    d_abs, d_arg, ...
*  AUFRUF
*    # include "complex.h"
*    double    d_abs(complex_t a);
*    double    d_arg(complex_t a);
*    complex_t c_abs(complex_t a);
*    complex_t c_arg(complex_t a);
*    complex_t c_add(complex_t a,complex_t b);
*    complex_t c_sub(complex_t a,complex_t b);
*    complex_t c_mul(complex_t a,complex_t b);
*    complex_t c_div(complex_t a,complex_t b);
*    complex_t c_exp(complex_t a);
*    complex_t c_log(complex_t a);
*    complex_t c_log10(complex_t a);
*    complex_t c_log2(complex_t a);
*    complex_t c_pow(complex_t a,complex_t b);
*    complex_t c_sin(complex_t a);
*    complex_t c_cos(complex_t a);
*    complex_t c_tan(complex_t a);
*    complex_t c_cot(complex_t a);
*    complex_t c_asin(complex_t a);
*    complex_t c_acos(complex_t a);
*    complex_t c_atan(complex_t a);
*    complex_t c_atan2(complex_t y, complex_t x);
*    complex_t c_acot(complex_t a);
*    complex_t c_sinh(complex_t a);
*    complex_t c_cosh(complex_t a);
*    complex_t c_tanh(complex_t a);
*    complex_t c_coth(complex_t a);
*    complex_t c_arsinh(complex_t a);
*    complex_t c_arcosh(complex_t a);
*    complex_t c_artanh(complex_t a);
*    complex_t c_arcoth(complex_t a);
*    complex_t c_sqrt(complex_t a);
*    complex_t c_cbrt(complex_t a);
*    complex_t c_sqr(complex_t a);
*    complex_t c_ln_gamma(complex_t a);
*  FUNKTION
*    Berechnet die entsprechenden mathematischen Funktionen für komplexe
*    Argumente
*  ANMERKUNG
*
*  SIEHE AUCH
*
***/


/* d_abs() und d_arg() sind im Normalfall als Makros definiert */
/* --> hier die Namen in Klammern damit die Makros nicht wirken */
double (d_abs)(complex_t a) /* Naja */
{
  return sqrt(a.real*a.real+a.imag*a.imag);
}

double (d_arg)(complex_t a) /* Naja */
{
  return atan2(a.imag,a.real);
}


complex_t c_init(double real, double imag)
{
  C_RETURN(real,imag);
}

complex_t c_abs(complex_t a)
{
  C_RETURN(sqrt(a.real*a.real+a.imag*a.imag),0);
}

complex_t c_arg(complex_t a)
{
  C_RETURN(atan2(a.imag,a.real),0);
}

complex_t c_add(complex_t a,complex_t b)
{
  C_RETURN(a.real+b.real,a.imag+b.imag);
}

complex_t c_sub(complex_t a,complex_t b)
{
  C_RETURN(a.real-b.real,a.imag-b.imag);
}

/* (a+ib)(c+id)=ac-bd+iad+ibc */
//hier eine Testversion ohne Pointer
int c_mul_test(double par)
{
 // struct real_imag_s result;
  //result=(complex_t *)malloc(sizeof(complex_t *));
//	int k=(par==10);
/*  
  result->real=a->real*b->real-a->imag*b->imag;
  result->imag=a->real*b->imag+a->imag*b->real;
  */
  return 0;
  //C_RETURN(a.real*b.real-a.imag*b.imag,a.real*b.imag+a.imag*b.real);
}



complex_t c_mul(complex_t a,complex_t b)
{
  C_RETURN(a.real*b.real-a.imag*b.imag,a.real*b.imag+a.imag*b.real);
}


/*
 * a+ib   (a+ib)(c-id)   ac+bd-iad+ibc
 * ---- = ------------ = -------------
 * c+id   (c+id)(c-id)      cc+dd
 */
complex_t c_div(complex_t a,complex_t b)
{
  double n=b.real*b.real+b.imag*b.imag;
  C_RETURN((a.real*b.real+a.imag*b.imag)/n,(a.imag*b.real-a.real*b.imag)/n);
}

/* e^(a+ib)=e^a (cos(b)+i sin(b)) */
complex_t c_exp(complex_t a)
{
  double e=exp(a.real);
  C_RETURN(e*cos(a.imag),e*sin(a.imag));
}

/*
 * ln(x)=ln|x|+i arg x :
 *
 * e^y=e^(ln|x|+i arg x)=|x|(cos arg x + i sin arg x)=x QED
 */
complex_t c_log(complex_t a)
{
  C_RETURN(log(sqrt(a.real*a.real+a.imag*a.imag)),atan2(a.imag,a.real));
}

/*
 * log10(x)=ln(x)/ln(10) :
 *
 *    x =10^y=(e^ln(10))^y=e^(ln(10)y)
 * ln(x)=                     ln(10)y
 */
complex_t c_log10(complex_t a)
{
  static double l = 2.302585092994045684018; /* ln10 */
  C_RETURN(log(sqrt(a.real*a.real+a.imag*a.imag))/l,atan2(a.imag,a.real)/l);
}

/*
 * log2(x)=ln(x)/ln(2) :
 */
complex_t c_log2(complex_t a)
{
  static double l = 0.693147180559945309422; /* ln2 */
  C_RETURN(log(sqrt(a.real*a.real+a.imag*a.imag))/l,atan2(a.imag,a.real)/l);
}

/* a^b=e^(b*ln(a)) */
complex_t c_pow(complex_t a,complex_t b)
{
  if(!a.real&&!a.imag)
    return a;
  return c_exp(c_mul(b,c_log(a)));
}

/* sin(x) = .5/i (e^(ix)-e^(-ix)) = -i sinh(ix) */
complex_t c_sin(complex_t a)
{
  double e1,e2;
  e1=exp(a.imag);
  e2=exp(-a.imag);
  C_RETURN(.5*sin(a.real)*(e1+e2),.5*cos(a.real)*(e1-e2));
}

/* cos(x) = .5 (e^(ix)+e^(-ix)) = cosh(ix) */
complex_t c_cos(complex_t a)
{
  double e1,e2;
  e1=exp(a.imag);
  e2=exp(-a.imag);
  C_RETURN(.5*cos(a.real)*(e2+e1),.5*sin(a.real)*(e2-e1));
}

/* tan(x)=sin(x)/cos(x) */
complex_t c_tan(complex_t a)
{
  return c_div(c_sin(a),c_cos(a));
}

/* cot(x)=cos(x)/sin(x) */
complex_t c_cot(complex_t a)
{
  return c_div(c_cos(a),c_sin(a));
}

/*
 * arcsin(x)=-i arsinh(ix) :
 *
 *         x =  sin(y)
 *        ix =i sin(y)=sinh(iy)
 * arsinh(ix)=              iy
 */
complex_t c_asin(complex_t a)
{
  complex_t b;
  b.real=-a.imag;
  b.imag= a.real;
  b=c_arsinh(b);
  C_RETURN(b.imag,-b.real);
}

/*
 * arccos(x)=-i arcosh(x) :
 *
 *        x =cos(y)=cosh(iy)
 * arcosh(x)=            iy
 */
complex_t c_acos(complex_t a)
{
  a=c_arcosh(a);
  C_RETURN(a.imag,-a.real);
}

/*
 * arctan(x)=-i artanh(ix) :
 *
 *         x =  tan(y)
 *        ix =i tan(y)=tanh(iy)
 * artanh(ix)=              iy
 */
complex_t c_atan(complex_t a)
{
  complex_t b;
  b.real=-a.imag;
  b.imag= a.real;
  b=c_artanh(b);
  C_RETURN(b.imag,-b.real);
}

complex_t c_atan2(complex_t y, complex_t x)
{
	/* Bereichsangaben aus der reellen Analogie */
	complex_t c;
	double absx, absy;
	absx = d_abs(x);
	absy = d_abs(y);
	if (absx >= absy) {
		c = c_atan(c_div(y,x));
		if (x.real > 0) {
			/* -Pi/4 .. Pi/4 */
			/* fall through */
		} else {
			if (y.real >= 0) {
				/* 3/4 Pi .. Pi */
				c.real += PI;
			} else {
				/* -Pi .. -3/4 Pi */
				c.real -= PI;
			}
		}
	} else {
		c = c_atan(c_div(x,y));
		c.real = -c.real;
		c.imag = -c.imag;
		if (y.real > 0) {
			/* Pi/4 .. 3/4 Pi */
			c.real += PI/2;
		} else {
			/* -3/4 Pi .. -Pi/4 */
			c.real -= PI/2;
		}
	}
	return c;
}

/*
 * arccot(x)=-i arcoth(-ix) :
 *
 *          x =   cot(y)
 *        -ix =-i cot(y)=coth(iy)
 * arcoth(-ix)=               iy
 */
complex_t c_acot(complex_t a)
{
  complex_t b;
  b.real= a.imag;
  b.imag=-a.real;
  b=c_arcoth(b);
  C_RETURN(b.imag,-b.real);
}

/* sinh(x)=1/2 (e^x - e^-x) */
complex_t c_sinh(complex_t a)
{
  double e1,e2;
  e1=exp(a.real);
  e2=exp(-a.real);
  C_RETURN(.5*(e1-e2)*cos(a.imag),.5*(e1+e2)*sin(a.imag));
}

/* cosh(x)=1/2 (e^x + e^-x) */
complex_t c_cosh(complex_t a)
{
  double e1,e2;
  e1=exp(a.real);
  e2=exp(-a.real);
  C_RETURN(.5*(e1+e2)*cos(a.imag),.5*(e1-e2)*sin(a.imag));
}

/* tanh(x)=sinh(x)/cosh(x) */
complex_t c_tanh(complex_t a)
{
  return c_div(c_sinh(a),c_cosh(a));
}

/* coth(x)=cosh(x)/sinh(x) */
complex_t c_coth(complex_t a)
{
  return c_div(c_cosh(a),c_sinh(a));
}

/*
 * arsinh(x)=ln(x+sqrt(x^2+1)) :
 *
 * x=sinh(y)=1/2 (e^y - e^-y)
 * Subst. e^y=u; e^-y=1/u; y=ln(u), nach y auflösen
 */
complex_t c_arsinh(complex_t a)
{
  complex_t b;
  b=c_sqr(a);
  b.real+=1;
  return c_log(c_add(a,c_sqrt(b)));
}

/*
 * arcosh(x)=ln(x+sqrt(x^2-1)) :
 *
 * x=cosh(y)=1/2 (e^y + e^-y), dann s.o.
 */
complex_t c_arcosh(complex_t a)
{
  complex_t b;
  b=c_sqr(a);
  b.real-=1;
  return c_log(c_add(a,c_sqrt(b)));
}

/*
 * artanh(x)=1/2 ln((1+x)/(1-x)) :
 *
 * x=tanh(y)=sinh(y)/cosh(y), dann s.o.
 */
complex_t c_artanh(complex_t a)
{
  complex_t b,c,d;
  b.real=1+a.real;
  b.imag=  a.imag;
  c.real=1-a.real;
  c.imag= -a.imag;
  d=c_log(c_div(b,c));
  C_RETURN(d.real/2,d.imag/2);
}

/*
 * arcoth(x)=1/2 ln((x+1)/(x-1)) :
 *
 * x=coth(y)=cosh(y)/sinh(y), dann s.o.
 */
extern complex_t c_arcoth(complex_t a)
{
  complex_t b,c,d;
  b.real=a.real+1;
  b.imag=a.imag;
  c.real=a.real-1;
  c.imag=a.imag;
  d=c_log(c_div(b,c));
  C_RETURN(d.real/2,d.imag/2);
}

/*
 * sqrt(x)=sqrt|x|e^(1/2 i arg x) :
 *
 * yy=|x|e^(i arg x)=x
 */
complex_t c_sqrt(complex_t a)
{
  double b,w;
  b=sqrt(sqrt(a.real*a.real+a.imag*a.imag));
  w=atan2(a.imag,a.real)/2;
  C_RETURN(b*cos(w),b*sin(w));
}

/*
 * cbrt(x)=cbrt|x|e^(1/3 i arg x) :
 *
 * yyy=|x|e^(i arg x)=x
 */
complex_t c_cbrt(complex_t a)
{
  double b,w;
  b=pow(a.real*a.real+a.imag*a.imag,1.0/6.0);
  w=atan2(a.imag,a.real)/3;
  C_RETURN(b*cos(w),b*sin(w));
}

complex_t c_sqr(complex_t a) /* (a+ib)^2=aa-bb+2iab */
{
  C_RETURN(a.real*a.real-a.imag*a.imag,2*a.real*a.imag);
}

/*
 * ln_gamma(x) = ln((x-1)!) = int(0,inf) t^(x-1) exp(-t) dt
 * Siehe Numerical Recipes, s. 214
 */
complex_t c_ln_gamma(complex_t a)
{
  complex_t x,y,t,s;
  static const double cof[]=
  { 76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, .1208650973866179e-2, -.5395239384953e-5 };
  int j;

  y=x=t=a;
  t.real+=5.5;
  a.real+=.5;
  t=c_sub(t,c_mul(a,c_log(t)));
  s.real=1.000000000190015;
  s.imag=0;
  for(j=0;j<6;j++)
  {
    double z;
    y.real++;
    z=y.real*y.real+y.imag*y.imag;
    s.real+=cof[j]*y.real/z;
    s.imag-=cof[j]*y.imag/z;
  }
  s.real*=2.5066282746310005;
  s.imag*=2.5066282746310005;
  return c_sub(c_log(c_div(s,x)),t);
}
