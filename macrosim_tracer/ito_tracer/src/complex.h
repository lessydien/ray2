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

/* $ITO: complex.h 1.9 1997/04/17 11:25:42 haible Exp $ */

#ifndef M_COMPLEX_H
#define M_COMPLEX_H
/*
 * Komplexe Mathematik
 */

void testfunc(int m);

/* über type cast kompatibel zu struct complex aus math.h */

#ifndef _REAL_IMAG_S_
#define _REAL_IMAG_S_

struct real_imag_s {
    double real;
    double imag;
};

#endif 

typedef struct real_imag_s complex_t;

#define C_RETURN(a,b)   { complex_t _r; _r.real=a; _r.imag=b; return _r; }

//int c_mul_test(complex_t *a,complex_t *b,complex_t *result);
int c_mul_test(double par);


#define d_real(a) ((a).real)
#define d_imag(a) ((a).imag)

#if 1	/* Vorsicht Nebeneffekte */
#define d_abs(_c) (sqrt(_c.real*_c.real+_c.imag*_c.imag))
#define d_arg(_c) (atan2(_c.imag,_c.real))
#else
double    d_abs(complex_t a);
double    d_arg(complex_t a);
#endif

/* operatoren */
__inline complex_t operator-(complex_t &a)
{
    C_RETURN(-a.real, -a.imag);
}
/* scalar operators */
__inline complex_t operator-(complex_t a, double b)
{
	C_RETURN(a.real-b, a.imag);
}

__inline complex_t operator-(double a, complex_t b)
{
	C_RETURN(a-b.real, -b.imag);
}

__inline complex_t operator+(complex_t a, double b)
{
	C_RETURN(a.real+b, a.imag);
}

__inline complex_t operator+(double a, complex_t b)
{
	C_RETURN(a+b.real, b.imag);
}

__inline complex_t operator*(complex_t a, double b)
{
	C_RETURN(a.real*b, a.imag*b);
}

__inline complex_t operator*(double a, complex_t b)
{
	C_RETURN(a*b.real, a*b.imag);
}

__inline complex_t operator/(complex_t a, double b)
{
	C_RETURN(a.real/b, a.imag/b);
}

__inline complex_t operator/(double a, complex_t b)
{
	double n=b.real*b.real+b.imag*b.imag;
	C_RETURN(a*b.real/n, -a*b.imag/n);
}

/* complex opertaors */
__inline complex_t operator*(complex_t a, complex_t b)
{
	C_RETURN(a.real*b.real-a.imag*b.imag,a.real*b.imag+a.imag*b.real);
}

__inline complex_t operator/(complex_t a, complex_t b)
{
  double n=b.real*b.real+b.imag*b.imag;
  C_RETURN((a.real*b.real+a.imag*b.imag)/n,(a.imag*b.real-a.real*b.imag)/n);
}

__inline complex_t operator-(complex_t a, complex_t b)
{
	C_RETURN(a.real-b.real, a.imag-b.imag);
}

__inline complex_t operator+(complex_t a, complex_t b)
{
	C_RETURN(a.real+b.real, a.imag+b.imag);
}


complex_t c_init(double real, double imag);
complex_t c_abs(complex_t a);
complex_t c_arg(complex_t a);
complex_t c_add(complex_t a,complex_t b);
complex_t c_sub(complex_t a,complex_t b);
complex_t c_mul(complex_t a,complex_t b);
complex_t c_div(complex_t a,complex_t b);
complex_t c_exp(complex_t a);
complex_t c_log(complex_t a);
complex_t c_log10(complex_t a);
complex_t c_log2(complex_t a);
complex_t c_pow(complex_t a,complex_t b);
complex_t c_sin(complex_t a);
complex_t c_cos(complex_t a);
complex_t c_tan(complex_t a);
complex_t c_cot(complex_t a);
complex_t c_asin(complex_t a);
complex_t c_acos(complex_t a);
complex_t c_atan(complex_t a);
complex_t c_atan2(complex_t y, complex_t x);
complex_t c_acot(complex_t a);
complex_t c_sinh(complex_t a);
complex_t c_cosh(complex_t a);
complex_t c_tanh(complex_t a);
complex_t c_coth(complex_t a);
complex_t c_arsinh(complex_t a);
complex_t c_arcosh(complex_t a);
complex_t c_artanh(complex_t a);
complex_t c_arcoth(complex_t a);
complex_t c_sqrt(complex_t a);
complex_t c_cbrt(complex_t a);
complex_t c_sqr(complex_t a);
complex_t c_ln_gamma(complex_t a);

#endif /* M_COMPLEX_H */
