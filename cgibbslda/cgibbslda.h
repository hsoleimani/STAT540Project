/*
 * sparseLDA.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef SPARSELDA_H_
#define SPARSELDA_H_

//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-30
#define PI 3.14159265359
#define max(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })



typedef struct
{
    int* words;
    int* counts;
    int length;
    int total;
    int* z;
    double* mcz;
    double* sz2;
    double* yz;
} cgibbslda_document;


typedef struct
{
	cgibbslda_document* docs;
    int nterms;
    int ndocs;
} cgibbslda_corpus;


typedef struct cgibbslda_model
{
    int m;
    int D;
    int n;
    double** beta;
    double** theta;
    double** mcbeta;
    double** mctheta;
    double alpha;
    double nu;
} cgibbslda_model;


typedef struct cgibbslda_ss
{
    double** m;
    double** t;
    double* sumt;
    double n;
    int b;
    double a;
    int change;
    double mcse;
    double mcse_num;
} cgibbslda_ss;






#endif /* SPARSELDA_H_ */
