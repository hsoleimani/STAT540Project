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


#define NUM_INIT 10
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
} document;


typedef struct
{
    document* docs;
    int nterms;
    int ndocs;
} gibbslda_corpus;


typedef struct gibbslda_model
{
    int m;
    int D;
    int n;
    double** beta;
    double** theta;
    double alpha;
    double nu;
} gibbslda_model;


typedef struct gibbslda_ss
{
    double** m;
    double** t;
} gibbslda_ss;

typedef struct gibbslda_mc
{
    double** beta;
    double** theta;
    double** ybeta;
    double** ytheta;
    double** sbeta2;
    double** stheta2;
    double n;
    int change;
    double mcse;
    double mcse_num;
    int b;
    double a;
} gibbslda_mc;




#endif /* SPARSELDA_H_ */
