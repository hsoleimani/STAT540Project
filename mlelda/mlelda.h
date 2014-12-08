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
#include <gsl/gsl_sf_psi.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-50;
#define PI 3.14159265359
#define max(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })



typedef struct
{
    int* words;
    int* counts;
    int length;
    int total;
    double** phi;
} document;


typedef struct
{
    document* docs;
    int nterms;
    int ndocs;
} mlelda_corpus;


typedef struct mlelda_model
{
    int m;
    int D;
    int n;
    double** beta;
    double** theta;
} mlelda_model;


typedef struct mlelda_ss
{
    double** theta;
    double** beta;
    double* sumbeta;
    double sumtheta;
    //double* psimu;
} mlelda_ss;





#endif /* SPARSELDA_H_ */
