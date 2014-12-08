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
#define EPS 1e-30
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
} vblda_corpus;


typedef struct vblda_model
{
    int m;
    int D;
    int n;
    double** mu;
    double** psimu;
    double** gamma;
    double alpha;
    double nu;
} vblda_model;


typedef struct vblda_ss
{
    double** m;
    double** t;
    double* oldphi;
    double* summu;
    double* sumgamma;
    //double* psimu;
} vblda_ss;





#endif /* SPARSELDA_H_ */
