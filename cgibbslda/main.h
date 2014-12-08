/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "cgibbslda.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
//#include <gsl/gsl_rng.h>
#include <math.h>


#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
const gsl_rng_type * T;
gsl_rng * r;
double CONVERGED;
int MAXITER;
int NUMC;
int MCSIZE;
int NUMINIT;
double *gt;
int NTOP;
int update_all;
int freeze_iter;

cgibbslda_ss * new_cgibbslda_ss(cgibbslda_model* model);
cgibbslda_model* new_cgibbslda_model(int ntopics, int nterms, int ndocs, double alpha, double nu);


void train(char* dataset, int ntopics, char* start, char* dir, double alpha, double nu, char* model_name);

int main(int argc, char* argv[]);
void write_cgibbslda_model(cgibbslda_model * model, char * root, cgibbslda_corpus * corpus);
void corpus_initialize_model(cgibbslda_model* model, cgibbslda_corpus* corpus, cgibbslda_ss* ss);
int max_corpus_length(cgibbslda_corpus* c);
cgibbslda_corpus* read_data(const char* data_filename);

void mcsample(cgibbslda_corpus* corpus, cgibbslda_model* model, cgibbslda_ss* ss,
		double* lhood0, double* lhood1, int testchk);

cgibbslda_model* load_model(char* model_root, int ndocs);
void write_pred_time(cgibbslda_corpus* corpus, char * filename);
void read_time(cgibbslda_corpus* corpus, char * filename);

//void corpus_initialize_model(cgibbslda_var* alpha, cgibbslda_model* model, cgibbslda_corpus* c);//, gsl_rng * r);
void test(char* dataset, char* model_name, char* dir);
void random_initialize_model(cgibbslda_model * model, cgibbslda_corpus* corpus, cgibbslda_ss* ss);
void write_word_assignment(cgibbslda_corpus* c,char * filename, cgibbslda_model* model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
