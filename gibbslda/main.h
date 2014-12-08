/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "gibbslda.h"
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

gibbslda_ss * new_gibbslda_ss(gibbslda_model* model);
gibbslda_model* new_gibbslda_model(int ntopics, int nterms, int ndocs, double alpha, double nu);


void train(char* dataset, int ntopics, char* start, char* dir, double alpha, double nu, char* model_name);

int main(int argc, char* argv[]);
void write_gibbslda_model(gibbslda_model * model, char * root, gibbslda_corpus * corpus, gibbslda_mc* mc);
void corpus_initialize_model(gibbslda_mc* mc, gibbslda_model* model, gibbslda_corpus* corpus, gibbslda_ss* ss);
gibbslda_mc * new_gibbslda_mc(int ndocs, int ntopics, int nterms);
int max_corpus_length(gibbslda_corpus* c);
gibbslda_corpus* read_data(const char* data_filename);

void mcsample(gibbslda_corpus* corpus, gibbslda_model* model, gibbslda_ss* ss, gibbslda_mc* mc,
		double* lhood0, double* lhood1, int testchk);

gibbslda_model* load_model(char* model_root, int ndocs);
void write_pred_time(gibbslda_corpus* corpus, char * filename);
void read_time(gibbslda_corpus* corpus, char * filename);

//void corpus_initialize_model(gibbslda_var* alpha, gibbslda_model* model, gibbslda_corpus* c);//, gsl_rng * r);
void test(char* dataset, char* model_name, char* dir);
void random_initialize_model(gibbslda_mc * mc, gibbslda_model * model, gibbslda_corpus* corpus, gibbslda_ss* ss);
void write_word_assignment(gibbslda_corpus* c,char * filename, gibbslda_model* model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
