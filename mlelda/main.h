/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "mlelda.h"
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

mlelda_ss * new_mlelda_ss(mlelda_model* model);
mlelda_model* new_mlelda_model(int ntopics, int nterms, int ndocs);


void train(char* dataset, int ntopics, char* start, char* dir, char* model_name);

int main(int argc, char* argv[]);
void write_mlelda_model(mlelda_model * model, char * root, mlelda_corpus * corpus);
void corpus_initialize_model(mlelda_model* model, mlelda_corpus* corpus, mlelda_ss* ss);
int max_corpus_length(mlelda_corpus* c);
mlelda_corpus* read_data(const char* data_filename, int ntopics);

double doc_inference(mlelda_corpus* corpus, mlelda_model* model, mlelda_ss* ss, int d, int test);

mlelda_model* load_model(char* model_root, int ndocs);
void write_pred_time(mlelda_corpus* corpus, char * filename);
void read_time(mlelda_corpus* corpus, char * filename);

//void corpus_initialize_model(mlelda_var* alpha, mlelda_model* model, mlelda_corpus* c);//, gsl_rng * r);
void test(char* dataset, char* model_name, char* dir);
void random_initialize_model(mlelda_model * model, mlelda_corpus* corpus, mlelda_ss* ss);
void write_word_assignment(mlelda_corpus* c,char * filename, mlelda_model* model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
