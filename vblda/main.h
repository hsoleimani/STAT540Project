/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "vblda.h"
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

vblda_ss * new_vblda_ss(vblda_model* model);
vblda_model* new_vblda_model(int ntopics, int nterms, int ndocs, double alpha, double nu);


void train(char* dataset, int ntopics, char* start, char* dir, double alpha, double nu, char* model_name);

int main(int argc, char* argv[]);
void write_vblda_model(vblda_model * model, char * root, vblda_corpus * corpus);
void corpus_initialize_model(vblda_model* model, vblda_corpus* corpus, vblda_ss* ss);
int max_corpus_length(vblda_corpus* c);
vblda_corpus* read_data(const char* data_filename, int ntopics);

double doc_inference(vblda_corpus* corpus, vblda_model* model, vblda_ss* ss, int d, int test);

vblda_model* load_model(char* model_root, int ndocs);
void write_pred_time(vblda_corpus* corpus, char * filename);
void read_time(vblda_corpus* corpus, char * filename);

//void corpus_initialize_model(vblda_var* alpha, vblda_model* model, vblda_corpus* c);//, gsl_rng * r);
void test(char* dataset, char* model_name, char* dir);
void random_initialize_model(vblda_model * model, vblda_corpus* corpus, vblda_ss* ss);
void write_word_assignment(vblda_corpus* c,char * filename, vblda_model* model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
