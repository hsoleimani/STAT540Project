#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char model_name[400];
	char init[400];
	int ntopics;
	double alpha;
	double nu;
	long int seed;

	seed = atoi(argv[1]);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);

	MAXITER = 100;
	CONVERGED = 1e-4;
	NUMINIT = 10;

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

    if (argc > 1)
    {
        if (strcmp(task, "est")==0)
        {
        	ntopics = atoi(argv[4]);
			strcpy(init,argv[5]);
			strcpy(dir,argv[6]);
			if ((strcmp(init,"loadbeta")==0))
				strcpy(model_name,argv[7]);
			train(corpus_file, ntopics, init, dir, model_name);

			gsl_rng_free (r);
            return(0);
        }
        if (strcmp(task, "inf")==0)
        {
			strcpy(model_name,argv[4]);
			strcpy(dir,argv[5]);
			test(corpus_file, model_name, dir);

			gsl_rng_free (r);
            return(0);
        }
    }
    return(0);
}

void train(char* dataset, int ntopics, char* start, char* dir, char* model_name)
{
    FILE* lhood_fptr;
    FILE* fp;
    char string[100];
    char filename[100];
    int iteration;
	double lhood, prev_lhood, conv, doclkh;
	double y;
	int d, n, j;
    mlelda_corpus* corpus;
    mlelda_model *model = NULL;
    mlelda_ss* ss = NULL;
    time_t t1,t2;
    //FILE* fileptr;
    //float x;
    //double y;

    corpus = read_data(dataset, ntopics);


    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    if (strcmp(start, "seeded")==0){  //not updated
    	printf("seeded\n");
    	model = new_mlelda_model(ntopics, corpus->nterms, corpus->ndocs);
		ss = new_mlelda_ss(model);
		//random_initialize_model(mc, model, corpus, ss);

		corpus_initialize_model(model, corpus, ss);
		//corpus_initialize_model2(var, model, corpus);

    }
    else if (strcmp(start, "random")==0){
    	printf("random\n");
    	model = new_mlelda_model(ntopics, corpus->nterms, corpus->ndocs);
		ss = new_mlelda_ss(model);

		random_initialize_model(model, corpus, ss);

    }
    else if (strcmp(start, "loadbeta")==0){ //old
    	printf("load beta\n");
    	model = new_mlelda_model(ntopics, corpus->nterms, corpus->ndocs);
		ss = new_mlelda_ss(model);
	    sprintf(filename, "%s.beta", model_name);
	    printf("%s\n",filename);
	    fp = fopen(filename, "r");
	    for (n = 0; n < model->n; n++){
	    	for (j = 0; j < model->m; j++){
				fscanf(fp, " %lf", &y);
				model->beta[j][n] = y;
	    	}
	    }
	    fclose(fp);
    }


	//zero init tss
	for (j = 0; j < model->m; j++){
		ss->sumbeta[j] = 0.0;
		for (n = 0; n < model->n; n++){
			ss->beta[j][n] = 0;
			ss->sumbeta[j] += ss->beta[j][n];
		}
	}
    iteration = 0;
    sprintf(filename, "%s/%03d", dir,iteration);
    printf("%s\n",filename);
	write_mlelda_model(model, filename, corpus);

    time(&t1);
	prev_lhood = -1e100;

	do{

		printf("***** EM ITERATION %d *****\n", iteration);
		lhood = 0.0;
		for (d = 0; d < corpus->ndocs; d++){
			ss->sumtheta = 0.0;
			for (j = 0; j < model->m; j++){
				ss->theta[j][d] = 0.0;
			}
			doclkh = doc_inference(corpus, model, ss, d, 0);
			lhood += doclkh;
		}

		//update beta
		for (j = 0; j < model->m; j++){
			for (n = 0; n < model->n; n++){
				model->beta[j][n] = ss->beta[j][n]/ss->sumbeta[j];
				if (model->beta[j][n] == 0)	model->beta[j][n] = EPS;
				ss->beta[j][n] = 0.0;
			}
			ss->sumbeta[j] = 0.0;
		}

		conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);
		prev_lhood = lhood;

		if (prev_lhood > lhood){
			printf("oops ... \n");
		}
		time(&t2);

		sprintf(filename, "%s/%03d", dir,1);
		write_mlelda_model(model, filename, corpus);

		fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (int)t2-t1);
		fflush(lhood_fptr);
		iteration ++;

	}while((iteration < MAXITER) && (conv > CONVERGED));
	fclose(lhood_fptr);

    sprintf(filename, "%s/final", dir);
    write_mlelda_model(model, filename, corpus);

}

double doc_inference(mlelda_corpus* corpus, mlelda_model* model, mlelda_ss* ss, int d, int test){

	int n, j, w;
	double c, phisum, temp, cphi;
	double varlkh;

	varlkh = 0.0;
	for (n = 0; n < corpus->docs[d].length; n++){
		w = corpus->docs[d].words[n];
		c = (double) corpus->docs[d].counts[n];

		phisum = 0.0;
		for (j = 0; j < model->m; j++){

			corpus->docs[d].phi[n][j] = model->theta[j][d]*model->beta[j][w];
			phisum += corpus->docs[d].phi[n][j];
		}
		for (j = 0; j < model->m; j++){

			corpus->docs[d].phi[n][j] /= phisum;

			temp = c*corpus->docs[d].phi[n][j];
			ss->theta[j][d] += temp;
			ss->sumtheta += temp;
			if (test == 0){
				ss->beta[j][w] += temp;
				ss->sumbeta[j] += temp;
			}

			if (corpus->docs[d].phi[n][j] > 0){
				cphi = c*corpus->docs[d].phi[n][j];
				varlkh += cphi*(log(model->beta[j][w]) + log(model->theta[j][d]) -log(corpus->docs[d].phi[n][j]));
			}
		}
	}
	for (j = 0; j < model->m; j++){
		model->theta[j][d] = ss->theta[j][d]/ss->sumtheta;
		if (model->theta[j][d] == 0)	model->theta[j][d] = EPS;
	}

	return(varlkh);

}





void test(char* dataset, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	int iteration;
	int d, n, j, doclkh, ntopics;
	double lhood;
	//double sumt;

	mlelda_corpus* corpus;
	mlelda_model *model = NULL;
	mlelda_ss* ss = NULL;
	time_t t1,t2;
	//float x;
	//double y;

	sprintf(filename, "%s.other", model_name);
	printf("loading %s\n", filename);
	fp = fopen(filename, "r");
	fscanf(fp, "num_topics %d\n", &ntopics);
	fclose(fp);


	corpus = read_data(dataset, ntopics);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file

	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = load_model(model_name, corpus->ndocs);
	ss = new_mlelda_ss(model);

	//*************************************
	double sum;
	for (j = 0; j < model->m; j++){
		sum = 0.0;
		for (n = 0; n < model->n; n++){
			sum += model->beta[j][n];
		}
		for (n = 0; n < model->n; n++){
			model->beta[j][n] /= sum;
		}
	}

    iteration = 0;
    sprintf(filename, "%s/test%03d", dir,iteration);
    printf("%s\n",filename);
	write_mlelda_model(model, filename, corpus);

	double prev_lhood, conv;
    time(&t1);

	lhood = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = 1.0/((double)model->m);
		}
		prev_lhood = -1e100;
		iteration = 0;
		do{
			ss->sumtheta = 0.0;
			for (j = 0; j < model->m; j++){
				ss->theta[j][d] = 0.0;
			}
			doclkh = doc_inference(corpus, model, ss, d, 1);

			conv = fabs(prev_lhood - doclkh)/fabs(prev_lhood);
			prev_lhood = doclkh;

			if (prev_lhood > doclkh){
				printf("oops ... \n");
			}
			time(&t2);
			iteration ++;

		}while((iteration < MAXITER) && (conv > CONVERGED));
		lhood += doclkh;
	}

	time(&t2);

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, 0.0, (int)t2-t1);
	fflush(lhood_fptr);
	fclose(lhood_fptr);
	//*************************************

	sprintf(filename, "%s/testfinal", dir);
	write_mlelda_model(model, filename, corpus);

}


mlelda_model* new_mlelda_model(int ntopics, int nterms, int ndocs)
{
	int n, j, d;

	mlelda_model* model = malloc(sizeof(mlelda_model));
	model->D = ndocs;
    model->m = ntopics;
    model->n = nterms;
    model->beta = malloc(sizeof(double*)*ntopics);
    model->theta = malloc(sizeof(double*)*ntopics);
    for (j = 0; j < ntopics; j++){
		model->beta[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->beta[j][n] = 0.0;
		}
		model->theta[j] = malloc(sizeof(double)*ndocs);
		for (d = 0; d < ndocs; d++){
			model->theta[j][d] = 0.0;
		}
	}

    return(model);
}


/*
 * create and delete sufficient statistics
 *
 */

mlelda_ss * new_mlelda_ss(mlelda_model* model)
{
	int j, n, d;
	mlelda_ss * ss;
    ss = malloc(sizeof(mlelda_ss));
    ss->sumtheta = 0.0;
	ss->beta = malloc(sizeof(double*)*model->m);
	ss->theta = malloc(sizeof(double*)*model->m);
	ss->sumbeta = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		ss->sumbeta[j] = 0.0;
		ss->beta[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			ss->beta[j][n] = 0.0;
		}
		ss->theta[j] = malloc(sizeof(double)*model->D);
		for (d = 0; d < model->D; d++){
			ss->theta[j][d] = 0.0;
		}
	}

    return(ss);
}



mlelda_corpus* read_data(const char* data_filename, int ntopics)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw, j;
	mlelda_corpus* c;

	printf("reading data from %s\n", data_filename);
	c = malloc(sizeof(mlelda_corpus));
	c->docs = 0;
	c->nterms = 0;
	c->ndocs = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d", &length) != EOF)){
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].words = malloc(sizeof(int)*length);
		c->docs[nd].counts = malloc(sizeof(int)*length);
		c->docs[nd].phi = malloc(sizeof(double*)*length);
		for (n = 0; n < length; n++){
			fscanf(fileptr, "%10d:%10d", &word, &count);
			c->docs[nd].words[n] = word;
			c->docs[nd].counts[n] = count;
			c->docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
			c->docs[nd].phi[n] = malloc(sizeof(double*)*ntopics);
			for (j = 0; j < ntopics; j++)
				c->docs[nd].phi[n][j] = 0.0;
		}
		nd++;
	}
	fclose(fileptr);
	c->ndocs = nd;
	c->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);
	return(c);
}

int max_corpus_length(mlelda_corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->ndocs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}


void write_mlelda_model(mlelda_model * model, char * root,mlelda_corpus * corpus)
{
    char filename[200];
    FILE* fileptr;
    int n, j, d;

    //beta
    sprintf(filename, "%s.beta", root);
    fileptr = fopen(filename, "w");
    for (n = 0; n < model->n; n++){
    	for (j = 0; j < model->m; j++){
    		fprintf(fileptr, "%.10lf ",log(model->beta[j][n]));
    	}
    	fprintf(fileptr, "\n");
    }
    fclose(fileptr);

    //theta
	sprintf(filename, "%s.theta", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%5.10lf ", log(model->theta[j][d]));
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);


	sprintf(filename, "%s.other", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr,"num_topics %d \n",model->m);
	fprintf(fileptr,"num_terms %d \n",model->n);
	fprintf(fileptr,"num_docs %d \n",model->D);
	fclose(fileptr);

}

mlelda_model* load_model(char* model_root, int ndocs){

	char filename[100];
	FILE* fileptr;
	int j, n, num_topics, num_terms, num_docs;
	//float x;
	double y;

	mlelda_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fclose(fileptr);

	model  = new_mlelda_model(num_topics, num_terms, ndocs);
	model->n = num_terms;
	model->m = num_topics;
	model->D = ndocs;


	sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (n = 0; n < num_terms; n++){
		for (j = 0; j < num_topics; j++){
			fscanf(fileptr, " %lf", &y);
			model->beta[j][n] = exp(y);
		}
	}
    fclose(fileptr);

    return(model);
}


void corpus_initialize_model(mlelda_model* model, mlelda_corpus* corpus, mlelda_ss* ss)
{

	int n, j, d, i, count;
	//double sum, temp, maxlkh;
	double sum;
	int* sdocs = malloc(sizeof(int)*corpus->ndocs);
	for (d = 0; d < corpus->ndocs; d++){
		sdocs[d] = -1;
	}
	/*for (j = 0; j < model->m; j++){
		theta[j] = 0.0;
		z[j] = 0;
		p[j] = 0.0;
		alpha[j] = 0.0;
	}
	for (n = 0; n < model->n; n++){
		beta[n] = 0.0;
		nu[n] = 0.0;
	}*/

	//init topics locally
	for (j = 0; j < model->m; j++){
		//maxcnt = -10;
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = 0.1;
		}
		for (i = 0; i < NUM_INIT; i++){
			//choose a doc from this tim and init
			count = 0;
			while (1){
				d = floor(gsl_rng_uniform(r) * corpus->ndocs);
				if(sdocs[d] != -1){
					count ++;
					continue;
				}
				else{
					sdocs[d] = j;
					break;
				}
			}

			for (n = 0; n < corpus->docs[d].length; n++){
				model->beta[j][corpus->docs[d].words[n]] += (double) corpus->docs[d].counts[n];
			}
		}
		sum = 0.0;
		for (n = 0; n < model->n; n++){
			sum += model->beta[j][n];
		}
		for (n = 0; n < model->n; n++){
			model->beta[j][n] /= sum;
		}
		for (d = 0; d < corpus->ndocs; d++){
			model->theta[j][d] = 1.0/((double)model->m);
		}
	}

	//hard assign other docs to topics
	/*for (d = 0; d < corpus->ndocs; d++){
		if (sdocs[d] != -1)
			continue;
		maxlkh = -1e50;
		argmaxlkh = 0;
		for (j = 0; j < model->m; j++){
			temp = 0.0;
			for (n = 0; n < corpus->docs[d].total; n++){
				temp += log(model->beta[j][corpus->docs[d].words[n]]);
			}
			if (temp > maxlkh){
				maxlkh = temp;
				argmaxlkh = j;
			}
		}
		j = argmaxlkh;
		for (n = 0; n < corpus->docs[d].total; n++){
			corpus->docs[d].z[corpus->docs[d].words[n]] = j;
			corpus->docs[d].mcz[corpus->docs[d].words[n]] = j;
			ss->m[j][d] += 1.0;
			ss->t[j][corpus->docs[d].words[n]] += 1.0;
		}
	}*/



  	free(sdocs);


}



void random_initialize_model(mlelda_model * model, mlelda_corpus* corpus, mlelda_ss* ss){

	int n, j, d;
	double sum;

	for (j = 0; j < model->m; j++){
		sum = 0.0;
		for (n = 0; n < model->n; n++){
			//model->mu[j][n] = beta[n];
			//model->mu[j][n] = gsl_ran_gamma(r, 100.0, 1.0/100.);
			model->beta[j][n] = 0.1 + gsl_rng_uniform(r);
			sum += model->beta[j][n];
		}

		for (n = 0; n < model->n; n++){
			model->beta[j][n] /= sum;
		}
		for (d = 0; d < corpus->ndocs; d++){
			model->theta[j][d] = 1.0/((double)model->m);
		}
	}
}



/*
 * given log(a) and log(b), return log(a + b)
 *
 */

double log_sum(double log_a, double log_b)
{
  double v;

  if (log_a < log_b)
  {
      v = log_b+log(1 + exp(log_a-log_b));
  }
  else
  {
      v = log_a+log(1 + exp(log_b-log_a));
  }
  return(v);
}

