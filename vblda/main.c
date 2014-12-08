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
			alpha = atof(argv[7]);
			nu = atof(argv[8]);
			if ((strcmp(init,"loadbeta")==0))
				strcpy(model_name,argv[9]);
			train(corpus_file, ntopics, init, dir, alpha, nu, model_name);

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

void train(char* dataset, int ntopics, char* start, char* dir, double alpha, double nu, char* model_name)
{
    FILE* lhood_fptr;
    FILE* fp;
    char string[100];
    char filename[100];
    int iteration;
	double lhood, prev_lhood, conv, doclkh;
	double c, y;
	int d, n, j, w;
    vblda_corpus* corpus;
    vblda_model *model = NULL;
    vblda_ss* ss = NULL;
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
    	model = new_vblda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
		ss = new_vblda_ss(model);
		//random_initialize_model(mc, model, corpus, ss);

		corpus_initialize_model(model, corpus, ss);
		//corpus_initialize_model2(var, model, corpus);

    }
    else if (strcmp(start, "random")==0){
    	printf("random\n");
    	model = new_vblda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
		ss = new_vblda_ss(model);

		random_initialize_model(model, corpus, ss);

    }
    else if (strcmp(start, "loadbeta")==0){ //old
    	printf("load beta\n");
    	model = new_vblda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
		ss = new_vblda_ss(model);
	    sprintf(filename, "%s.beta", model_name);
	    printf("%s\n",filename);
	    fp = fopen(filename, "r");
	    for (n = 0; n < model->n; n++){
	    	for (j = 0; j < model->m; j++){
				fscanf(fp, " %lf", &y);
				model->mu[j][n] = y;
	    	}
	    }
	    fclose(fp);
    }


	//init ss

	for (d = 0; d < corpus->ndocs; d++){
		ss->sumgamma[d] = 0.0;
		for (j = 0; j < model->m; j++){
			model->gamma[j][d] = model->alpha + (double)corpus->docs[d].total/((double)model->m);
			ss->sumgamma[d] += model->gamma[j][d];
		}
		for (n = 0; n < corpus->docs[d].length; n++){
			w = corpus->docs[d].words[n];
			c = (double) corpus->docs[d].counts[n];
			for (j = 0; j < model->m; j++){
				corpus->docs[d].phi[n][j] = 1.0/((double)model->m);
			}
		}
	}
	for (j = 0; j < model->m; j++){
		ss->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			ss->summu[j] += model->mu[j][n];
		}
		for (n = 0; n < model->n; n++){
			//model->psimu[j][n] = gsl_sf_psi(model->mu[j][n])-gsl_sf_psi(ss->summu[j]);
			y = model->mu[j][n]/ss->summu[j];
			if (y == 0) y = 1e-50;
			model->psimu[j][n] = log(y);
		}
	}

	//zero init tss
	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			ss->t[j][n] = 0.0;;
		}
	}
    iteration = 0;
    sprintf(filename, "%s/%03d", dir,iteration);
    printf("%s\n",filename);
	write_vblda_model(model, filename, corpus);

    time(&t1);
	prev_lhood = -1e100;

	do{

		printf("***** VB ITERATION %d *****\n", iteration);
		//if (iteration == 10) prev_lhood = -1e100;
		lhood = 0.0;
		for (d = 0; d < corpus->ndocs; d++){
			ss->sumgamma[d] = 0.0;
			for (j = 0; j < model->m; j++){
				model->gamma[j][d] = model->alpha + (double)corpus->docs[d].total/((double)model->m);
				ss->sumgamma[d] += model->gamma[j][d];
			}
			for (n = 0; n < corpus->docs[d].length; n++){
				w = corpus->docs[d].words[n];
				c = (double) corpus->docs[d].counts[n];
				for (j = 0; j < model->m; j++){
					corpus->docs[d].phi[n][j] = 1.0/((double)model->m);
				}
			}
			doclkh = doc_inference(corpus, model, ss, d, 0);
			lhood += doclkh;
		}

		//update mu and lhood
		if ((iteration < 10)){
			for (j = 0; j < model->m; j++){
				ss->summu[j] = 0.0;
				for (n = 0; n < model->n; n++){
					model->mu[j][n] = ss->t[j][n];
					ss->summu[j] += model->mu[j][n];
				}

				for (n = 0; n < model->n; n++){
					y = model->mu[j][n]/ss->summu[j];
					if (y == 0) y = 1e-50;
					model->psimu[j][n] = log(y);
					//model->psimu[j][n] = gsl_sf_psi(model->mu[j][n])-gsl_sf_psi(ss->summu[j]);
					lhood += ss->t[j][n]*model->psimu[j][n];
					ss->t[j][n] = 0.0;
				}
				prev_lhood = -1e100;
			}
		}else{
			for (j = 0; j < model->m; j++){
				ss->summu[j] = 0.0;
				for (n = 0; n < model->n; n++){
					model->mu[j][n] = model->nu + ss->t[j][n];
					ss->summu[j] += model->mu[j][n];

					ss->t[j][n] = 0.0;

					lhood += lgamma(model->mu[j][n]);
				}
				lhood -= lgamma(ss->summu[j]);

				for (n = 0; n < model->n; n++){
					model->psimu[j][n] = gsl_sf_psi(model->mu[j][n])-gsl_sf_psi(ss->summu[j]);
				}
			}
		}

		conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);
		prev_lhood = lhood;

		if (prev_lhood > lhood){
			printf("oops ... \n");
		}
		time(&t2);

		sprintf(filename, "%s/%03d", dir,1);
		write_vblda_model(model, filename, corpus);

		fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (int)t2-t1);
		fflush(lhood_fptr);
		iteration ++;

	}while((iteration < MAXITER) && (conv > CONVERGED));
	fclose(lhood_fptr);

    sprintf(filename, "%s/final", dir);
    write_vblda_model(model, filename, corpus);

}

double doc_inference(vblda_corpus* corpus, vblda_model* model, vblda_ss* ss, int d, int test){

	int n, j, variter, w;
	double c, phisum, temp, cphi;
	double varlkh, prev_varlkh, conv;

	prev_varlkh = -1e100;
	conv = 0.0;
	variter = 0;
	do{
		varlkh = 0.0;
		for (n = 0; n < corpus->docs[d].length; n++){
			w = corpus->docs[d].words[n];
			c = (double) corpus->docs[d].counts[n];

			phisum = 0.0;
			for (j = 0; j < model->m; j++){
				ss->oldphi[j] = corpus->docs[d].phi[n][j];

				corpus->docs[d].phi[n][j] = gsl_sf_psi(model->gamma[j][d]) + model->psimu[j][w];
				if (j > 0)
					phisum = log_sum(phisum, corpus->docs[d].phi[n][j]);
				else
					phisum = corpus->docs[d].phi[n][j];
			}
			for (j = 0; j < model->m; j++){

				corpus->docs[d].phi[n][j] = exp(corpus->docs[d].phi[n][j] - phisum);

				temp = c*(corpus->docs[d].phi[n][j] - ss->oldphi[j]);
				model->gamma[j][d] += temp;
				ss->sumgamma[d] += temp;

				if (corpus->docs[d].phi[n][j] > 0){
					cphi = c*corpus->docs[d].phi[n][j];
					varlkh += cphi*(model->psimu[j][w]-log(corpus->docs[d].phi[n][j]));
				}
			}
		}
		varlkh -= lgamma(ss->sumgamma[d]);
		for (j = 0; j < model->m; j++){
			varlkh += lgamma(model->gamma[j][d]);
		}

		conv = fabs(prev_varlkh - varlkh)/fabs(prev_varlkh);
		if (prev_varlkh > varlkh){
			printf("ooops doc %d, %lf %lf, %5.10e\n", d, varlkh, prev_varlkh, conv);
		}
		prev_varlkh = varlkh;
		variter ++;

	}while((variter < MAXITER) && (conv > CONVERGED));

	if (test == 0){
		for (n = 0; n < corpus->docs[d].length; n++){
			w = corpus->docs[d].words[n];
			c = (double) corpus->docs[d].counts[n];
			for (j = 0; j < model->m; j++){
				cphi = corpus->docs[d].phi[n][j];
				varlkh -= cphi*model->psimu[j][w];
				ss->t[j][w] += cphi;
			}
		}
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

	vblda_corpus* corpus;
	vblda_model *model = NULL;
	vblda_ss* ss = NULL;
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
	ss = new_vblda_ss(model);

	//*************************************
	for (j = 0; j < model->m; j++){
		ss->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			ss->summu[j] += model->mu[j][n];
		}
		for (n = 0; n < model->n; n++){
			model->psimu[j][n] = log(model->mu[j][n]/ss->summu[j]);
		}
	}

    iteration = 0;
    sprintf(filename, "%s/test%03d", dir,iteration);
    printf("%s\n",filename);
	write_vblda_model(model, filename, corpus);

    time(&t1);

	lhood = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		ss->sumgamma[d] = 0.0;
		for (j = 0; j < model->m; j++){
			model->gamma[j][d] = model->alpha + (double)corpus->docs[d].total/((double)model->m);
			ss->sumgamma[d] += model->gamma[j][d];
		}
		for (n = 0; n < corpus->docs[d].length; n++){
			//w = corpus->docs[d].words[n];
			//c = (double) corpus->docs[d].counts[n];
			for (j = 0; j < model->m; j++){
				corpus->docs[d].phi[n][j] = 1.0/((double)model->m);
			}
		}
		doclkh = doc_inference(corpus, model, ss, d, 1);
		lhood += doclkh;
	}

	time(&t2);

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, 0.0, (int)t2-t1);
	fflush(lhood_fptr);
	fclose(lhood_fptr);
	//*************************************

	sprintf(filename, "%s/testfinal", dir);
	write_vblda_model(model, filename, corpus);

}


vblda_model* new_vblda_model(int ntopics, int nterms, int ndocs, double alpha, double nu)
{
	int n, j, d;

	vblda_model* model = malloc(sizeof(vblda_model));
	model->D = ndocs;
    model->m = ntopics;
    model->n = nterms;
    model->nu = nu;
    model->mu = malloc(sizeof(double*)*ntopics);
    model->psimu = malloc(sizeof(double*)*ntopics);
    model->gamma = malloc(sizeof(double*)*ntopics);
    for (j = 0; j < ntopics; j++){
		model->mu[j] = malloc(sizeof(double)*nterms);
		model->psimu[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->mu[j][n] = 0.0;
			model->psimu[j][n] = 0.0;
		}
		model->gamma[j] = malloc(sizeof(double)*ndocs);
		for (d = 0; d < ndocs; d++){
			model->gamma[j][d] = 0.0;
		}
	}
    model->nu = nu;
    model->alpha = alpha;

    return(model);
}


/*
 * create and delete sufficient statistics
 *
 */

vblda_ss * new_vblda_ss(vblda_model* model)
{
	int j, n, d;
	vblda_ss * ss;
    ss = malloc(sizeof(vblda_ss));
	ss->m = malloc(sizeof(double*)*model->m);
	ss->t = malloc(sizeof(double*)*model->m);
	ss->oldphi = malloc(sizeof(double)*model->m);
	ss->summu = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		ss->oldphi[j] = 0.0;
		ss->summu[j] = 0.0;
		ss->t[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			ss->t[j][n] = 0.0;
		}
		ss->m[j] = malloc(sizeof(double)*model->D);
		for (d = 0; d < model->D; d++){
			ss->m[j][d] = 0.0;
		}
	}

	ss->sumgamma = malloc(sizeof(double)*model->D);
	for (d = 0; d < model->D; d++){
		ss->sumgamma[d] = 0.0;
	}

    return(ss);
}



vblda_corpus* read_data(const char* data_filename, int ntopics)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw, j;
	vblda_corpus* c;

	printf("reading data from %s\n", data_filename);
	c = malloc(sizeof(vblda_corpus));
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

int max_corpus_length(vblda_corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->ndocs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}


void write_vblda_model(vblda_model * model, char * root,vblda_corpus * corpus)
{
    char filename[200];
    FILE* fileptr;
    int n, j, d;

    //beta
    sprintf(filename, "%s.beta", root);
    fileptr = fopen(filename, "w");
    for (n = 0; n < model->n; n++){
    	for (j = 0; j < model->m; j++){
    		fprintf(fileptr, "%.10lf ",model->mu[j][n]);
    	}
    	fprintf(fileptr, "\n");
    }
    fclose(fileptr);

    //theta
	sprintf(filename, "%s.theta", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%5.10lf ", model->gamma[j][d]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);


	sprintf(filename, "%s.other", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr,"num_topics %d \n",model->m);
	fprintf(fileptr,"num_terms %d \n",model->n);
	fprintf(fileptr,"num_docs %d \n",model->D);
	fprintf(fileptr,"alpha %lf \n",model->alpha);
	fprintf(fileptr,"nu %lf \n",model->nu);
	fclose(fileptr);

}

vblda_model* load_model(char* model_root, int ndocs){

	char filename[100];
	FILE* fileptr;
	int j, n, num_topics, num_terms, num_docs;
	//float x;
	double y, alpha, nu;

	vblda_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fscanf(fileptr, "alpha %lf\n", &alpha);
	fscanf(fileptr, "nu %lf\n", &nu);
	fclose(fileptr);

	model  = new_vblda_model(num_topics, num_terms, ndocs, alpha, nu);
	model->n = num_terms;
	model->m = num_topics;
	model->D = ndocs;
	model->alpha = alpha;
	model->nu = nu;


	sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (n = 0; n < num_terms; n++){
		for (j = 0; j < num_topics; j++){
			fscanf(fileptr, " %lf", &y);
			model->mu[j][n] = y;
		}
	}
    fclose(fileptr);

    return(model);
}


void corpus_initialize_model(vblda_model* model, vblda_corpus* corpus, vblda_ss* ss)
{

	int n, j, d, i, count;
	//double sum, temp, maxlkh;
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
			model->mu[j][n] = model->nu;
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
				model->mu[j][corpus->docs[d].words[n]] += (double) corpus->docs[d].counts[n];
			}
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



void random_initialize_model(vblda_model * model, vblda_corpus* corpus, vblda_ss* ss){

	int n, j, d;
	double* beta = malloc(sizeof(double)*model->n);
	double* nu = malloc(sizeof(double)*model->n);

	for (n = 0; n < model->n; n++){
		beta[n] = 0.0;
		nu[n] = 0.0;
	}


	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			nu[n] = model->nu;
		}
		gsl_ran_dirichlet (r, model->n, nu, beta);
		for (n = 0; n < model->n; n++){
			//model->mu[j][n] = beta[n];
			//model->mu[j][n] = gsl_ran_gamma(r, 100.0, 1.0/100.);
			model->mu[j][n] = model->nu + 1.0 + gsl_rng_uniform(r);
		}
	}

  	free(beta);
  	free(nu);
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

