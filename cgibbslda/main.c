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

	MCSIZE = 5e4;
	CONVERGED = 1e-4;
	NUMINIT = 20;

	seed = atoi(argv[1]);
	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

    printf("SEED = %ld\n", seed);

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
			if ((strcmp(init,"load")==0) || (strcmp(init,"loadtopics")==0))
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
    char string[100];
    char filename[100];
    int iteration;

    cgibbslda_corpus* corpus;
    cgibbslda_model *model = NULL;
    cgibbslda_ss* ss = NULL;
    time_t t1,t2;
    //FILE* fileptr;
    //float x;
    //double y;
    corpus = read_data(dataset);


    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    if (strcmp(start, "seeded")==0){  //not updated
    	printf("seeded\n");
    	model = new_cgibbslda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
		ss = new_cgibbslda_ss(model);
		//random_initialize_model(mc, model, corpus, ss);

		corpus_initialize_model(model, corpus, ss);
		//corpus_initialize_model2(var, model, corpus);

    }
    else if (strcmp(start, "random")==0){
    	printf("random\n");
    	model = new_cgibbslda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
		ss = new_cgibbslda_ss(model);

		random_initialize_model(model, corpus, ss);

    }
    else if (strcmp(start, "load")==0){ //old
    	printf("load\n");
    	model = load_model(model_name, corpus->ndocs);
		ss = new_cgibbslda_ss(model);
		/*for (d = 0; d < corpus->ndocs; d++){
			for (j = 0; j < model->m; j++){
				corpus->docs[d].theta[j] = 1.0/((double)model->m);
				corpus->docs[d].thetahat[j] = (corpus->docs[d].theta[j]);
			}
		}*/
    }


//*************************************
	double lhood0, lhood1, prev_lhood0, prev_lhood1, conv0, conv1;
    time(&t1);
    iteration = 1;
    sprintf(filename, "%s/%03d", dir, iteration);
    printf("%s\n",filename);
	write_cgibbslda_model(model, filename, corpus);

	prev_lhood0 = 0.0;
	prev_lhood1 = 0.0;
	//for (iteration = 2; iteration < MCSIZE; iteration++){
	iteration = 2;
	do{
		mcsample(corpus, model, ss, &lhood0, &lhood1, 0);

		conv0 = (prev_lhood0 - lhood0)/prev_lhood0;
		conv1 = (prev_lhood1- lhood1)/prev_lhood1;
		prev_lhood0 = lhood0;
		prev_lhood1 = lhood1;

		time(&t2);

		//sprintf(filename, "%s/%03d", dir,1);
		//write_cgibbslda_model(model, filename, corpus);
		if ((iteration%ss->b) == 0){
			printf("***** MCMC ITERATION %d *****\n", iteration);
			fprintf(lhood_fptr, "%d %5.5e %5.5e %5.5e %5.5e %d %5ld %5.5e\n",iteration, lhood0, conv0, lhood1, conv1,
					ss->change, (int)t2-t1, ss->mcse);
			fflush(lhood_fptr);
		}
		iteration ++;
		if ((iteration > MCSIZE) || ((iteration > 5*ss->b) && (ss->mcse < 0.03)))
			break;
	}while(1);
	fclose(lhood_fptr);

    sprintf(filename, "%s/final", dir);
    write_cgibbslda_model(model, filename, corpus);

}


void mcsample(cgibbslda_corpus* corpus, cgibbslda_model* model, cgibbslda_ss* ss,
		double* lhood0, double* lhood1, int testchk)
{
    int j, d, n, prevz, w, tt;
    double lkh0, lkh1, temp0, temp1, sum, a, temp;

	double* theta = malloc(sizeof(double)*model->m);
	double* phi = malloc(sizeof(double)*model->m);
	double* beta = malloc(sizeof(double)*model->n);
	double* nu = malloc(sizeof(double)*model->n);
	double* alpha = malloc(sizeof(double)*model->m);
	unsigned int* z = malloc(sizeof(int)*model->m);
	for (j = 0; j < model->m; j++){
		theta[j] = 0.0;
		z[j] = 0;
		phi[j] = 0.0;
		alpha[j] = 0.0;
	}
	for (n = 0; n < model->n; n++){
		beta[n] = 0.0;
		nu[n] = 0.0;
	}
	ss->change = 0;

	int chk;
	ss->n += 1.0;
	tt = (int)ss->n%ss->b;
	if (tt == 0){
		ss->mcse = 0.0;
		ss->mcse_num = 0.0;
	}
	lkh0 = 0.0;
	lkh1 = 0.0;
	//sample theta and z
	for (d = 0; d < corpus->ndocs; d++){

		for (n = 0; n < corpus->docs[d].total; n++){
			temp0 = 0.0;
			temp1 = 0.0;
			w = corpus->docs[d].words[n];
			prevz = corpus->docs[d].z[n];
			ss->m[prevz][d] -= 1.0;
			ss->t[prevz][corpus->docs[d].words[n]] -= 1.0;
			ss->sumt[prevz] -= 1.0;
			for (j = 0; j < model->m; j++){

				if (testchk == 0){
					phi[j] = 1000.0*(model->nu+ss->t[j][w])*(model->alpha+ss->m[j][d])/
						(model->nu*model->n+ss->sumt[j]);
				}else{
					phi[j] = (model->alpha+ss->m[j][d])*model->beta[j][w];
				}

				temp0 += model->theta[j][d]*model->beta[j][w];
				temp1 += model->mctheta[j][d]*model->mcbeta[j][w];
			}
			lkh0 += log(temp0);
			lkh1 += log(temp1);
			if ((isinf(lkh0)) || (isnan(lkh0))){
				for (j = 0; j < model->m; j++){
					printf("%lf %lf - ",model->theta[j][d],model->beta[j][w]);
				}
				printf("\n");
			}

			gsl_ran_multinomial (r, model->m, 1, phi, z);
			chk = 0;
			for (j = 0; j < model->m; j++){
				if (z[j] == 1){
					chk = 1;
					corpus->docs[d].z[n] = j;
					temp = corpus->docs[d].mcz[n];
					corpus->docs[d].mcz[n] = (corpus->docs[d].mcz[n]*(ss->n-1) + j)/(ss->n);
					temp = corpus->docs[d].mcz[n];
					tt = (int)ss->n%ss->b;
					if (tt == 0){
						corpus->docs[d].yz[n] = (corpus->docs[d].yz[n]*(ss->b-1) + j)/ss->b; //last one int his batch
						a = ss->a;
						if (a > 1)
							corpus->docs[d].sz2[n] = (a-1)*corpus->docs[d].sz2[n]/a+
											ss->b*(corpus->docs[d].yz[n]*corpus->docs[d].yz[n])/a;
						else
							corpus->docs[d].sz2[n] = ss->b*(corpus->docs[d].yz[n]*corpus->docs[d].yz[n]);

						if (isnan(corpus->docs[d].sz2[n])){
							printf("%lf %lf\n", corpus->docs[d].yz[n], corpus->docs[d].mcz[n]);
						}
						ss->mcse_num += 1.0;
						temp = (corpus->docs[d].sz2[n] - ss->b*corpus->docs[d].mcz[n]*corpus->docs[d].mcz[n]);
						if (temp > 0) temp = sqrt(temp/ss->n);
						if (temp > ss->mcse) ss->mcse = temp;	
						if (isnan(ss->mcse)){
							printf("%lf %lf\n", corpus->docs[d].sz2[n], ss->b*corpus->docs[d].mcz[n]*corpus->docs[d].mcz[n]);
						}
					}
					else{
						temp = corpus->docs[d].yz[n];
						if (tt == 1)
							corpus->docs[d].yz[n] = (double) j;
						else
							corpus->docs[d].yz[n] = (corpus->docs[d].yz[n]*(tt-1.0) + j)/tt;
						temp = corpus->docs[d].yz[n];
					}


					ss->m[j][d] += 1.0;
					ss->t[j][corpus->docs[d].words[n]] += 1.0;
					ss->sumt[j] += 1.0;
					if (j != prevz) ss->change += 1;
					break;
				}
			}
			if (chk == 0){
				for (j = 0; j < model->m; j++){
					printf("%lf %lf %lf - ",model->theta[j][d],model->beta[j][w], phi[j]);
				}
				printf("\n");
			}

		}
		sum = 0.0;
		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha + ss->m[j][d];
			sum += alpha[j];
		}
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = alpha[j]/sum;
			model->mctheta[j][d] = (model->mctheta[j][d]*(ss->n - 1.0) + model->theta[j][d])/ss->n;
		}

	}

	if (testchk == 0){
		for (j = 0; j < model->m; j++){
			for (n = 0; n < model->n; n++){
				model->beta[j][n] = (model->nu+ss->t[j][n])/(model->nu*model->n+ss->sumt[j]);
				model->mcbeta[j][n] = (model->mcbeta[j][n]*(ss->n - 1.0) + model->beta[j][n])/ss->n;
			}
		}
	}


	tt = (int)ss->n%ss->b;
	if (tt == 0){
		ss->a += 1.0;
		//if (ss->mcse_num > 0)
		//	ss->mcse /= ss->mcse_num;
	}

	*lhood0 = lkh0;
	*lhood1 = lkh1;

	free(theta);
	free(phi);
	free(beta);
	free(z);
	free(nu);
	free(alpha);
}




void test(char* dataset, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	char string[400];
	char filename[400];
	int iteration;
	double lhood0, lhood1, conv0, conv1, prev_lhood1, prev_lhood0;
	double sum;

	cgibbslda_corpus* corpus;
	cgibbslda_model *model = NULL;
	cgibbslda_ss* ss = NULL;
	time_t t1,t2;
	//float x;
	//double y;

	corpus = read_data(dataset);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file

	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = load_model(model_name, corpus->ndocs);
	ss = new_cgibbslda_ss(model);

	//init topic proportions and zs
	ss->n = 1;
	ss->a = 1;

	int d, n, j;
	double* theta = malloc(sizeof(double)*model->m);
	double* phi = malloc(sizeof(double)*model->m);
	double* alpha = malloc(sizeof(double)*model->m);
	unsigned int* z = malloc(sizeof(int)*model->m);
	for (j = 0; j < model->m; j++){
		theta[j] = 0.0;
		z[j] = 0;
		phi[j] = 0.0;
		alpha[j] = 0.0;
	}

	for (d = 0; d < corpus->ndocs; d++){
		for (n = 0; n < corpus->docs[d].total; n++){
			for (j = 0; j < model->m; j++){
				phi[j] = model->beta[j][corpus->docs[d].words[n]];
			}
			gsl_ran_multinomial (r, model->m, 1, phi, z);
			for (j = 0; j < model->m; j++){
				if (z[j] == 1){
					corpus->docs[d].z[n] = j;
					corpus->docs[d].mcz[n] = (double)j;
					corpus->docs[d].yz[n] = (double)j;
					ss->m[j][d] += 1.0;
					break;
				}
			}
		}
		sum = 0.0;
		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha + ss->m[j][d];
			sum += alpha[j];
		}
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = alpha[j]/sum;
			model->mctheta[j][d] = model->theta[j][d];
		}
	}
	free(z);
	free(phi);
	free(alpha);
	free(theta);

	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			model->mcbeta[j][n] = model->beta[j][n];
		}
	}

	//*************************************
	//double lhood2;
	time(&t1);
	iteration = 1;
	sprintf(filename, "%s/test%03d", dir,iteration);
	printf("%s\n",filename);
	write_cgibbslda_model(model, filename, corpus);

	prev_lhood0 = 0.0;
	prev_lhood1 = 0.0;
	iteration = 2;
	//for (iteration = 0; iteration < MCSIZE; iteration++){
	do{
		//printf("***** MCMC ITERATION %d *****\n", iteration);

		mcsample(corpus, model, ss, &lhood0, &lhood1, 1);

		conv0 = (prev_lhood0 - lhood0)/prev_lhood0;
		conv1 = (prev_lhood1- lhood1)/prev_lhood1;
		prev_lhood0 = lhood0;
		prev_lhood1 = lhood1;

		time(&t2);

		if ((iteration%ss->b) == 0){
			printf("***** MCMC ITERATION %d *****\n", iteration);
			fprintf(lhood_fptr, "%d %5.5e %5.5e %5.5e %5.5e %d %5ld %5.5e\n",iteration, lhood0, conv0, lhood1, conv1,
					ss->change, (int)t2-t1, ss->mcse);
			fflush(lhood_fptr);
		}
		iteration ++;
		if ((iteration > MCSIZE) || ((iteration > 5*ss->b) && (ss->mcse < 0.03)))
			break;

	}while(1);
	fclose(lhood_fptr);

	sprintf(filename, "%s/testfinal", dir);
	write_cgibbslda_model(model, filename, corpus);

}


cgibbslda_model* new_cgibbslda_model(int ntopics, int nterms, int ndocs, double alpha, double nu)
{
	int n, j, d;

	cgibbslda_model* model = malloc(sizeof(cgibbslda_model));
	model->D = ndocs;
    model->m = ntopics;
    model->n = nterms;
    model->nu = nu;
    model->beta = malloc(sizeof(double*)*ntopics);
    model->theta = malloc(sizeof(double*)*ntopics);
    model->mcbeta = malloc(sizeof(double*)*ntopics);
    model->mctheta = malloc(sizeof(double*)*ntopics);
    for (j = 0; j < ntopics; j++){
		model->beta[j] = malloc(sizeof(double)*nterms);
		model->mcbeta[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->beta[j][n] = 0.0;
			model->mcbeta[j][n] = 0.0;
		}
		model->theta[j] = malloc(sizeof(double)*ndocs);
		model->mctheta[j] = malloc(sizeof(double)*ndocs);
		for (d = 0; d < ndocs; d++){
			model->theta[j][d] = 0.0;
			model->mctheta[j][d] = 0.0;
		}
	}
    model->nu = nu;
    model->alpha = alpha;

    return(model);
}


cgibbslda_ss * new_cgibbslda_ss(cgibbslda_model* model)
{
	int j, n, d;
	cgibbslda_ss * ss;
    ss = malloc(sizeof(cgibbslda_ss));
	ss->m = malloc(sizeof(double*)*model->m);
	ss->t = malloc(sizeof(double*)*model->m);
	ss->sumt = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		ss->t[j] = malloc(sizeof(double)*model->n);
		ss->sumt[j] = 0.0;
		for (n = 0; n < model->n; n++){
			ss->t[j][n] = 0.0;
		}
		ss->m[j] = malloc(sizeof(double)*model->D);
		for (d = 0; d < model->D; d++){
			ss->m[j][d] = 0.0;
		}
	}

	ss->n = 0;
	ss->b = 50;
	ss->a = 0;
    return(ss);
}



cgibbslda_corpus* read_data(const char* data_filename)
{
	int OFFSET = 0;
    FILE *fileptr;
    int length, count, word, n, nd, nw, corpus_total = 0;
    int total, i;
    int* wrds = malloc(sizeof(int)*100000);
    int* cnts = malloc(sizeof(int)*100000);

    cgibbslda_corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(cgibbslda_corpus));
    fileptr = fopen(data_filename, "r");
    nd = 0; nw = 0;
    c->docs = malloc(sizeof(cgibbslda_document) * 1);
    while ((fscanf(fileptr, "%10d", &length) != EOF)){

	c->docs = (cgibbslda_document*) realloc(c->docs, sizeof(cgibbslda_document)*(nd+1));
	c->docs[nd].length = length;
	c->docs[nd].total = 0;

	total = 0;
	for (n = 0; n < length; n++){
	    fscanf(fileptr, "%10d:%10d", &word, &count);
	    word = word - OFFSET;
	    for (i = 0; i < count; i++){
	    	wrds[total] = word;
	    	cnts[total] = count;
	    	total ++;
	    }
	    //c->docs[nd].words[n] = word;
	    //c->docs[nd].counts[n] = count;
	    c->docs[nd].total += count;
	    if (word >= nw) { nw = word + 1; }
	}
	c->docs[nd].words = malloc(sizeof(int)*total);
	c->docs[nd].counts = malloc(sizeof(int)*total);
	c->docs[nd].z = malloc(sizeof(int*)*total);
	c->docs[nd].mcz = malloc(sizeof(double*)*total);
	c->docs[nd].sz2 = malloc(sizeof(double*)*total);
	c->docs[nd].yz = malloc(sizeof(double*)*total);
	for (n = 0; n < c->docs[nd].total; n++){
		c->docs[nd].words[n] = wrds[n];
		c->docs[nd].counts[n] = cnts[n];
		c->docs[nd].z[n] = 0;
		c->docs[nd].mcz[n] = 0.0;
		c->docs[nd].sz2[n] = 0.0;
		c->docs[nd].yz[n] = 0.0;
	}
	corpus_total += c->docs[nd].total;
        nd++;
    }
    fclose(fileptr);
    c->ndocs = nd;
    c->nterms = nw;
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
    printf("total             : %d\n", corpus_total);
    free(wrds);
    free(cnts);
    return(c);
}

int max_corpus_length(cgibbslda_corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->ndocs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}


void write_cgibbslda_model(cgibbslda_model * model, char * root,cgibbslda_corpus * corpus)
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

	sprintf(filename, "%s.z", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		fprintf(fileptr, "%03d", corpus->docs[d].total);
		for (n = 0; n < corpus->docs[d].total; n++){
			fprintf(fileptr, " %04d:%02d",corpus->docs[d].words[n],(int)floor(corpus->docs[d].mcz[n]));
		}
		fprintf(fileptr, "\n");
		fflush(fileptr);
	}
	fclose(fileptr);

	sprintf(filename, "%s.other", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr,"num_topics %d \n",model->m);
	fprintf(fileptr,"num_terms %d \n",model->n);
	fprintf(fileptr,"num_docs %d \n",model->D);
	fprintf(fileptr,"alpha %lf \n",model->alpha);
	fprintf(fileptr,"nu %lf \n",model->nu);
	fprintf(fileptr,"T %d \n",(int)MCSIZE);
	fclose(fileptr);

}

cgibbslda_model* load_model(char* model_root, int ndocs){

	char filename[100];
	FILE* fileptr;
	int j, n, num_topics, num_terms, num_docs;
	//float x;
	double y, alpha, nu;


	cgibbslda_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fscanf(fileptr, "alpha %lf\n", &alpha);
	fscanf(fileptr, "nu %lf\n", &nu);
	fclose(fileptr);

	model  = new_cgibbslda_model(num_topics, num_terms, ndocs, alpha, nu);
	model->n = num_terms;
	model->m = num_topics;
	model->D = ndocs;
	model->alpha = alpha;
	model->nu = nu;

	double* sumbeta = malloc(sizeof(double)*num_topics);
	for (j = 0; j < num_topics; j++){
		sumbeta[j] = 0.0;
	}
	sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (n = 0; n < num_terms; n++){
		for (j = 0; j < num_topics; j++){
			fscanf(fileptr, " %lf", &y);
			model->beta[j][n] = exp(y);
			sumbeta[j] += model->beta[j][n];
		}
	}
    fclose(fileptr);
    //normalize beta
    for (j = 0; j < num_topics; j++){
    	for (n = 0; n < num_terms; n++){
    		model->beta[j][n] /= sumbeta[j];
    	}
    }

    free(sumbeta);
    return(model);
}


void corpus_initialize_model(cgibbslda_model* model, cgibbslda_corpus* corpus, cgibbslda_ss* ss)
{

	int n, j, d, i, count;
	double sum;
	double* theta = malloc(sizeof(double)*model->m);
	double* p = malloc(sizeof(double)*model->m);
	double* beta = malloc(sizeof(double)*model->n);
	double* alpha = malloc(sizeof(double)*model->m);
	double* nu = malloc(sizeof(double)*model->n);
	unsigned int* z = malloc(sizeof(int)*model->m);
	int* sdocs = malloc(sizeof(int)*corpus->ndocs);
	for (d = 0; d < corpus->ndocs; d++){
		sdocs[d] = -1;
	}
	for (j = 0; j < model->m; j++){
		theta[j] = 0.0;
		z[j] = 0;
		p[j] = 0.0;
		alpha[j] = 0.0;
	}
	for (n = 0; n < model->n; n++){
		beta[n] = 0.0;
		nu[n] = 0.0;
	}

	ss->n = 1;
	ss->a = 1;
	//init local topics
	for (j = 0; j < model->m; j++){
		//maxcnt = -10;
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = model->nu;
		}
		sum = (double)model->n*model->nu;
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

			for (n = 0; n < corpus->docs[d].total; n++){
				model->beta[j][corpus->docs[d].words[n]] += 1.0;
				sum += 1.0;
			}
		}
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = model->beta[j][n]/sum;
			model->mcbeta[j][n] = model->beta[j][n];
		}
	}

	//******
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = 1.0/((double)model->m);
			model->mctheta[j][d] = model->theta[j][d];
		}

		for (n = 0; n < corpus->docs[d].total; n++){
			for (j = 0; j < model->m; j++){
				p[j] = 1000.0*model->theta[j][d]*model->beta[j][corpus->docs[d].words[n]];
			}
			gsl_ran_multinomial (r, model->m, 1, p, z);
			for (j = 0; j < model->m; j++){
				if (z[j] == 1){
					corpus->docs[d].z[n] = j;
					corpus->docs[d].mcz[n] = (double)j;
					corpus->docs[d].yz[n] = (double)j;
					ss->m[j][d] += 1.0;
					ss->t[j][corpus->docs[d].words[n]] += 1.0;
					ss->sumt[j] += 1.0;
					break;
				}
			}
		}
		//sample theta
		sum = 0.0;
		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha + ss->m[j][d];
			sum += alpha[j];
		}
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = alpha[j]/sum;
			model->mctheta[j][d] = model->theta[j][d];
		}
	}


	//sample beta
	for (j = 0; j < model->m; j++){
		sum = 0.0;
		for (n = 0; n < model->n; n++){
			nu[n] = model->nu + ss->t[j][n];
			sum += nu[n];
		}
		//gsl_ran_dirichlet (r, model->n, nu, beta);
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = nu[n]/sum;
			model->mcbeta[j][n] = model->beta[j][n];
		}
	}



  	free(theta);
  	free(beta);
  	free(z);
  	free(p);
  	free(alpha);
  	free(nu);
  	free(sdocs);


}



void random_initialize_model(cgibbslda_model * model, cgibbslda_corpus* corpus, cgibbslda_ss* ss){

	int n, j, d;
	double* theta = malloc(sizeof(double)*model->m);
	double* p = malloc(sizeof(double)*model->m);
	double* beta = malloc(sizeof(double)*model->n);
	double* alpha = malloc(sizeof(double)*model->m);
	double* nu = malloc(sizeof(double)*model->n);
	unsigned int* z = malloc(sizeof(int)*model->m);
	for (j = 0; j < model->m; j++){
		theta[j] = 0.0;
		z[j] = 0;
		p[j] = 0.0;
		alpha[j] = 0.0;
	}
	for (n = 0; n < model->n; n++){
		beta[n] = 0.0;
		nu[n] = 0.0;
	}

	ss->n = 1;
	ss->a = 1;

	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			nu[n] = model->nu;
		}
		gsl_ran_dirichlet (r, model->n, nu, beta);
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = beta[n];
			model->mcbeta[j][n] = beta[n];
		}
	}

	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha;
		}
		gsl_ran_dirichlet (r, model->m, alpha, theta);
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = theta[j];
			model->mctheta[j][d] = theta[j];
		}

		for (n = 0; n < corpus->docs[d].total; n++){
			for (j = 0; j < model->m; j++){
				p[j] = 1000.0*model->theta[j][d]*model->beta[j][corpus->docs[d].words[n]];
			}
			gsl_ran_multinomial (r, model->m, 1, p, z);
			for (j = 0; j < model->m; j++){
				if (z[j] == 1){
					corpus->docs[d].z[n] = j;
					corpus->docs[d].mcz[n] = (double)j;
					corpus->docs[d].yz[n] = (double)j;
					ss->m[j][d] += 1.0;
					ss->t[j][corpus->docs[d].words[n]] += 1.0;
					ss->sumt[j] += 1.0;
					break;
				}
			}
		}
	}


  	free(theta);
  	free(beta);
  	free(z);
  	free(p);
  	free(alpha);
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

