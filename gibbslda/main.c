#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[40];
	char corpus_file[40];
	char model_name[40];
	char init[40];
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
	//task = argv[1];
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

    gibbslda_corpus* corpus;
    gibbslda_model *model = NULL;
    gibbslda_mc *mc = NULL;
    gibbslda_ss* ss = NULL;
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
    	model = new_gibbslda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
		mc = new_gibbslda_mc(corpus->ndocs, ntopics, corpus->nterms);
		ss = new_gibbslda_ss(model);
		//random_initialize_model(mc, model, corpus, ss);

		corpus_initialize_model(mc, model, corpus, ss);
		//corpus_initialize_model2(var, model, corpus);

    }
    else if (strcmp(start, "random")==0){
    	printf("random\n");
    	model = new_gibbslda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
		mc = new_gibbslda_mc(corpus->ndocs, ntopics, corpus->nterms);
		ss = new_gibbslda_ss(model);

		random_initialize_model(mc, model, corpus, ss);

    }
    else if (strcmp(start, "load")==0){ //old
    	printf("load\n");
    	model = load_model(model_name, corpus->ndocs);
    	mc = new_gibbslda_mc(corpus->ndocs, ntopics, corpus->nterms);
		ss = new_gibbslda_ss(model);
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
    sprintf(filename, "%s/%03d", dir,iteration);
    printf("%s\n",filename);
	write_gibbslda_model(model, filename, corpus, mc);

	prev_lhood0 = 0.0;
	prev_lhood1 = 0.0;
	//for (iteration = 2; iteration < MCSIZE; iteration++){
	iteration = 2;
	do{
		//printf("***** MCMC ITERATION %d *****\n", iteration);

		mcsample(corpus, model, ss, mc, &lhood0, &lhood1, 0);

		conv0 = (prev_lhood0 - lhood0)/prev_lhood0;
		conv1 = (prev_lhood1- lhood1)/prev_lhood1;
		prev_lhood0 = lhood0;
		prev_lhood1 = lhood1;

		time(&t2);

		//sprintf(filename, "%s/%03d", dir,1);
		//write_gibbslda_model(model, filename, corpus, mc);

		if ((iteration%mc->b) == 0){
			printf("***** MCMC ITERATION %d *****\n", iteration);
			fprintf(lhood_fptr, "%d %5.5e %5.5e %5.5e %5.5e %d %5ld %5.5e\n",iteration, lhood0, conv0, lhood1, conv1,
					mc->change, (int)t2-t1, mc->mcse);
			fflush(lhood_fptr);
		}
		iteration ++;
		if ((iteration > MCSIZE) || ((iteration > 5*mc->b) && (mc->mcse < 0.03)))
			break;
	}while(1);
	fclose(lhood_fptr);

    sprintf(filename, "%s/final", dir);
    write_gibbslda_model(model, filename, corpus, mc);

}


void mcsample(gibbslda_corpus* corpus, gibbslda_model* model, gibbslda_ss* ss, gibbslda_mc* mc,
		double* lhood0, double* lhood1, int testchk)
{
    int j, d, n, prevz, tt;
    double lkh0, lkh1, temp0, temp1, a, temp;

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
	mc->change = 0;
	mc->n += 1.0;
	tt = (int)mc->n%mc->b;
	if (tt == 0){
		mc->mcse = 0.0;
		mc->mcse_num = 0.0;
	}

	lkh0 = 0.0;
	lkh1 = 0.0;
	//sample theta and z
	for (d = 0; d < corpus->ndocs; d++){

		for (n = 0; n < corpus->docs[d].total; n++){
			temp0 = 0.0;
			temp1 = 0.0;
			for (j = 0; j < model->m; j++){
				phi[j] = 1000.0*model->theta[j][d]*model->beta[j][corpus->docs[d].words[n]];

				temp0 += model->theta[j][d]*model->beta[j][corpus->docs[d].words[n]];
				temp1 += mc->theta[j][d]*mc->beta[j][corpus->docs[d].words[n]];
			}
			lkh0 += log(temp0);
			lkh1 += log(temp1);
			prevz = corpus->docs[d].z[n];
			ss->m[prevz][d] -= 1.0;
			ss->t[prevz][corpus->docs[d].words[n]] -= 1.0;

			gsl_ran_multinomial (r, model->m, 1, phi, z);

			for (j = 0; j < model->m; j++){
				if (z[j] == 1){
					corpus->docs[d].z[n] = j;
					corpus->docs[d].mcz[n] = (corpus->docs[d].mcz[n]*(mc->n-1.0) + j)/mc->n;

					if (tt == 0){
						corpus->docs[d].yz[n] = (corpus->docs[d].yz[n]*(mc->b-1) + j)/mc->b; //last one int his batch
						a = mc->a;
						if (a > 1)
							corpus->docs[d].sz2[n] = (a-1)*corpus->docs[d].sz2[n]/a+
											mc->b*(corpus->docs[d].yz[n]*corpus->docs[d].yz[n])/a;
						else
							corpus->docs[d].sz2[n] = mc->b*(corpus->docs[d].yz[n]*corpus->docs[d].yz[n]);

						if (isnan(corpus->docs[d].sz2[n])){
							printf("%lf %lf\n", corpus->docs[d].yz[n], corpus->docs[d].mcz[n]);
						}
						mc->mcse_num += 1.0;
						temp = (corpus->docs[d].sz2[n] - mc->b*corpus->docs[d].mcz[n]*corpus->docs[d].mcz[n]);
						if (temp > 0) temp = sqrt(temp/mc->n);
						if (temp > mc->mcse) mc->mcse = temp;	
						if (isnan(mc->mcse)){
							printf("%lf %lf\n", corpus->docs[d].sz2[n], mc->b*corpus->docs[d].mcz[n]*corpus->docs[d].mcz[n]);
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
					if (j != prevz) mc->change += 1;
					break;
				}
			}
		}

		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha + ss->m[j][d];
		}
		gsl_ran_dirichlet (r, model->m, alpha, theta);
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = theta[j];
			mc->theta[j][d] = (mc->theta[j][d]*(mc->n-1.0) + theta[j])/mc->n;

			if (tt == 0){
				mc->ytheta[j][d] = (mc->ytheta[j][d]*(mc->b-1) + theta[j])/mc->b; //last one int his batch
				a = mc->a;
				if (a > 1)
					mc->stheta2[j][d] = (a-1)*mc->stheta2[j][d]/a+
									mc->b*(mc->ytheta[j][d]*mc->ytheta[j][d])/a;
				else
					mc->stheta2[j][d] = mc->b*(mc->ytheta[j][d]*mc->ytheta[j][d]);

				if (isnan(mc->stheta2[j][d])){
					printf("%lf %lf\n", mc->ytheta[j][d], mc->theta[j][d]);
				}
				mc->mcse_num += 1.0;
				temp = (mc->stheta2[j][d] - mc->b*mc->theta[j][d]*mc->theta[j][d]);
				if (temp > 0) temp = sqrt(temp/mc->n);
				if (temp > mc->mcse) mc->mcse = temp;	
				if (isnan(mc->mcse)){
					printf("%lf %lf\n", mc->stheta2[j][d], mc->b*mc->theta[j][d]*mc->theta[j][d]);
				}
			}
			else{
				//temp = mc->ytheta[j][d];
				if (tt == 1)
					mc->ytheta[j][d] = theta[j];
				else
					mc->ytheta[j][d] = (mc->ytheta[j][d]*(tt-1.0) + theta[j])/tt;
				//temp = mc->ytheta[j][d];
			}
		}

	}

	if (testchk == 0){
		//sample beta
		for (j = 0; j < model->m; j++){
			for (n = 0; n < model->n; n++){
				nu[n] = model->nu + ss->t[j][n];
			}
			gsl_ran_dirichlet (r, model->n, nu, beta);
			for (n = 0; n < model->n; n++){
				model->beta[j][n] = beta[n];
				mc->beta[j][n] = (mc->beta[j][n]*mc->n + beta[n])/(mc->n + 1.0);

				if (tt == 0){
					mc->ybeta[j][n] = (mc->ybeta[j][n]*(mc->b-1) + beta[n])/mc->b; //last one int his batch
					a = mc->a;
					if (a > 1)
						mc->sbeta2[j][n] = (a-1)*mc->sbeta2[j][n]/a+
										mc->b*(mc->ybeta[j][n]*mc->ybeta[j][n])/a;
					else
						mc->sbeta2[j][n] = mc->b*(mc->ybeta[j][n]*mc->ybeta[j][n]);

					if (isnan(mc->sbeta2[j][n])){
						printf("%lf %lf\n", mc->ybeta[j][n], mc->beta[j][n]);
					}
					mc->mcse_num += 1.0;
					temp = (mc->sbeta2[j][n] - mc->b*mc->beta[j][n]*mc->beta[j][n]);
					if (temp > 0) temp = sqrt(temp/mc->n);
					if (temp > mc->mcse) mc->mcse = temp;	
					if (isnan(mc->mcse)){
						printf("%lf %lf\n", mc->sbeta2[j][n], mc->b*mc->beta[j][n]*mc->beta[j][n]);
					}
				}
				else{
					//temp = mc->ytheta[j][d];
					if (tt == 1)
						mc->ybeta[j][n] = beta[n];
					else
						mc->ybeta[j][n] = (mc->ybeta[j][n]*(tt-1.0) + beta[n])/tt;
					//temp = mc->ytheta[j][d];
				}
			}
		}
	}

	if (tt == 0){
		mc->a += 1.0;
		//if (mc->mcse_num > 0)
		//	mc->mcse /= mc->mcse_num;
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
	char string[100];
	char filename[100];
	int iteration;
	double lhood0, lhood1, conv0, conv1, prev_lhood1, prev_lhood0;
	//double sumt;

	gibbslda_corpus* corpus;
	gibbslda_model *model = NULL;
	gibbslda_mc *mc = NULL;
	gibbslda_ss* ss = NULL;
	time_t t1,t2;
	//float x;
	//double y;

	corpus = read_data(dataset);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file

	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = load_model(model_name, corpus->ndocs);
	mc = new_gibbslda_mc(corpus->ndocs, model->m, model->n);
	ss = new_gibbslda_ss(model);

	//init topic proportions and zs
	mc->n = 1;
	mc->a = 1;

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
		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha + ss->m[j][d];
		}
		gsl_ran_dirichlet (r, model->m, alpha, theta);
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = theta[j];
			mc->theta[j][d] = model->theta[j][d];
			mc->ytheta[j][d] = model->theta[j][d];
		}
	}
    for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			mc->beta[j][n] = model->beta[j][n];
			mc->beta[j][n] = model->beta[j][n];
			mc->ybeta[j][n] = model->beta[j][n];
		}
	}
	free(z);
	free(phi);
	free(alpha);
	free(theta);


	//*************************************
	//double lhood2;
	time(&t1);
	iteration = 1;
	sprintf(filename, "%s/test%03d", dir,iteration);
	printf("%s\n",filename);
	write_gibbslda_model(model, filename, corpus, mc);

	prev_lhood0 = 0.0;
	prev_lhood1 = 0.0;
	//for (iteration = 0; iteration < MCSIZE; iteration++){
	iteration = 2;
	do{
		//printf("***** MCMC ITERATION %d *****\n", iteration);

		mcsample(corpus, model, ss, mc, &lhood0, &lhood1, 1);

		conv0 = (prev_lhood0 - lhood0)/prev_lhood0;
		conv1 = (prev_lhood1- lhood1)/prev_lhood1;
		prev_lhood0 = lhood0;
		prev_lhood1 = lhood1;

		time(&t2);

		//sprintf(filename, "%s/%03d", dir,1);
		//write_gibbslda_model(model, filename, corpus, mc);

		if ((iteration%mc->b) == 0){
			printf("***** MCMC ITERATION %d *****\n", iteration);
			fprintf(lhood_fptr, "%d %5.5e %5.5e %5.5e %5.5e %d %5ld %5.5e\n",iteration, lhood0, conv0, lhood1, conv1,
					mc->change, (int)t2-t1, mc->mcse);
			fflush(lhood_fptr);
		}
		iteration ++;
		if ((iteration > MCSIZE) || ((iteration > 5*mc->b) && (mc->mcse < 0.03)))
			break;

	}while(1);
	fclose(lhood_fptr);

	sprintf(filename, "%s/testfinal", dir);
	write_gibbslda_model(model, filename, corpus, mc);

}


gibbslda_model* new_gibbslda_model(int ntopics, int nterms, int ndocs, double alpha, double nu)
{
	int n, j, d;

	gibbslda_model* model = malloc(sizeof(gibbslda_model));
	model->D = ndocs;
    model->m = ntopics;
    model->n = nterms;
    model->nu = nu;
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
    model->nu = nu;
    model->alpha = alpha;

    return(model);
}


gibbslda_mc * new_gibbslda_mc(int ndocs, int ntopics, int nterms)
{
	int n, j, d;

    gibbslda_mc * mc = malloc(sizeof(gibbslda_mc));

    mc->n = 0;
    mc->a = 0;
    mc->b = 50.0;
    mc->mcse = 0.0;
    mc->mcse_num = 0.0;
    mc->beta = malloc(sizeof(double*)*ntopics);
    mc->theta = malloc(sizeof(double*)*ntopics);
    mc->ybeta = malloc(sizeof(double*)*ntopics);
    mc->ytheta = malloc(sizeof(double*)*ntopics);
    mc->sbeta2 = malloc(sizeof(double*)*ntopics);
    mc->stheta2 = malloc(sizeof(double*)*ntopics);
    for (j = 0; j < ntopics; j++){
    	mc->beta[j] = malloc(sizeof(double)*nterms);
    	mc->ybeta[j] = malloc(sizeof(double)*nterms);
    	mc->sbeta2[j] = malloc(sizeof(double)*nterms);
    	for (n = 0; n < nterms; n++){
    		mc->beta[j][n] = 0.0;
    		mc->ybeta[j][n] = 0.0;
    		mc->sbeta2[j][n] = 0.0;
    	}
    	mc->theta[j] = malloc(sizeof(double)*ndocs);
    	mc->ytheta[j] = malloc(sizeof(double)*ndocs);
    	mc->stheta2[j] = malloc(sizeof(double)*ndocs);
    	for (d = 0; d < ndocs; d++){
    		mc->theta[j][d] = 0.0;
    		mc->ytheta[j][d] = 0.0;
    		mc->stheta2[j][d] = 0.0;
    	}
    }
	return(mc);
}

/*
 * create and delete sufficient statistics
 *
 */

gibbslda_ss * new_gibbslda_ss(gibbslda_model* model)
{
	int j, n, d;
	gibbslda_ss * ss;
    ss = malloc(sizeof(gibbslda_ss));
	ss->m = malloc(sizeof(double*)*model->m);
	ss->t = malloc(sizeof(double*)*model->m);
	for (j = 0; j < model->m; j++){
		ss->t[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			ss->t[j][n] = 0.0;
		}
		ss->m[j] = malloc(sizeof(double)*model->D);
		for (d = 0; d < model->D; d++){
			ss->m[j][d] = 0.0;
		}
	}

    return(ss);
}



gibbslda_corpus* read_data(const char* data_filename)
{
	int OFFSET = 0;
    FILE *fileptr;
    int length, count, word, n, nd, nw, corpus_total = 0;
    int total, i;
    int* wrds = malloc(sizeof(int)*100000);
    int* cnts = malloc(sizeof(int)*100000);

    gibbslda_corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(gibbslda_corpus));
    fileptr = fopen(data_filename, "r");
    nd = 0; nw = 0;
    c->docs = malloc(sizeof(document) * 1);
    while ((fscanf(fileptr, "%10d", &length) != EOF)){

	c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
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
	c->docs[nd].yz = malloc(sizeof(double*)*total);
	c->docs[nd].sz2 = malloc(sizeof(double*)*total);
	for (n = 0; n < c->docs[nd].total; n++){
		c->docs[nd].words[n] = wrds[n];
		c->docs[nd].counts[n] = cnts[n];
		c->docs[nd].z[n] = 0;
		c->docs[nd].mcz[n] = 0.0;
		c->docs[nd].yz[n] = 0.0;
		c->docs[nd].sz2[n] = 0.0;
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

int max_corpus_length(gibbslda_corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->ndocs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}


void write_gibbslda_model(gibbslda_model * model, char * root,gibbslda_corpus * corpus, gibbslda_mc* mc)
{
    char filename[200];
    FILE* fileptr;
    int n, j, d;

    //beta
    sprintf(filename, "%s.beta", root);
    fileptr = fopen(filename, "w");
    for (n = 0; n < model->n; n++){
    	for (j = 0; j < model->m; j++){
    		fprintf(fileptr, "%.10lf ",log(mc->beta[j][n]));
    	}
    	fprintf(fileptr, "\n");
    }
    fclose(fileptr);

    //theta
	sprintf(filename, "%s.theta", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%5.10lf ", log(mc->theta[j][d]));
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
	fprintf(fileptr,"T %d \n",(int)mc->n);
	fclose(fileptr);

}

gibbslda_model* load_model(char* model_root, int ndocs){

	char filename[100];
	FILE* fileptr;
	int j, n, num_topics, num_terms, num_docs;
	//float x;
	double y, alpha, nu;

	gibbslda_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fscanf(fileptr, "alpha %lf\n", &alpha);
	fscanf(fileptr, "nu %lf\n", &nu);
	fclose(fileptr);

	model  = new_gibbslda_model(num_topics, num_terms, ndocs, alpha, nu);
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


void corpus_initialize_model(gibbslda_mc* mc, gibbslda_model* model, gibbslda_corpus* corpus, gibbslda_ss* ss)
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

	mc->n = 1;
	mc->a = 1;

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
			//mc->beta[j][n] = model->beta[j][n];
			//mc->ybeta[j][n] = model->beta[j][n];
		}
	}

	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = 1.0/((double)model->m);
			//model->mctheta[j][d] = model->theta[j][d];
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
			mc->theta[j][d] = model->theta[j][d];
			mc->ytheta[j][d] = model->theta[j][d];
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
			mc->beta[j][n] = model->beta[j][n];
			mc->ybeta[j][n] = model->beta[j][n];
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



void random_initialize_model(gibbslda_mc * mc, gibbslda_model * model, gibbslda_corpus* corpus, gibbslda_ss* ss){

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

	mc->n = 1;

	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			nu[n] = model->nu;
		}
		gsl_ran_dirichlet (r, model->n, nu, beta);
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = beta[n];
			mc->beta[j][n] = beta[n];
			mc->ybeta[j][n] = model->beta[j][n];
		}
	}

	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha;
		}
		gsl_ran_dirichlet (r, model->m, alpha, theta);
		for (j = 0; j < model->m; j++){
			model->theta[j][d] = theta[j];
			mc->theta[j][d] = theta[j];
			mc->ytheta[j][d] = theta[j];
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

