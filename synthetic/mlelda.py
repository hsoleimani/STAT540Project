import os
import numpy as np
import random as random
import funcs
import datagen

codepath = '/gpfs/home/hus152/work/stat540/Final_max/mlelda/mlelda'
dirpath = 'mlelda'
resfile = 'mlelda_res.txt'

fp = open(resfile,'w+')
fp.write('')
fp.close()

seed = 1399677451
np.random.seed(seed)

M = 10
C = 10
N = 2000
alpha = 0.1
nu = 0.1;
T = 500

t0 = 0
#res = np.loadtxt(resfile)
#t0 = res.shape[0]
state1 = np.loadtxt('state1.txt',delimiter=', ')
state2 = np.loadtxt('state2.txt')
rndst = np.random.get_state()
for b in range(t0,T):
	
	print(b)
	# read state of random number generator
	rndstate = (rndst[0],np.uint(state2[b,:]),np.int(state1[b,0]),np.int(state1[b,1]),np.float(state1[b,1]))
	np.random.set_state(rndstate)

	# generate data
	os.system('mkdir -p ' + dirpath)
	datagen.datagen(dirpath)
	trdocs = dirpath + '/trdocs_'+str(N)+'.txt'
	trlbls = dirpath + '/trlbls_'+str(N)+'.txt'
	tdocs = dirpath + '/tdocs_'+str(N)+'.txt'
	tlbls = dirpath + '/tlbls_'+str(N)+'.txt'
	odocs = dirpath + '/odocs_'+str(N)+'.txt'
	hdocs = dirpath + '/hdocs_'+str(N)+'.txt'

	# train model
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' est ' + trdocs + ' ' + str(M) + ' seeded '+ dirpath
	print(cmdtxt)
	os.system(cmdtxt)

	# read train time
	runtime = np.loadtxt(dirpath+'/likelihood.dat')[-1,3]

	# load theta and beta
	trtheta = np.exp(np.loadtxt(dirpath+'/final.theta'))
	D = trtheta.shape[0]
	trtheta = trtheta/(np.sum(trtheta,1).reshape(D,1))
	beta = np.exp(np.loadtxt(dirpath+'/final.beta'))
	beta = beta/np.sum(beta,0)

	# test model
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' inf ' + tdocs + ' ' + dirpath + '/final ' + dirpath
	#print(cmdtxt)
	os.system(cmdtxt)

	# read test theta
	ttheta = np.exp(np.loadtxt(dirpath+'/testfinal.theta'))
	D = ttheta.shape[0]
	ttheta = ttheta/(np.sum(ttheta,1).reshape(D,1))

	# compute likelihood on test
	tlkh = funcs.compute_lkh(tdocs, beta, ttheta)

	# compute ccr
	(trccr,tpc_lbl_distn) = funcs.classifier_training(trlbls,trtheta,C,M)
	tccr = funcs.classifier_test(tlbls,tpc_lbl_distn,ttheta)

	# compute beta-mse, theta-mse
	gbeta = np.loadtxt(dirpath+'/beta.txt')
	gbeta = gbeta/np.sum(gbeta,0)
	beta_mse = np.mean((beta[:,np.argsort(np.argmax(tpc_lbl_distn,1))]-gbeta)**2.);

	gtheta = np.loadtxt(dirpath+'/theta.txt')
	D = gtheta.shape[0]
	gtheta = gtheta/(np.sum(gtheta,1).reshape(D,1))
	theta_mse = np.mean((ttheta[:,np.argsort(np.argmax(tpc_lbl_distn,1))]-gtheta)**2.);	

	# write results
	temp = np.array([runtime, tlkh, 100.0*tccr, beta_mse, theta_mse]).reshape(1,5)
	fp = open(resfile,'ab')
	np.savetxt(fp,temp,'%d %5.5f %5.5f %.5e %.5e')
	fp.close()	

	#check convergence
	if b > 50:
		temp = np.loadtxt(resfile)
		if (np.std(temp[:,2])/np.sqrt(b)) < 0.05:
			break
	
