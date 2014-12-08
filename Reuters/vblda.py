import os
import re
import numpy as np
import random as random
import funcs

codepath = '/gpfs/home/hus152/work/stat540/Final_max/vblda/vblda'
dirpath = 'vblda'
resfile = 'vblda_res.txt'
mainresfile = 'vblda_mainres.txt'
seedfile1 = 'vblda_seed1.txt'
seedfile2 = 'vblda_seed2.txt'

fp = open(mainresfile,'w+')
fp.write('')
fp.close()
fp = open(resfile,'w+')
fp.write('')
fp.close()

seed = 1399677451
np.random.seed(seed)

M = 8
C = 8
alpha = 0.1
nu = 0.1;

#*************** main run
trdocs = 'data/trdocs_7k.txt'
trlbls = 'data/trlbls_7k.txt'
tdocs = 'data/tdocs_7k.txt'
tlbls = 'data/tlbls_7k.txt'
os.system('mkdir -p ' + dirpath)
if 0:
	# load gtrtheta and gbeta
	gtrtheta = np.loadtxt(dirpath+'/gtrtheta.txt')
	D = gtrtheta.shape[0]
	gtrtheta = gtrtheta/(np.sum(gtrtheta,1).reshape(D,1))
	gbeta = np.loadtxt(dirpath+'/gbeta.txt')
	gbeta = gbeta/np.sum(gbeta,0)
	N = gbeta.shape[0]
	# read test theta
	gttheta = np.loadtxt(dirpath+'/gttheta.txt')
	D = gttheta.shape[0]
	gttheta = gttheta/(np.sum(gttheta,1).reshape(D,1))
	# compute ccr
	(trccr,gtpc_lbl_distn) = funcs.classifier_training(trlbls,gtrtheta,C,M)
	tccr = funcs.classifier_test(tlbls,gtpc_lbl_distn,gttheta)

	state1 = np.loadtxt(seedfile1 ,delimiter=', ')
	state2 = np.loadtxt(seedfile2)
	rndst = np.random.get_state()
	rndstate = (rndst[0],np.uint(state2),np.int(state1[0]),np.int(state1[1]),np.float(state1[2]))
	np.random.set_state(rndstate)

if 1:
	# train model
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' est ' + trdocs + ' ' + str(M) + ' seeded '+ dirpath
	cmdtxt += ' ' + str(alpha) + ' ' + str(nu)
	os.system(cmdtxt)
	# read train time
	runtime = np.loadtxt(dirpath+'/likelihood.dat')[-1,3]
	# load gtrtheta and gbeta
	gtrtheta = np.loadtxt(dirpath+'/final.theta')
	D = gtrtheta.shape[0]
	gtrtheta = gtrtheta/(np.sum(gtrtheta,1).reshape(D,1))
	np.savetxt(dirpath+'/gtrtheta.txt',gtrtheta)
	gbeta = np.loadtxt(dirpath+'/final.beta')
	gbeta = gbeta/np.sum(gbeta,0)
	np.savetxt(dirpath+'/gbeta.txt',gbeta)
	N = gbeta.shape[0]
	# test model
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' inf ' + tdocs + ' ' + dirpath + '/final ' + dirpath
	os.system(cmdtxt)
	# read test theta
	gttheta = np.loadtxt(dirpath+'/testfinal.theta')
	D = gttheta.shape[0]
	gttheta = gttheta/(np.sum(gttheta,1).reshape(D,1))
	np.savetxt(dirpath+'/gttheta.txt',gttheta)
	# compute likelihood on test
	tlkh = funcs.compute_lkh(tdocs, gbeta, gttheta)
	# compute ccr
	(trccr,gtpc_lbl_distn) = funcs.classifier_training(trlbls,gtrtheta,C,M)
	tccr = funcs.classifier_test(tlbls,gtpc_lbl_distn,gttheta)
	# write results
	temp = np.array([runtime, tlkh, 100.0*tccr]).reshape(1,3)
	fp = open(mainresfile,'ab')
	np.savetxt(fp,temp,'%d %5.5f %5.5f')
	fp.close()	

# read length of train docs
trdoclen = list()
fp = open(trdocs,'r')
d = 0
while True:
    doc = fp.readline()
    if (len(doc)==0):
        break
    cnts = re.findall('[0-9]*:([0-9]*)',doc)
    cs = [np.int(x) for x in cnts]
    trdoclen.append(np.sum(cs))
    d += 1 
Dtr = d   
fp.close() 
# read length of test docs
tdoclen = list()
fp = open(tdocs,'r')
d = 0
while True:
    doc = fp.readline()
    if (len(doc)==0):
        break
    cnts = re.findall('[0-9]*:([0-9]*)',doc)
    cs = [np.int(x) for x in cnts]
    tdoclen.append(np.sum(cs))
    d += 1 
Dt = d   
fp.close() 

##****************************** parametric bootstrap
T = 500
t0 = 0
#res = np.loadtxt(resfile)
#t0 = res.shape[0]
for b in range(t0,T):

	# save seed
	s = np.random.get_state()
	fp1 = open(seedfile1,'w')
	fp2 = open(seedfile2,'wb')
	fp1.write('%d, %d, %f' %(s[2],s[3],s[4]))
	fp1.close()
	np.savetxt(fp2, s[1].reshape(1,len(s[1])),'%u')
	fp2.close()

	print('bootstrap iteration %d' %b)
	#********** generate data
	trdocs = dirpath + '/trdocs.txt'
	#trlbls = 'train-label.dat' # the same labels as the original file b/c tpc proportions are the same
	tdocs = dirpath + '/tdocs.txt'
	#tlbls = 'test-label.dat'
	
	# generate train docs
	fp1 = open(trdocs,'w+')
	wrdchk = np.zeros(N)
	docwrds = [-1]*N
	doccnts = [-1]*N
	docstr = ''
	for d in range(Dtr):
		l = 0
		print(d)
		nd = trdoclen[d]
		for n in range(nd):
			z = np.random.choice(M, 1, p = gtrtheta[d,:])[0]
			wid = np.random.choice(N, 1, p = gbeta[:,z])[0]
			try:
				ind = docwrds[:l+1].index(wid)
				doccnts[ind] += 1				
			except ValueError:
				docwrds[l] = wid
				doccnts[l] = 1
				l += 1
				wrdchk[wid] = 1
		if d == (Dtr-1): #add missing words to the last doc (sometimes 1 word is missing)
			for x,wid in enumerate(np.where(wrdchk==0)[0]):
				wrdchk[x] = 1
				docwrds[l] = wid
				doccnts[l] = 1
				l += 1	
		#fp1.write(str(l)+' ')	
		docstr += str(l)+' '	
		for i in range(l):
			#fp1.write(str(docwrds[i])+':'+str(doccnts[i])+' ')
			docstr += (str(docwrds[i])+':'+str(doccnts[i])+' ')
		docstr += '\n'
	fp1.write(docstr)
	fp1.close()
	print('tr done')
	# generate test docs
	fp1 = open(tdocs,'w+')
	docstr = ''
	for d in range(Dt):
		l = 0
		nd = tdoclen[d]
		for n in range(nd):
			z = np.random.choice(M, 1, p = gttheta[d,:])[0]
			wid = np.random.choice(N, 1, p = gbeta[:,z])[0]
			if wrdchk[wid] == 0:
				continue
			try:
				ind = docwrds[:l+1].index(wid)
				doccnts[ind] += 1				
			except ValueError:
				docwrds[l] = wid
				doccnts[l] = 1
				l += 1
		docstr += str(l)+' '
		for i in range(l):
			docstr += str(docwrds[i])+':'+str(doccnts[i])+' '
		docstr += '\n'
	fp1.write(docstr)	
	fp1.close()
	print('t done')
	#**********

	# train model
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' est ' + trdocs + ' ' + str(M) + ' seeded '+ dirpath
	cmdtxt += ' ' + str(alpha) + ' ' + str(nu)
	#print(cmdtxt)
	os.system(cmdtxt)

	# read train time
	runtime = np.loadtxt(dirpath+'/likelihood.dat')[-1,3]

	# load theta and beta
	trtheta = np.loadtxt(dirpath+'/final.theta')
	D = trtheta.shape[0]
	trtheta = trtheta/(np.sum(trtheta,1).reshape(D,1))
	beta = np.loadtxt(dirpath+'/final.beta')
	beta = beta/np.sum(beta,0)

	# test model
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' inf ' + tdocs + ' ' + dirpath + '/final ' + dirpath
	#print(cmdtxt)
	os.system(cmdtxt)

	# read test theta
	ttheta = np.loadtxt(dirpath+'/testfinal.theta')
	D = ttheta.shape[0]
	ttheta = ttheta/(np.sum(ttheta,1).reshape(D,1))

	# compute likelihood on test
	tlkh = funcs.compute_lkh(tdocs, beta, ttheta)

	# compute ccr
	(trccr,tpc_lbl_distn) = funcs.classifier_training(trlbls,trtheta,C,M)
	tccr = funcs.classifier_test(tlbls,tpc_lbl_distn,ttheta)

	# compute beta-mse, theta-mse
	beta_mse = np.mean((beta[:,np.argsort(np.argmax(tpc_lbl_distn,1))]-gbeta[:,np.argsort(np.argmax(gtpc_lbl_distn,1))])**2.);
	theta_mse = np.mean((ttheta[:,np.argsort(np.argmax(tpc_lbl_distn,1))]-gttheta[:,np.argsort(np.argmax(gtpc_lbl_distn,1))])**2.);	


	# write results
	temp = np.array([runtime, tlkh, 100.0*tccr, beta_mse, theta_mse]).reshape(1,5)
	fp = open(resfile,'ab')
	np.savetxt(fp,temp,'%d %5.5f %5.5f %.5e %.5e')
	fp.close()	


	#check termination criterion
	if b > 50:
		temp = np.loadtxt(resfile)
		l1 = temp.shape[0]
		mcstderr = np.std(temp[:,2])/np.sqrt(l1)
		if (mcstderr < 0.05):
			break

