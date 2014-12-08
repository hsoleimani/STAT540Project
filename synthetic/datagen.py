import numpy as np
import random as random
import os


def datagen(path):

	M = 10
	N = 2000
	D = M*200
	lam = 100

	beta = np.zeros((N,M))
	n1 = int(np.floor(N*0.01))
	remainingwrds = np.arange(N)
	highprobvec = np.zeros((n1,M+1))
	for j in range(M):
		highprobs = np.random.choice(np.arange(np.int(np.floor(N/M))), n1, replace = False)
		highprobs = highprobs + j*np.int(np.floor(N/M))
		highprobvec[:,j] = highprobs
		beta[:,j] = 0.1*np.random.uniform(0,1,N)
		beta[(highprobs),j] = 0.7 + 0.1*np.random.uniform(0,1,n1)
		#for n in range(N):
		#	if n in highprobs:
		#		beta[n,j] = 0.7 + 0.1*np.random.uniform(0, 1)
		#	else:
		#		beta[n,j] = 0.1*np.random.uniform(0, 1)
		beta[:,j] = beta[:,j]/np.sum(beta[:,j])
		remainingwrds = np.setdiff1d(remainingwrds, highprobs)


	np.savetxt(path+'/beta.txt', beta, '%5.10f')

	#generate trainig docs
	while True:
		fp1 = open(path+'/trdocs_'+str(N)+'.txt','w')
		fp2 = open(path+'/trlbls_'+str(N)+'.txt','w')
		#fp3 = open(path+'/traintheta.txt','w')
		wchk = np.zeros(N)
		for d in range(D):
			j = d%M
			theta = np.ones(M)
			theta[j] = 50.0
			theta = theta/np.sum(theta)
			ld = 0
			nd = np.random.poisson(lam, 1)[0]
			docwrds = [-1]*nd
			doccnts = [-1]*nd
			for i in range(nd):
				z = np.random.choice(M, 1, p = theta)[0]
				w = np.random.choice(N, 1, p = beta[:,z])[0]
				try:
					ind = docwrds[:ld+1].index(w)
					doccnts[ind] += 1				
				except ValueError:
					docwrds[ld] = w
					doccnts[ld] = 1
					ld += 1
					wchk[w] = 1
			fp1.write(str(ld) + ' ')
			for i in range(ld):
				fp1.write(str(docwrds[i])+':'+str(doccnts[i])+' ')
			fp1.write('\n')
			fp2.write(str(j) + '\n')

		fp1.close()
		fp2.close()
		if (np.sum(wchk==0) == 0):
			break


	#generate test docs
	fp1 = open(path+'/tdocs_'+str(N)+'.txt','w')
	fp2 = open(path+'/tlbls_'+str(N)+'.txt','w')
	Dtest = int(np.floor(M*100))
	thetamat = np.zeros((Dtest,M))
	for d in range(Dtest):
		j = d%M
		theta = np.ones(M)
		theta[j] = 50.0
		theta = theta/np.sum(theta)
		thetamat[d,:] = theta
		ld = 0
		nd = np.random.poisson(lam, 1)[0]
		docwrds = [-1]*nd
		doccnts = [-1]*nd
		for i in range(nd):
			z = np.random.choice(M, 1, p = theta)[0]
			w = np.random.choice(N, 1, p = beta[:,z])[0]
			try:
				ind = docwrds[:ld+1].index(w)
				doccnts[ind] += 1				
			except ValueError:
				docwrds[ld] = w
				doccnts[ld] = 1
				ld += 1
		fp1.write(str(ld) + ' ')
		for i in range(ld):
			fp1.write(str(docwrds[i])+':'+str(doccnts[i])+' ')
		fp1.write('\n')
		fp2.write(str(j) + '\n')

	fp1.close()
	fp2.close()
	# save theta
	np.savetxt(path+'/theta.txt', thetamat, '%5.10f')





