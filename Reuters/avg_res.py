import numpy as np

resfile = 'R8_results.txt'
fp = open(resfile,'w+')
fp.write('\n')
fp.close()

method = 'mlelda'
mres = np.loadtxt(method+'_mainres.txt')
bsres = np.loadtxt(method+'_res.txt')
fp = open(resfile, 'ab')
fp.write(method+'\n')
np.savetxt(fp, mres.reshape(1,len(mres)),'%f')
n = bsres.shape[0]
fp.write('BS samples: %d\n' %n)
fp.write('MC std err: %f\n' %float(np.std(bsres[:,2])/np.sqrt(n)))
m = bsres.shape[1]
np.savetxt(fp, np.mean(bsres,0).reshape(1,m),'%.10f')
np.savetxt(fp, np.std(bsres,0).reshape(1,m),'%.10f')
fp.write('\n\n')
fp.close()

method = 'vblda'
mres = np.loadtxt(method+'_mainres.txt')
bsres = np.loadtxt(method+'_res.txt')
fp = open(resfile, 'ab')
fp.write(method+'\n')
np.savetxt(fp, mres.reshape(1,len(mres)),'%f')
n = bsres.shape[0]
fp.write('BS samples: %d\n' %n)
fp.write('MC std err: %f\n' %float(np.std(bsres[:,2])/np.sqrt(n)))
m = bsres.shape[1]
np.savetxt(fp, np.mean(bsres,0).reshape(1,m),'%.10f')
np.savetxt(fp, np.std(bsres,0).reshape(1,m),'%.10f')
fp.write('\n\n')
fp.close()

method = 'gibbslda'
mres = np.loadtxt(method+'_mainres.txt')
bsres = np.loadtxt(method+'_res.txt')
fp = open(resfile, 'ab')
fp.write(method+'\n')
np.savetxt(fp, mres.reshape(1,len(mres)),'%f')
n = bsres.shape[0]
fp.write('BS samples: %d\n' %n)
fp.write('MC std err: %f\n' %float(np.std(bsres[:,2])/np.sqrt(n)))
m = bsres.shape[1]
np.savetxt(fp, np.mean(bsres,0).reshape(1,m),'%.10f')
np.savetxt(fp, np.std(bsres,0).reshape(1,m),'%.10f')
fp.write('\n\n')
fp.close()

method = 'cgibbslda'
mres = np.loadtxt(method+'_mainres.txt')
bsres = np.loadtxt(method+'_res.txt')
fp = open(resfile, 'ab')
fp.write(method+'\n')
np.savetxt(fp, mres.reshape(1,len(mres)),'%f')
n = bsres.shape[0]
fp.write('BS samples: %d\n' %n)
fp.write('MC std err: %f\n' %float(np.std(bsres[:,2])/np.sqrt(n)))
m = bsres.shape[1]
np.savetxt(fp, np.mean(bsres,0).reshape(1,m),'%.10f')
np.savetxt(fp, np.std(bsres,0).reshape(1,m),'%.10f')
fp.write('\n\n')
fp.close()
