import numpy as np

resfile = 'synthetic_results.txt'
fp = open(resfile,'w+')
fp.write('\n')
fp.close()


method = 'mlelda'
res = np.loadtxt(method+'_res.txt')
fp = open(resfile, 'ab')
fp.write(method+'\n')
n = res.shape[0]
fp.write('BS samples: %d\n' %n)
fp.write('MC std err: %f\n' %float(np.std(res[:,2])/np.sqrt(n)))
m = res.shape[1]
np.savetxt(fp, np.mean(res,0).reshape(1,m),'%.10f')
np.savetxt(fp, np.std(res,0).reshape(1,m),'%.10f')
fp.write('\n\n')
fp.close()

method = 'vblda'
res = np.loadtxt(method+'_res.txt')
fp = open(resfile, 'ab')
fp.write(method+'\n')
n = res.shape[0]
fp.write('BS samples: %d\n' %n)
fp.write('MC std err: %f\n' %float(np.std(res[:,2])/np.sqrt(n)))
m = res.shape[1]
np.savetxt(fp, np.mean(res,0).reshape(1,m),'%.10f')
np.savetxt(fp, np.std(res,0).reshape(1,m),'%.10f')
fp.write('\n\n')
fp.close()

method = 'gibbslda'
res = np.loadtxt(method+'_res.txt')
fp = open(resfile, 'ab')
fp.write(method+'\n')
n = res.shape[0]
fp.write('BS samples: %d\n' %n)
fp.write('MC std err: %f\n' %float(np.std(res[:,2])/np.sqrt(n)))
m = res.shape[1]
np.savetxt(fp, np.mean(res,0).reshape(1,m),'%.10f')
np.savetxt(fp, np.std(res,0).reshape(1,m),'%.10f')
fp.write('\n\n')
fp.close()


method = 'cgibbslda'
res = np.loadtxt(method+'_res.txt')
fp = open(resfile, 'ab')
fp.write(method+'\n')
n = res.shape[0]
fp.write('BS samples: %d\n' %n)
fp.write('MC std err: %f\n' %float(np.std(res[:,2])/np.sqrt(n)))
m = res.shape[1]
np.savetxt(fp, np.mean(res,0).reshape(1,m),'%.10f')
np.savetxt(fp, np.std(res,0).reshape(1,m),'%.10f')
fp.write('\n\n')
fp.close()
