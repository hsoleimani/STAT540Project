import numpy as np
import re

def classifier_training(tr_label_file,theta,C,M):
    ## read training labels
    tr_label = open(tr_label_file,'r')
    tpc_lbl_distn = np.zeros((M,C))
    
    #learning topic_label_multinomials
    d = 0
    while True:
        labelline = tr_label.readline()
        if len(labelline)==0:
            break
        lbl = int(labelline.split()[0]) # assuming single label for now/ should start from zero
        #print(lbl)
        tpc_lbl_distn[:,lbl] += theta[d,].T
        
        d += 1  
    
    tpc_lbl_distn = tpc_lbl_distn/(np.tile(np.sum(tpc_lbl_distn,1),(C,1)).T)
    
    # ccr on the training set
    tr_label.seek(0)
    ccr = 0.0
    d = 0
    while True:
        labelline = tr_label.readline()
        if len(labelline)==0:
            break
        lbl = int(labelline.split()[0]) # assuming single label for now/ should start from zero
        #if lbl!=0 and lbl!=1:
        #    continue
        dlp = np.dot(theta[d,:],tpc_lbl_distn)
        pred_lbl = np.argmax(dlp)
        if (pred_lbl==lbl):
            ccr += 1.0
        #else:
        #    print(d)
        d += 1
        
    ccr = ccr/float(d)
    tr_label.close()
    return(ccr,tpc_lbl_distn)

def classifier_test(label_file,tpc_lbl_distn,theta):
    
    labelfile = open(label_file,'r')
    ccr = 0.0
    d = 0
    while True:
        labelline = labelfile.readline()
        if len(labelline)==0:
            break
        lbl = int(labelline.split()[0]) # assuming single label for now/ should start from zero
        dlp = np.dot(theta[d,:],tpc_lbl_distn)
        pred_lbl = np.argmax(dlp)
        if (pred_lbl==lbl):
            ccr += 1.0
        d += 1
        
    ccr = ccr/float(d)
    labelfile.close()
    return(ccr)

def compute_lkh(docfile, beta, theta):
    
    EPS = 1e-100
    lkh = 0.0
    nterms = beta.shape[0]   
 
    fp = open(docfile,'r')
    d = 0
    while True:
        doc = fp.readline()
        if (len(doc)==0):
            break
        nd = 0.0
        wrds = re.findall('([0-9]*):[0-9]*',doc)
        cnts = re.findall('[0-9]*:([0-9]*)',doc)
        ws = [np.int(x) for x in wrds]
        cs = [np.float(x) for x in cnts]
        ld = len(wrds)
        nd = np.sum(cs)
        for n in range(ld):
            lkh += float(cnts[n])*np.log(np.dot(theta[d,],beta[int(ws[n]),:])+EPS)
        d += 1    
    fp.close()
    return(lkh)

