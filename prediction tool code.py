#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numba import jit
import numpy as np
import numpy.linalg as LA
import scipy.linalg as LG
import pandas as pd
@jit
def solve_l1l2(W,lambda1):
    nv=W.shape[1]#the number of columns in W
    F=W.copy()
    for p in range(nv):
        nw=LA.norm(W[:,p],"fro")
        if nw>lambda1:
            F[:,p]=(nw-lambda1)*W[:,p]/nw
        else:F[:,p]=np.zeros((W[:,p].shape[0],1))
    return F
 
a=np.mat(np.zeros((495,383)))
AA=a.astype(int)

#read the numbered miRNA-disease interation data
b=np.loadtxt(r'.\Data\knowndiseasemirnainteraction.txt')
B=b.astype(int)

#construct the adjacency matrix AA
for x in B:
    AA[x[0]-1,x[1]-1]=1
#read the disease semantic similarity 1
c1=np.loadtxt(r'.\Data\disease semantic similarity 1.txt')

#read the disease semantic similarity 2
c2=np.loadtxt(r'.\Data\disease semantic similarity 2.txt')

#read the weighted disease semantic similarity
c=np.loadtxt(r'.\Data\weighted disease semantic similarity.txt')
C=0.5*(c1+c2)#the average disease semantic similarity

#read the miRNA functional similarity
D=np.loadtxt(r'.\Data\miRNA functional similarity.txt')

#read the weighted miRNA functional similarity 
d=np.loadtxt(r'.\Data\weighted miRNA functional similarity.txt')

A=AA.copy()

#Initialization of parameters
alpha=0.1
J=np.mat(np.zeros((383,383)))
X=np.mat(np.zeros((383,383)))
E=np.mat(np.zeros((495,383)))
Y1=np.mat(np.zeros((495,383)))
Y2=np.mat(np.zeros((383,383)))
mu=10**-4
max_mu=10**10
rho=1.1
epsilon=10**-6

while True:
    [U,sigma1,V]=LG.svd(X+Y2/mu,lapack_driver='gesvd')
    G=[sigma1[k] for k in range(len(sigma1)) if sigma1[k]>1/mu]
    svp=len(G)
    if svp>=1:
        sigma1=sigma1[0:svp]-1/mu
    else:
        sigma1=[0]
        svp=1
    J=np.mat(U[:,0:svp])*np.mat(np.diag(sigma1))*np.mat(V[0:svp,:])
    ATA=A.T*A
    X=(ATA+np.eye(383)).I*(ATA-A.T*E+J+(A.T*Y1-Y2)/mu)
    temp1=A-A*X
    E=solve_l1l2(temp1+Y1/mu,alpha/mu)
    Y1=Y1+mu*(temp1-E)
    Y2=Y2+mu*(X-J)
    mu=min(rho*mu,max_mu)
    if LA.norm(temp1-E,np.inf)<epsilon and LA.norm(X-J,np.inf)<epsilon:break
P=A*X

#calculate the Gaussian interaction profile kernel similarity KD and the integrated similarity SD for diseases
gamad1=1
sum1=0
for nm in range(383):
    sum1=sum1+LA.norm(P[:,nm],"fro")**2
gamaD1=gamad1*383/sum1
KD=np.mat(np.zeros((383,383)))
for ab in range(383):
    for ba in range(383):
        KD[ab,ba]=np.exp(-gamaD1*LA.norm(P[:,ab]-P[:,ba],"fro")**2)       
SD=np.multiply((C+KD)*0.5,c)+np.multiply(KD,1-c)

#the normalization of SD
SD1=SD.copy()
for nn1 in range(383):
    for nn2 in range(383):
        SD[nn1,nn2]=SD[nn1,nn2]/(np.sqrt(np.sum(SD1[nn1,:]))*np.sqrt(np.sum(SD1[nn2,:])))

#calculate the Gaussian interaction profile kernel similarity KM and the integrated similarity SM for miRNAs
gamad2=1
sum2=0
for mn in range(495):
    sum2=sum2+LA.norm(P[mn,:],"fro")**2
gamaD2=gamad2*495/sum2
KM=np.mat(np.zeros((495,495)))
for cd in range(495):
    for dc in range(495):
        KM[cd,dc]=np.exp(-gamaD2*LA.norm(P[cd,:]-P[dc,:],"fro")**2)
SM=np.multiply((D+KM)*0.5,d)+np.multiply(KM,1-d)

#the normalization of SM
SM1=SM.copy()
for mm1 in range(495):
    for mm2 in range(495):
        SM[mm1,mm2]=SM[mm1,mm2]/(np.sqrt(np.sum(SM1[mm1,:]))*np.sqrt(np.sum(SM1[mm2,:])))

#calculate the score matrix S
S=np.mat(np.random.rand(495,383))
Si=0.4*SM*S*SD+0.6*P
while LA.norm(Si-S,1)>10**-6:
    S=Si
    Si=0.4*SM*S*SD+0.6*P
S=np.array(S)


disease_bianhao=np.genfromtxt(r'.\Data\disease number.txt',dtype=str,delimiter='\t')
miRNA_bianhao=np.loadtxt(r'.\Data\miRNA number.txt',dtype=bytes).astype(str)
disease_yanzheng=np.genfromtxt(r'.\Data\investigated diseases.txt',dtype=str,delimiter='\t')
dbDEMCdatebase=np.genfromtxt(r'.\Data\known miRNA disease associations in dbDEMC.txt',dtype=str,delimiter='\t')
miR2Diseasedatebase=np.genfromtxt(r'.\Data\known miRNA disease associations in miR2Disease.txt',dtype=str,delimiter='\t')


i_disease1=input("please input the disease number of your intreast(1<=number<=383) or 'exit' to terminate the program:")
while i_disease1!='exit':
    i_disease=int(i_disease1)
    while i_disease<=0 or i_disease>=384:
        print("input error")
        i_disease1=input("please input the disease number of your intreast(1<=number<=383) or 'exit' to terminate the program:")
        i_disease=int(i_disease1)
    connectmiRNA = np.argwhere(A[:,i_disease-1]==0)[:,0]
    miRNAranknumbercode_all=np.argsort(-S[:,i_disease-1])
    miRNAranknumbercode_0 = [y for y in miRNAranknumbercode_all if y in connectmiRNA]
        
    miRNAscorerank = S[:,i_disease-1][miRNAranknumbercode_0]
    miRNArankname = miRNA_bianhao[miRNAranknumbercode_0,1]
    diseasecode = []
    for i in range(miRNAscorerank.shape[0]):
        diseasecode.append(disease_bianhao[i_disease-1,1])
    diseasecode_pd =pd.Series(diseasecode)
    miRNArankname_pd=pd.Series(miRNArankname)
    miRNAscorerank_pd =pd.Series(miRNAscorerank)
    prediction_out = pd.concat([diseasecode_pd,miRNArankname_pd,miRNAscorerank_pd],axis=1)
    prediction_out.columns=['Disease','miRNA','Score']
    prediction_out.to_excel(disease_bianhao[i_disease-1,1]+'.xlsx', sheet_name='Sheet1',index=False)
    i_disease1=input("please input the disease number of your intreast(1<=number<=383) or 'exit' to terminate the program:")
print('Thank you for using our tools!')

    

            
            
                
            
    



    
