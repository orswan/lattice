# mapLZtunnel.py
from __future__ import print_function,division

from numpy import *
from scipy import *
from pylab import *
import lattice2D as l2

gm=10.
T=arange(0,10,.1)
idx = 50
n=10
aas = arange(3.,45.,3.)
naas = len(aas)
Ers = arange(1.,10.,2.)
nErs = len(Ers)
loss = zeros((naas,nErs))
bandgap = zeros((naas,nErs))

for i in range(naas):
	for j in range(nErs):
		c,px,py,h = l2.avUniform(aa=aas[i],Er=Ers[j],gm=gm,T=T,n=n,plt='')
		temp = l2.LZview(c,px,py,h,T,plt='',ret=True)
		loss[i,j] = amax(temp[0][:,1])
		bandgap[i,j] = amin(temp[1][:,1]-temp[1][:,0])

