# bands.py
'''Some functions for computing band structure.
	Natural units used s.t. hbar=1, 2*(mass of atom)=1
	This leaves a length scale free.  Consequently, whatever units u you
	use for wavevectors k, energy will then be in units u^2 hbar^2/2m
	'''

from __future__ import division,print_function
from numpy import *
from scipy import *
from pylab import *
from scipy import sparse
from scipy.linalg import expm

###################### 1 Dimension ###########################

def tridiag1(k,b,amp,n,M=False):
	'''Returns a tridiagonal (2n+1)x(2n+1) matrix representing the 
		1D optical lattice Schrodinger equation for quasimomentum k.
		* k is quasimomentum
		* b is the reciprocal lattice basis vector
		* amp is the amplitude of the lattice
		* n determines the size of the matrix (i.e. momentum cutoff)
		'''
	dia = (k+b*arange(-n,n+1))**2
	up  = amp/2. * ones(2*n,float)
	dn  = amp/2. * ones(2*n,float)
	
	if M:
		return diag(up,1)+diag(dia)+diag(dn,-1)
	else:
		return up,dia,dn
	#return tridiag(up,dia,dn)	# Check tridiag or equivalent

def eigeng1(k,b,amp,nbands,n=False):
	'''Returns nbands number of eigenenergies for quasimomentum k.
		b is the reciprocal lattice basis vector, amp is lattice
		amplitude, n is momentum cutoff (all as above).'''
	if not n:
		n = nbands
	if not iterable(k):
		k = array([k])
	eigs = zeros((len(k),nbands),float)
	for i in range(len(k)):
		M = tridiag1(k[i],b,amp,n,True)
		eigs[i,:] = sort(linalg.eigvalsh(M))[:nbands]
	return eigs

def eigs1(k,b,amp,nbands,n=False,returnM=False):
	'''Returns nbands number of eigenenergies and eigenvectors
		for quasimomentum k.
		b is the reciprocal lattice basis vector, amp is lattice
		amplitude, n is momentum cutoff (as above).
		Unlike eigeng1, k may not be iterable here.
	'''
	if not n:
		n = nbands
	M = tridiag1(k,b,amp,n,True)
	enrg0,evec0 = linalg.eigh(M)
	enrg = enrg0[:nbands]
	evec = evec0[:,:nbands]
	if not returnM:
		return enrg,evec
	else:
		return enrg,evec,M

def vConst(ts,v,bn,k,b,amp,nbands,n=False):
	'''Given initial data c[bn,k]=1, solves for c[m,k] for all times ts.
		v is lattice velocity (in units of hbar/(2m*length scale))
	'''
	if not n:
		n = nbands
	enrg,evec,M = eigs1(k,b,amp,nbands,n,True) #.....
	p = diag(diag(M))
	
	cs0 = zeros(nbands,complex)
	cs0[bn] = 1.0
	cs = zeros((nbands,len(ts)),complex)
	cs[:,0] = cs0
	ps = zeros((M.shape[0],len(ts)),complex)

	m1 = mat(evec)
	m2 = -1.0j*m1.H*(M+v*p)*m1
	m3 = 1.0j*p*v
	
	for i in range(1,len(ts)):
		Ex = expm(m2*ts[i])
		T  = expm(m3*ts[i])
		cs[:,i] = Ex.dot(cs0)
		# Get momentum state projections:
		ps[:,i] = T.dot(m1)[:,:nbands].dot(cs[:,i])
	return cs, ps, Ex, T
	

######################## 2 Dimensions ###########################

def dia2(k,bs,amps,n,M=True):
	'''Returns a k-space matrix Hamiltonian for an optical lattice.
		* k is a vector in the 1st Brillioun zone.
		* bs is a 2-tuple of k-space basis vectors
		* amps is a 3-tuple of amplitudes.  amps[0] and amps[1] 
			correspond to bs[0] and bs[1], and amps[2] corresponds to
			-bs[1]-bs[2]
		* n is the wavevector cutoff
		By default (M=True), a matrix is returned.  The matrix operates
		on vectors c[i] defined so that c[i] is the coefficient of 
		exp[ ((-n+(i mod(2n+1)))*bs[0]+k).r) + (((-n+floor(i/(2n+1)))*bs[1]+k).r) ]]
		in a Fourier expansion of the wavefunction.
		
		[Formerly, I had written 
		(-n+(i mod(2n+1)))exp((bs[0]+k).r) + (-n+floor(i/(2n+1)))exp((bs[1]+k).r)
		 instead of the above expression, but I think this was incorrect.]
		'''
	
	N=2*n+1
	idx = arange(N**2)
	# dia is the kinetic part of the Hamiltonian
	dia = (k[0] + bs[0][0]*(-n+mod(idx,N))+bs[1][0]*(-n+floor(idx/N)))**2 + \
			(k[1] + bs[0][1]*(-n+mod(idx,N))+bs[1][1]*(-n+floor(idx/N)))**2
	
	# The M's below capture the potential part of the Hamiltonian
	d0 = ones(N,float); d1 = ones(N-1,float);
	Mup = sparse.diags([d0*amps[1]/2.,d1*amps[2]/2.],[0,-1])
	Mdn = sparse.diags([d0*amps[1]/2.,d1*amps[2]/2.],[0,1])
	M0  = sparse.diags([d1*amps[0]/2.,d1*amps[0]/2.],[1,-1])
	Mkin = sparse.diags(dia,0)
	
	# Each k below corresponds to an M above, and will be used in a Kronecker product below
	kup = sparse.diags(d1,1)
	kdn = sparse.diags(d1,-1)
	k0 	= sparse.diags(d0,0)
	
	M = Mkin + sparse.kron(k0,M0) + sparse.kron(kup,Mup) + sparse.kron(kdn,Mdn)
	
	return M

"""		# Obsolete
def eigeng2(k,bs,amps,n,nbands):
	'''Returns nbands number of eigenenergies for quasimomentum k.
		b is the reciprocal lattice basis vector, amp is lattice
		amplitude, n is momentum cutoff (all as above).'''
	k = array(matrix(k))		# Sure way to make k have 2 dimensions
	eigs = zeros((k.shape[0],nbands),float)
	Amin = -abs(amps[0])-abs(amps[1])-abs(amps[2])
	for i in range(k.shape[0]):
		M = dia2(k[i],bs,amps,n,True)
		eigs[i,:] = sort(sparse.linalg.eigsh(M,nbands,sigma=Amin,return_eigenvectors=False))
	return eigs
"""

def eigeng2(k,bs,amps,nbands,n):
	'''Returns nbands number of eigenenergies for quasimomentum k.
		b is the reciprocal lattice basis vector, amp is lattice
		amplitude, n is momentum cutoff (all as above).'''
	k = array(k); D = len(k.shape)
	if D==1:			# Need to ensure k can be iterated over
		k = expand_dims(k,axis=0)
		D = 2
	
	eigs = zeros(k.shape[:-1]+(nbands,),float)
	Amin = -abs(amps[0])-abs(amps[1])-abs(amps[2])
	for i in ndindex(k.shape[:-1]):
		M = dia2(k[i],bs,amps,n,True)
		eigs[i] = sort(sparse.linalg.eigsh(M,nbands,sigma=Amin,return_eigenvectors=False))
	return eigs

def eigs2(k,bs,amps,nbands,n=None):
	'''Returns nbands number of eigenvectors/values for quasimomentum k.
		bs are reciprocal lattice basis vectors (there should be 2), 
		amps are amplitudes (there should be three), and n (if supplied)
		is the wavevector cutoff (so eigenvectors have length 2n+1).  
		If not supplied, n is taken to be nbands.
		k may not be iterable.
		'''
	if n is None:
		n = nbands
	M = dia2(k,bs,amps,n)
	eigvals,eigvecs = sparse.linalg.eigsh(M,nbands,sigma=Amin)
	s = argsort(eigvals)
	eigvals = (eigvals[s])[:nbands]
	eigvecs = (eigvecs[:,s])[:,:nbands]
	return eigvals,eigvecs