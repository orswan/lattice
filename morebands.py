# morebands.py

from __future__ import division,print_function

from numpy import *
from scipy import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import numbers
from copy import deepcopy
import itertools

if 1:		# Units.  I'm wrapping this in a conditional so I can block it and collapse it in Notepad++/Geany
	# MKS units:
	hbar = 1.05457e-34
	C = 299792458.0
	eps0 = 8.85418782e-12
	mu0 = 4.e-7*pi
	eta0 = 377					# Impedance of free space
	g = 9.81					# Earth's gravitational acceleration
	M2 = 2*1.455e-25			# 2*Sr mass in kg
	d00 = 461.e-9				# A typical length scale in meters (Sr transition wavelength)
	d0 = d00/(2*pi)				# NOTE: THIS IS THE RIGHT CHARACTERISTIC LENGTH FOR LATTICE RECOIL UNITS!
	k0 = 1./d0					# Sr transition k-vector length
	mu = 2.58e-29				# Sr 461 dipole strength in meter second Amperes
	fSr = C/d0					# Sr 461 transition frequency in Hz					= 6.503 e14
	wSr = fSr*2*pi				# Sr 461 transition angular frequency				= 4.086 e15
	f0 = hbar/(M2*d0**2)		# Sr 461 characteristic frequency hbar/(2*m*d0^2)	= 1.705 e3
	w0 = f0*2*pi				# Sr 461 characteristic angular frequency			= 1.071 e4
	U0 = hbar*f0				# Sr 461 characteristic energy						= 1.798 e-31
	E0 = U0/mu					# Sr 461 characteristic electric field				= 6.970 e-3
	a0 = d0*f0**2				# Sr 461 characteristic acceleration				= 1.340
	Ug = g*d0*M2				# Sr 461 characteristic gravitational energy		= 1.316 e-30 = gSr*U0
	############## Sr 461 units:
	wSr0 = wSr/w0				# Sr 461 frequency in 461 units						= 3.814 e11
	gSr = g/a0					# Gravitational acceleration in 461 units			= .0295
	############## Next are not "natural" but practical values
	delta_typical = 5e12		# Typical (optimized) detuning for 461 nm transition (in Hz)
	dtn0 = delta_typical/f0			# ... in 461 units = 2.93e9
	Etyp = sqrt(2*dtn0*gSr)		# Corresponding typical electric field = 2.07e5 in 461 units

if 1:		# Defaults
	# Default laser parameters:
	sphi = pi/8				# Default angle for the lasers
	sk1 = array([cos(sphi),sin(sphi)])
	sk2 = array([cos(sphi),-sin(sphi)])
	sk3 = array([-1.0,0.0])
	sks = array([sk1,sk2,sk3]).T
	sE1 = Etyp; sE2 = Etyp; sE3 = Etyp; sEs = array([sE1,sE2,sE3])

# Take a Hamiltonian H = p^2/2m + A0 cos(b0*x) + A1 cos(b1*x) + A2 cos((-b0-b1)*x)
# and an eigenmatrix v, and find the angle between v and Hv
def trans(m,lr=0,ud=0):
	"""shift matrix m left/right by lr (+ is right) and up/down by ud (+ is up).
		Values do not wrap around axes, and zeros are used where no value enters from m"""
	M = zeros(m.shape,dtype=m.dtype)
	if lr==0:
		if ud==0:
			M[:] 		= m[:]
		elif ud>0:
			M[:-ud,:]	= m[ud:,:]
		else:
			M[-ud:,:]	= m[:ud,:]
	elif lr>0:
		if ud==0:
			M[:,lr:]	= m[:,:-lr]
		elif ud>0:
			M[:-ud,lr:]	= m[ud:,:-lr]
		else:
			M[-ud:,lr:]	= m[:ud,:-lr]
	else:
		if ud==0:
			M[:,:lr]	= m[:,-lr:]
		elif ud>0:
			M[:-ud,:lr]	= m[ud:,-lr:]
		else:
			M[-ud:,:lr]	= m[:ud,-lr:]
	return M

def operate(ham,v,q):
	""" q is the quasimomentum of the MIDDLE value of eigenmatrix v """
	b0,b1,b2 = [ham.b[:,i] for i in {0,1,2}]
	A0,A1,A2 = [ham.A[i]/2. for i in {0,1,2}]		# NOTE THE /2, which makes these amplitudes for the translation operators
	N = v.shape[0]
	n = (N-1)//2
	m1,m0 = meshgrid(arange(-n,n+1),arange(-n,n+1))
	px = m0*b0[0] + m1*b1[0] + q[0]
	py = m0*b0[1] + m1*b1[1] + q[1]
	P2 = px**2 + py**2
	return P2*v + A0*(trans(v,ud=1)+trans(v,ud=-1)) + A1*(trans(v,lr=1)+trans(v,lr=-1)) + A2*(trans(v,1,-1)+trans(v,-1,1))

def norm(v):
	return sqrt(sum(v.conj()*v))

def dot(v1,v2):
	return sum(v1.conj()*v2)

def cosAngle(v1,v2):
	return dot(v1,v2)/norm(v1)/norm(v2)

def eignorm(ham,v,q):
	return norm(operate(ham,v,q))

def eigdot(ham,v,q):
	return dot(v,operate(ham,v,q))

def eigCosAngle(ham,v,q):
	return cosAngle(v,operate(ham,v,q))

def eij(n,i,j,dtype=complex):
	"""Makes a unit matrix e with e_kl = delta_ij"""
	E = zeros((n,n),dtype=dtype)
	E[i,j]+=1
	return E

def Hmatrix(ham,N,q):
	"""Makes a N^2 by N^2 Hamiltonian matrix the brute force way, using 'operate' above."""
	idxs = reshape(arange(N**2),(N,N),'F')	# This shows how the flattening is done
	H = zeros((N**2,N**2),dtype=complex)
	for i in range(N**2):
		il,jl = where(idxs==i)		# Indices for the left matrix
		il,jl = il[0],jl[0]
		#vl = eij(N,il,jl)			# Don't actually need this: we'll just use the indices directly
		for j in range(N**2):
			ir,jr = where(idxs==j)	# Indices for the right matrix
			ir,jr = ir[0],jr[0]
			vr = eij(N,ir,jr)
			H[i,j] = (operate(ham,vr,q))[il,jl]		# This could equally well be accomplished with a dot product
	return H