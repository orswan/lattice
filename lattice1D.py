#lattice1D.py
from __future__ import division,print_function
"""Functions for computing time evolution of wavefunctions in a moving 1D optical lattice.
	Units are "natural", with 1=hbar=2m, for m=mass of particle, and electrical units 
	such that the dipole strength is 1, i.e. Rabi frequency = electric field strength.
"""

from numpy import *
from scipy import *
from pylab import *
from scipy import sparse
from scipy.linalg import expm
import numpy
import numbers

# Units:
hbar = 1.05457e-34
C = 299792458.0
eps0 = 8.85418782e-12
mu0 = 4.e-7*pi
eta0 = 377					# Impedance of free space
g = 9.81					# Earth's gravitational acceleration
M2 = 2*1.455e-25			# 2*Sr mass in kg
#d0 = 461.e-9
lam0 = 461.e-9				# A typical length scale in meters (Sr transition wavelength)
d0 = lam0/(2*pi)			# NOTE: THIS IS THE RIGHT CHARACTERISTIC LENGTH FOR LATTICE RECOIL UNITS!
k0 = 1./d0					# Sr transition k-vector length
mu = 2.58e-29				# Sr 461 dipole strength in meter second Amperes
fSr = C/d0					# Sr 461 transition frequency in Hz					= 6.503 e14
wSr = fSr*2*pi				# Sr 461 transition angular frequency				= 4.086 e15
f0 = hbar/(M2*d0**2)		# Sr 461 characteristic frequency hbar/(2*m*d0^2)	= 1.705 e3
w0 = f0*2*pi				# Sr 461 characteristic angular frequency			= 1.071 e4
U0 = hbar*f0				# Sr 461 characteristic energy (recoil energy/(2pi)^2)	= 1.798 e-31
E0 = U0/mu					# Sr 461 characteristic electric field				= 6.970 e-3
a0 = d0*f0**2				# Sr 461 characteristic acceleration				= 1.340
wSr0 = wSr/w0				# Sr 461 frequency in 461 units						= 3.814 e11
gSr = g/a0					# Gravitational acceleration in Sr units (~8.6)		= 7.318
EgSr = g*d0*M2				# Gravitational characteristic energy for Sr		= 1.316 e-30 = gSr*U0
#---- Next are not "natural" but practical values
delta_typical = 5e12		# Typical (optimized) detuning for 461 nm transition
dtn0 = delta_typical/f0			# ... in 461 units = 2.93e9
Etyp = sqrt(2*dtn0*gSr)		# Corresponding typical electric field = 2.07e5 in 461 units
#Note: We want lattice amplitude to be >= gravitational potential over d0
# => E^2/2dtn0 >= gSr (times d0 times M2, both 1)
# => E >= sqrt(2*dtn0*gSr) = 2.1e5 (= 2.1e5 E0 = 1444 in MKS)
# => power/area (magnitude of Poynting vector) > 1444^2/(2*eta0) = 2765 W/m^2 = .2765 W/cm^2
# This is a reasonable laser intensity

'''Had this written before: I think it's wrong:
	#Note: We want electric field amplitude to be >= order of sqrt(gSr)~3
	# => electric field > 2e-2 
	# => power/area (magnitude of Poynting vector) > 2e-2/(2*eta0) = 3e-5 W/m^2
	# This is way smaller than real laser intensities (~1W/cm^2=1e4 W/m^2), so
	# typical electric field strengths in the range '''

# Default laser parameters:
sk1=1.0; sk2 = -1.0
sE1 = Etyp; sE2 = Etyp;			####### NEED TO CHECK AGAINST GRAVITY IN Sr UNITS
sy1 = 0; sy2 = 0;			# Phase
sw1 = wSr0 + dtn0; sw2 = lambda t: wSr0 + dtn0 + 10*t; 		# Angular frequency
# NOTE: 'Angular frequency' w is really (1/t) * int_0^t W(t')dt', where W(t') is the true instantaneous angular frequency

std =  {'k1':sk1,'w1':sw1,'y1':sy1,'E1':sE1,
		'k2':sk2,'w2':sw2,'y2':sy2,'E2':sE2,
		'grav':None,'wr':wSr0}

def funcify(a):
	''' Makes argument into a function of time .
		If argument is already a function of time, return it.
	'''
	if callable(a):
		return a
	else:
		return lambda t: a

def stdize(d,func="wyE"):
	'''Takes a dictionary of input parameters and turns 
		the appropriate parameters into functions (time).
		Also subs default values for those not provided.
		'func' indicates which parameters to make into 
		functions.
	'''
	dc = std.copy()
	dc.update(d)
	if func=="wyE":
		dc['w1'],dc['y1'],dc['E1'], \
		dc['w2'],dc['y2'],dc['E2'] = map(funcify, [\
			dc['w1'],dc['y1'],dc['E1'], \
			dc['w2'],dc['y2'],dc['E2']	])
	
	return dc

std1 = stdize(std)

############ momentum space ###########
def HParams(k1=None,w1=None,y1=None,E1=None,		# First laser's k-vector, frequency, phase, and amplitude
			k2=None,w2=None,y2=None,E2=None,		# Second laser
			grav=None,delta=None,wr=None,aa=1.0,d=std):
	''' Takes in laser parameters and spits out lattice Hamiltonian parameters
		(for momentum space Schrodinger equation).
		d is a dictionary for input parameters.  Values in d are overriden by
		parameters passed as arguments.
	'''
	d = d.copy()
	inputs = {	'k1':k1,'w1':w1,'y1':y1,'E1':E1,
				'k2':k2,'w2':w2,'y2':y2,'E2':E2,
				'grav':grav,'delta':delta,'wr':wr,'aa':aa}
	keys = inputs.keys()
	for i in keys:
		if inputs[i] is not None:	# Throw out null inputs
			d[i] = inputs[i]		# Wrap everything into d
	
	d = stdize(d)
	
	# Get the detuning figured out:
	if ('delta' not in d.keys()) or d['delta'] is None:
		if ('wr' not in d.keys()) or d['wr'] is None:
			raise ValueError('Either delta or wr must be supplied.')
		d['delta'] = d['w1'](0)-d['wr']	
	if d['delta']==0:
		raise ValueError("Detuning can't be zero.")
	
	k = d['k1']-d['k2']		########## MIGHT TRANSPOSE THIS
	w = lambda t: d['w1'](t)-d['w2'](t)
	y = lambda t: d['y1'](t)-d['y2'](t)
	A = lambda t: d['aa']*d['E1'](t)*d['E2'](t)/(2.*d['delta'])
	
	if grav is not None:
		p_g = lambda t: grav*t/2.	# Gravitational shift to momentum
	else: 
		p_g = lambda t: 0.0
	
	# From the above parameters, the lattice Hamiltonian (with gravity unitaried away)
	# is given by H = (p - p_g)^2 + A[j]*cos(k[j]*x - w[j]*t + y[j])
	
	ham = {'k':k,'w':w,'y':y,'A':A,'grav':grav,'p_g':p_g}
	return ham


def avUniform(accel=0.,vel=0.,grav=gSr,gm=1.0,aa=1.0,
				Run=True,q=0.,T=arange(0,3,.01),n=5,init=0,ret='cph',plt='c',talk=False):
	'''Sets up (and optionally runs) a system with uniform acceleration + 
		constant velocity shift.
		Signs for acceleration and velocity are the same as signs for x coordinates.
		gm stands for "g-multiplier" and simply multiplies grav by the constant 
			supplied.  This is so you can just take the default value of grav and
			scale it. 
		aa is an overall multiplier for the lattice depth
		run determines whether to run or just return Hamiltonian parameters.
		ret determines what to return: 'c' means coefficients, 'p' means 
			momenta, 'h' means Hamiltonian parameters.
		plt determines whether to plot output: 'c' means coefficients, 'b'
			means bars, 'p' means momentum expectation value. 
		'''
	g = grav*gm
	w1 = lambda t: wSr0 + dtn0 + vel + accel*t
	w2 = lambda t: wSr0 + dtn0
	h = HParams(w1=w1,w2=w2,grav=g,aa=aa,delta=dtn0)
	
	if Run:
		c,p = psolver(h,q,T,n=n,init=init,talk=talk)
	if plt:
		for i in plt:
			if i=='c':
				figure();plot(abs(c))
			if i=='b':
				bars(c,p,-1)
			if i=='p':
				pexpect(c,p,out=False,plt=True)
	retrn = []
	for i in ret:
		if i=='c':
			retrn.append(c)
		if i=='p':
			retrn.append(p)
		if i=='h':
			retrn.append(h)
	return tuple(retrn)


def psolver(ham,q=0.,T=arange(0,2,.02),dt0=.01,n=5,aa=1,init=0,talk='some',plt=False):
	"""Solves p-space Schrodinger equation with parameters given by ham
		for initial data given by 1 and init, such that the initial wavefunction
		is given by:
		|Y> = Sum_{j=0}^{2n+1} init[j] |q+(j-n)k>
		for k = lattice wavenumber. 
		
		init need not be supplied, and in this case the initial wavefunction is
		|Y> = |q>
		init may also be an integer, in which case it is interpreted to mean a
		lattice eigenvector with quasimomentum q and band index init >= 0. 
		
		input parameter aa is an overall scaling of lattice depth, for convenience.
		If plt is None or a number, abs(coefficients) is plotted on figure(plt)
	"""
	N=2*n+1									# Size of matrices
	c0 = zeros((len(T),N),dtype=complex)	# Matrix of coefficients
	
	k = ham['k']; p_g = ham['p_g']; A = ham['A']; y = ham['y']; w = ham['w'];
	
	if init is None:
		c0[0,n] = 1.0							# Initial data
	elif hasattr(init,'__len__'):
		c0[0,:] = init
	elif isinstance(init,int):
		tmp = eigs1(q,k,aa*A(0),init+1,n)
		c0[0,:] = tmp[1][:,init]
	else:
		raise ValueError("init type not recognized.  If you want a band eigenstate, make sure that init is an int.")
	
	P = (q + arange(-n,n+1)*k)			# Momentum
	UP = eye(N,k=1); DN = eye(N,k=-1);
	# Note: The way momentum is organized is so that increasing the index by 1 adds k
	
	def D(coef,t):		# Time derivative of coefficients
		ph = exp(-1.j*(w(t)*t - y(t)))			# phase
		return -1.j * ((P-p_g(t))**2*coef + aa*A(t)/2. * ((1./ph)*DN.dot(coef) + ph*UP.dot(coef)))
	
	tol = 1.e-6				# Absolute tolerance for time integration
	finer = 1.5				# Increase in resolution after each successive integration attempt
	for i in range(len(T)-1):
		dt = min(dt0,1./(abs(w(T[i]))+1.e-15),1./amax(abs(D(c0[i,:],T[i]))))
		nsteps = int(ceil((T[i+1]-T[i])/dt))
		
		coef = midpoint(c0[i,:],D,T[i],T[i+1],nsteps)
		
		err = tol*2
		while (err>tol):
			coef0 = coef
			nsteps = int(ceil(nsteps*finer))
			coef = midpoint(c0[i,:],D,T[i],T[i+1],nsteps)
			err = amax(abs(coef-coef0))
			if talk=='all':
				print("Convergence: ",err,' vs. ',tol)
				if err>tol:
					print("Doing another iteration")
		
		if talk=='all':
			print("Time step ",i,": initial dt=",dt,", final error ",err,", nsteps=",nsteps,"\n")
		elif talk=='some':
			print("Completed time step ",i," of ",len(T))
		c0[i+1,:] = coef
	
	if plt is not False:
		figure(plt)
		plot(abs(c0))	
	
	return c0, P-array([[p_g(t) for t in T]]).T


###################### Time steppers ##############################
def Euler(coef0,D,t0,t1,nsteps):
	"""Integrate coef from time t0 to t1 in nsteps Euler steps
		D is a function of coef and time t, returning the derivative
		of coef at that time."""
	coef = coef0
	nsteps = int(nsteps)
	if nsteps <= 0:
		raise ValueError("Number of steps for stepper must be positive.")
	dt = (t1-t0)/nsteps
	for i in range(nsteps):
		t = t0*(nsteps-i)/nsteps + t1*i/nsteps
		coef += dt*D(coef,t)
	return coef

def midpoint(coef0,D,t0,t1,nsteps):
	"""Integrate coef from time t0 to t1 in nsteps midpoint steps
		D is a function of coef and time t, returning the derivative
		of coef at that time."""
	coef = copy(coef0)			# Copy initial data so changes don't propagate backwards
	nsteps = int(nsteps)
	if nsteps <= 0:
		raise ValueError("Number of steps for stepper must be positive.")
	dt = (t1-t0)/nsteps
	for i in range(nsteps):
		t = t0*(nsteps-i)/nsteps + t1*i/nsteps
		coef += dt*D(coef+dt*D(coef,t)/2.,t+dt/2.)
	return coef


########################## 1D Band structure #####################
""" These 1D band structure functions are borrowed from bands.py"""

def tridiag1(q,b,amp,n,M=False):
	'''Returns a tridiagonal (2n+1)x(2n+1) matrix representing the 
		1D optical lattice Schrodinger equation for quasimomentum q.
		* q is quasimomentum
		* b is the reciprocal lattice basis vector
		* amp is the amplitude of the lattice
		* n determines the size of the matrix (i.e. momentum cutoff)
		'''
	dia = (q+b*arange(-n,n+1))**2
	up  = amp/2. * ones(2*n,float)
	dn  = amp/2. * ones(2*n,float)
	
	if M:
		return diag(up,1)+diag(dia)+diag(dn,-1)
	else:
		return up,dia,dn


def eigs1(q,b,amp,nbands,n=False,returnM=False):
	'''Returns nbands number of eigenenergies and eigenvectors
		for a single quasimomentum q.
		b is the reciprocal lattice basis vector, amp is lattice
		amplitude, n is momentum cutoff (see bands.py).
	'''
	if not n:
		n = nbands
	M = tridiag1(q,b,amp,n,True)
	enrg0,evec0 = linalg.eigh(M)
	enrg = enrg0[:nbands]
	evec = evec0[:,:nbands]
	if not returnM:
		return enrg,evec
	else:
		return enrg,evec,M

################# Visualizers #################


def getLat(ham,t,xs,aa=1):
	"""Returns lattice values at time t and points xs (which can be an array)."""
	A = ham['A'](t); k = ham['k']; y = ham['y'](t); w = ham['w'](t); g = ham['grav']
	if not isinstance(g,numbers.Number):
		g = 0
	return aa*A * cos(k*xs - w*t + y) + g*xs/2

def plotLat(ham,t,xs=None,N=2,fig=None,aa=1):
	"""Plots lattice using getLat.
		xs is an array of points to plot.  Alternatively,
		N is a number of periods to plot (centered at zero).
		xs overrides N."""
	figure(fig)
	if xs is not None:
		plot(xs,getLat(ham,t,xs,aa))
	else:
		k = ham['k']
		dx = 2.*pi/k/50.
		x = arange(-N*pi/k,N*pi/k+dx,dx)
		plot(x,getLat(ham,t,x,aa))

def bars(c,p,t,fn=abs,fig=None):
	'''Makes bar plot of the coefficients c at time step t 
		(as function of momentum p).
	'''
	figure(fig)
	poffset = (p[0,1]-p[0,0])/4.	# This shift is applied to center the bars on the momentum
	bar(p[t,:]-poffset,fn(c[t,:]))

def pexpect(c,p,t=None,plt=False,out=True,fig=None):
	'''Returns expectation value of momentum as a function of time
		(unless t is provided, in which case the expectation of p at just that
		time step is returned).
		When t is not provided, you can optionally plot the result in figure
		fig, and also suppress the output, using plt and out keywords, resp.
		'''
	if t is not None:
		return abs(c**2)[t,:].dot(p[t,:])
	elif not plt:
		return sum(abs(c**2)*p,1)
	else:
		pex = sum(abs(c**2)*p,1)
		figure(fig)
		plot(pex)
		if out:
			return pex

def checkUnitarity(c,plt=False,out='std',fig=None):
	'''Checks unitarity by computing variation of the l2 norm of c vs. time.'''
	l2s = sum(abs(c)**2,1)
	stdev = numpy.std(l2s)/mean(l2s)
	if plt:
		figure(fig)
		plot(l2s)
	if out=='std':
		return stdev
	elif out=='l2s':
		return l2s
		
"""
def LFrame(c,p,ham,q,t,a,adot,nband):
	p_g = ham['p_g'](t)
	k = ham['k']; amps = ham['A'](t)
	p1 = p + p_g
	nq = eigs1(q-p_g-adot/2.,k,amps,nband+1,n=len(c)//2)[1][:,-1]
	return nq.dot(exp(1j*p1*a)*c)
"""

def Dt(f,t0,dt=1.e-6):
	'''Time derivative of f(t) at t0.'''
	return (f(t0+dt) - f(t0-dt))/(2.*dt)

def LFrame(c,p,ham,t,T=None,band=None):
	if T is not None:		# This indicates t is an index, and T[t] is the corresponding time
		c = c[t]; p = p[t]; t = T[t]
	k = ham['k']; A = ham['A']; y = ham['y']; w = ham['w']; dwdt = Dt(w,t)
	v = (w(t) + dwdt*t)/k
	c2 = exp(1j*v**2*t/4.)*exp(1j*p*v*t) * c
	p2 = p - v/2.
	if band is None:
		return c2,p2
	else:				# If band is supplied, we dot the state into lattice frame eigenstates
		if not hasattr(band,'__len__'):
			band = array([band])
		N = c2.shape[0]; n = (N-1)/2
		q = p[n]
		ph = y(t) + k*v*t - w(t)*t
		evecs = eigs1(q,k,A(t),amax(band)+1,n)[1]
		out = []
		for i in range(len(band)):
			out.append(abs(sum(evecs[:,i].dot(c2)))**2)
		return array(out)

def SFrame(c,p,ham,t,T=None,band=None):
	if T is not None:
		c = c[t]; p = p[t]; t = T[t]
	k = ham['k']; A = ham['A']; y = ham['y']; w = ham['w'];
	B = lambda tau: (w(tau)*tau - y(tau))/k
	dBdt = Dt(B,t)
	c2 = exp(1j*B(t)*p)*c		# This ignores an overall phase exp(-1j*int_0^t (.5*m*dbdt^2))
	p2 = p-.5*dBdt
	if band is None:
		return c2,p2
	else:
		if not hasattr(band,'__len__'):
			band = array([band])
		N = c2.shape[0]; n = (N-1)/2
		q = p[n]
		evecs = eigs1(q,k,A(t),amax(band)+1,n)[1]
		out = []
		for i in range(len(band)):
			out.append(abs(sum(evecs[:,i].dot(c2)))**2)
		return array(out)

def SProj(c,p,ham,T,idx=slice(None),band=0,talk=False):
	if not c.shape==p.shape and c.shape[0]==len(T):
		raise ValueError('c, p, and T do not have consistent shapes.')
	L = c.shape[0]
	if not hasattr(band,'__len__'):
		band = array([band])
	nb = band.shape[0]
	idx = range(L)[idx]
	L = len(idx)
	out = zeros((L,nb))
	for j in range(L):
		if talk:
			print('step {} of {}'.format(j,L))
		i = idx[j]
		out[j] = SFrame(c[i],p[i],ham,T[i],band=band)
	return out

"""
def Eproj(c,p,t,ham,T=None,band=5):
	'''Project onto lattice eigenstates.'''
	if not hasattr(band,'__len__'):
		band = array([band])
	N = c.shape[0]; n = (N-1)/2;
	e,v = eigs1
