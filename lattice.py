#lattice.py
from __future__ import division,print_function
"""Functions for computing time evolution of wavefunctions in a moving optical lattice.
	Units are "natural", with 1=hbar=2m, for m=mass of particle, and electrical units 
	such that the dipole strength is 1, i.e. Rabi frequency = electric field strength.
	
	In general the lattice Hamiltonian can be expressed as:
	H = p^2/2m - (mgz) + |mu|^2/(2 hbar delta) * Sum_{a<b} (E_a E_b cos(k_{ab} x - delta_{ab} t + phi{ab})
	where E_a is the electric field strength, delta is the detuning (which is assumed to be 
	approximately the same for all lasers, so delta = w_a - w_0 for any laser a and w_0 = 
	transition frequency), phi is a phase, and subscript X_ab means X_a - X_b for any
	quantity X.  Note that this assumes all lasers are polarized in the same direction.
	To generalize to multiple polarizations, just replace |mu|^2 E_a E_b with 
	(mu.E_a)(mu.E_b) for real mu (for complex mu, there will also be a phase shift in the
	corresponding cosine).
	
	With the unit conventions adopted here, this Hamiltonian is reduced to:
	H = p^2 - (gz/2) + 1/(2 delta) * Sum_{a<b} (E_a E_b cos(k_{ab} x - delta_{ab} t + phi{ab})
	
	Of the parameters k_a, w_a, phi_a, and E_a, in principle all but k_a can be time dependent.
	k_a must remain (approximately) fixed, or else the lattice constants change, which skrews
	up the equations.  Note that k_a will change slightly if w_a changes, but we assume that 
	this can be neglected.  Similarly, delta is computed only once and remains fixed for the 
	remainder of the simulation (delta_{ab} does NOT remain fixed in this way).  In particular,
	delta is taken to be the value of delta_a at time t=0.  
	
	
"""

from numpy import *
from scipy import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import numbers

# MKS units:
hbar = 1.05457e-34
C = 299792458.0
eps0 = 8.85418782e-12
mu0 = 4.e-7*pi
eta0 = 377					# Impedance of free space
g = 9.81					# Earth's gravitational acceleration
M2 = 2*1.455e-25			# 2*Sr mass in kg
#d0 = 461.e-9				# A typical length scale in meters (Sr transition wavelength)
d00 = 461.e-9
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
# Sr 461 units:
wSr0 = wSr/w0				# Sr 461 frequency in 461 units						= 3.814 e11
gSr = g/a0					# Gravitational acceleration in 461 units			= 7.318
#---- Next are not "natural" but practical values
delta_typical = 5e12		# Typical (optimized) detuning for 461 nm transition (in Hz)
dtn0 = delta_typical/f0			# ... in 461 units = 2.93e9
Etyp = sqrt(2*dtn0*gSr)		# Corresponding typical electric field = 2.07e5 in 461 units
#Note: We want lattice amplitude to be >= gravitational potential accross d0
# => E^2/2dtn0 >= gSr (times d0 times M2, both 1)
# => E >= sqrt(2*dtn0*gSr) = 2.1e5 (= 2.1e5 E0 = 1444 in MKS)
# => power/area (magnitude of Poynting vector) > 1444^2/(2*eta0) = 2765 W/m^2 = .2765 W/cm^2
# This is a reasonable laser intensity

# Default laser parameters:
stheta = pi/8				# Default angle for the lasers
sk1 = array([cos(stheta),sin(stheta)])
sk2 = array([cos(stheta),-sin(stheta)])
sk3 = array([-1.0,0.0])
sE1 = Etyp; sE2 = Etyp; sE3 = Etyp;
sy1 = 0; sy2 = 0; sy3 = 0;	# Phase
sw1 = wSr0 + dtn0; sw2 = wSr0 + dtn0; sw3 = lambda t: wSr0 + dtn0 + 10.*t; 	# Angular frequency
# NOTE: 'Angular frequency' w is really (1/t) * int_0^t W(t')dt', where W(t') is the true instantaneous angular frequency

std =  {'k1':sk1,'w1':sw1,'y1':sy1,'E1':sE1,
		'k2':sk2,'w2':sw2,'y2':sy2,'E2':sE2,
		'k3':sk3,'w3':sw3,'y3':sy3,'E3':sE3,
		'grav':None,'wr':wSr0,'aa':1.0}

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
		dc['w2'],dc['y2'],dc['E2'], \
		dc['w3'],dc['y3'],dc['E3'] = map(funcify, [\
			dc['w1'],dc['y1'],dc['E1'], \
			dc['w2'],dc['y2'],dc['E2'], \
			dc['w3'],dc['y3'],dc['E3']	])
	
	return dc

std1 = stdize(std)

def shift(A,n,axis=0,method=0):
	'''Shifts array A by n indices along axis 'axis', zeroing indices that "fall off", using method 'method'.
		The default method=2 is about twice as fast as method=1 is about twice as fast
		as method=0.'''
	
	if method==0:
		B = zeros(A.shape,A.dtype)
		if n==0:
			B[:] = A[:]
		elif n>0:
			slcb = [slice(None)]*B.ndim
			slca = slcb.copy()
			slcb[axis] = slice(n,None)
			slca[axis] = slice(0,-n)
			B[slcb] = A[slca]
		else:
			slcb = [slice(None)]*B.ndim
			slca = slcb.copy()
			slcb[axis] = slice(0,n)
			slca[axis] = slice(-n,None)
			B[slcb] = A[slca]
		return B


############ momentum space ###########
def HParams(k1=None,w1=None,y1=None,E1=None,		# First laser's k-vector, frequency, phase, and amplitude
			k2=None,w2=None,y2=None,E2=None,		# Second laser
			k3=None,w3=None,y3=None,E3=None,		# Third laser
			grav=None,delta=None,wr=None,aa=1.0,	# Grav. accel., detuning, resonant ang. freq., lattice depth multiplier
			d=std):	
	''' Takes in laser parameters and spits out lattice Hamiltonian parameters
		(for momentum space Schrodinger equation).
		d is a dictionary for input parameters.  Values in d are overriden by
		parameters passed as arguments.
		
		Note that detuning delta and resonant angular frequency wr are redundant,
		as delta should equal w1-wr=w2-wr=w3-wr in principle.  delta overrides wr 
		in the case that both are supplied. 
	'''
	d = d.copy()
	inputs = {	'k1':k1,'w1':w1,'y1':y1,'E1':E1,
				'k2':k2,'w2':w2,'y2':y2,'E2':E2,
				'k3':k3,'w3':w3,'y3':y3,'E3':E3,
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
	
	k = array([d['k1']-d['k2'],d['k2']-d['k3'],d['k3']-d['k1']]).T		########## MIGHT TRANSPOSE THIS
	w = lambda t: array([d['w1'](t)-d['w2'](t),d['w2'](t)-d['w3'](t),d['w3'](t)-d['w1'](t)])
	y = lambda t: array([d['y1'](t)-d['y2'](t),d['y2'](t)-d['y3'](t),d['y3'](t)-d['y1'](t)])
	A = lambda t: d['aa']*array([d['E1'](t)*d['E2'](t),d['E2'](t)*d['E3'](t),d['E3'](t)*d['E1'](t)])/(2.*d['delta'])
	
	if grav is not None:
		p_g = lambda t: array([0,grav*t/2.])	# Gravitational shift to momentum
	else: 
		p_g = lambda t: array([0.0,0.0])
	
	# From the above parameters, the lattice Hamiltonian (with gravity unitaried away)
	# is given by H = (p - p_g)^2 + Sum_{j=1}^3 A[j]*cos(k[j]*x - w[j]*t + y[j])
	att = lambda t: {'k':k,'w':w(t),'y':y(t),'A':A(t),'grav':grav,'p_g':p_g(t)}
	ham = {'k':k,'w':w,'y':y,'A':A,'grav':grav,'p_g':p_g,'att':att}
	return ham


def avUniform(accel=0.,vel=0.,grav=gSr,gm=1.0,aa=1.0,
				Run=True,q=[0.,0.],T=arange(0,3,.01),n=5,init=0,ret='cxyh',plt='cp',talk=False):
	'''Sets up (and optionally runs) a system with uniform acceleration + 
		constant velocity shift.
		Signs for acceleration and velocity are the same as signs for x coordinates.
		gm stands for "g-multiplier" and simply multiplies grav by the constant 
			supplied.  This is so you can just take the default value of grav and
			scale it. 
		aa is an overall multiplier for the lattice depth
		run determines whether to run or just return Hamiltonian parameters.
		ret determines what to return: 'c' means coefficients, 'x' means 
			x momenta, 'y' means y momenta, 'h' means Hamiltonian parameters.
		plt determines whether to plot output: 'c' means coefficients, 'b'
			means bars, 'p' means momentum expectation value. 
		'''
	G = grav*gm
	w1 = lambda t: wSr0 + dtn0
	w2 = lambda t: wSr0 + dtn0
	w3 = lambda t: wSr0 + dtn0 + vel - accel*t
	h = HParams(w1=w1,w2=w2,w3=w3,grav=G,aa=aa,delta=dtn0)
	
	if Run:
		c,px,py = psolver(h,q,T,n=n,init=init,talk=talk)
	if plt:
		for i in plt:
			if i=='c':
				figure();plot(abs(reshape(c,(c.shape[0],c.shape[1]*c.shape[2]))))
			if i=='b':
				bars(c,-1)
			if i=='p':
				pexpect(c,px,py,out=False,plt=True)
	retrn = []
	for i in ret:
		if i=='c':
			retrn.append(c)
		if i=='x':
			retrn.append(px)
		if i=='y':
			retrn.append(py)
		if i=='h':
			retrn.append(h)
	return tuple(retrn)


def psolver(ham,q=[0.,0.],T=arange(0,2,.02),dt0=.01,n=5,init=None,aa=1.0,talk='some'):
	"""Solves p-space Schrodinger equation with parameters given by ham
	for initial data given by q and init, such that q is the momentum of a delta spike
	"""
	N=2*n+1									# Size of matrices
	c0 = zeros((len(T),N,N),dtype=complex)	# Matrix of coefficients
	
	k = ham['k']; p_g = ham['p_g']; A = ham['A']; y = ham['y']; w = ham['w'];
	
	if init is None:						# Initial data
		c0[0,n,n] = 1.0
	elif hasattr(init,'__len__'):
		c0[0,:,:] = init
	elif isinstance(init,int):
		tmp = eigs2(q,(k[:,0],k[:,1]),aa*A(0),init+1,y(0),n,returnM=True)
		c0[0,:,:] = wind(tmp[1][:,init])									# !!!NOTE: THIS INITIAL CONDITION MAY DIFFER BY A PHASE FROM ONE RUN TO ANOTHER!
	else:
		raise ValueError("init type not recognized.  If you want a band eigenstate, make sure that init is an int.")
	
	UP = eye(N,k=1); DN = eye(N,k=-1);
	m1,m2 = meshgrid(arange(-n,n+1),arange(-n,n+1))		# Temporary, for building momentum matrices
	Px = m2*k[0,0] + m1*k[0,1] + q[0]					# x momentum
	Py = m2*k[1,0] + m1*k[1,1] + q[1]					# y momentum
	# Note: The way momentum is organized is so that increasing the first index
	# by 1 adds k[0], and increasing the second index by 1 adds k[1]
	
	def D2(coef,t):		# Time derivative of coefficients
		At = A(t); 								# Minimize function calls
		ph = exp(-1.j*(w(t)*t - y(t)))			# phase
		return -1j* (((Px-p_g(t)[0])**2+(Py-p_g(t)[1])**2)*coef + \
			aa*At[0]/2. * ( ph[0] * UP.dot(coef) + conj(ph[0]) * DN.dot(coef) ) + \
			aa*At[1]/2. * ( ph[1] * coef.dot(DN) + conj(ph[1]) * coef.dot(UP) ) + \
			aa*At[2]/2. * ( ph[2] * DN.dot(coef.dot(UP)) + conj(ph[2]) * UP.dot(coef.dot(DN)) ))
	
	def D(coef,t):
		At = A(t); 								# Minimize function calls
		ph = exp(-1.j*(w(t)*t - y(t)))			# phase
		return -1j* (((Px-p_g(t)[0])**2+(Py-p_g(t)[1])**2)*coef + \
			aa*At[0]/2. * ( ph[0] * shift(coef,-1,0) + conj(ph[0]) * shift(coef,1,0) ) + \
			aa*At[1]/2. * ( ph[1] * shift(coef,-1,1) + conj(ph[1]) * shift(coef,1,1) ) + \
			aa*At[2]/2. * ( ph[2] * shift(shift(coef,1,1),1,0) + conj(ph[2]) * shift(shift(coef,-1,1),-1,0) ))
	
	#return c0[0,:,:],tmp[1][:,init],tmp[2],D
	
	tol = 1.e-6				# Absolute tolerance for time integration
	finer = 1.5				# Increase in resolution after each successive integration attempt
	for i in range(len(T)-1):
		dt = min(dt0,min(1./(abs(w(T[i]))+1.e-15)),1./amax(abs(D(c0[i,:,:],T[i]))))
		nsteps = int(ceil((T[i+1]-T[i])/dt))
		
		coef = midpoint(c0[i,:,:],D,T[i],T[i+1],nsteps)
		
		err = tol*2
		while (err>tol):
			coef0 = coef
			nsteps = int(ceil(nsteps*finer))
			coef = midpoint(c0[i,:,:],D,T[i],T[i+1],nsteps)
			err = amax(abs(coef-coef0))
			if talk=='all':
				print("Convergence: ",err,' vs. ',tol)
				if err>tol:
					print("Doing another iteration")
		
		if talk=='all':
			print("Time step ",i,": initial dt=",dt,", final error ",err,", nsteps=",nsteps,"\n")
		elif talk=='some':
			print("Completed time step ",i," of ",len(T))
		c0[i+1,:,:] = coef
	
	# The following shifts the momentum unitarily back to physical coordinates 
	pgx = rollaxis(array([p_g(t)[0] for t in T])[newaxis][newaxis],2)
	pgy = rollaxis(array([p_g(t)[1] for t in T])[newaxis][newaxis],2)
	return c0, Px[newaxis] - pgx, Py[newaxis] - pgy

##################### Time steppers ################
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

################### 2D band structure ############################
""" Taken from bands.py """

def dia2(q,bs,amps,n,ys=[0,0,0],M=True):
	'''Returns a k-space matrix Hamiltonian for an optical lattice.
		* q is a vector in the 1st Brillioun zone.
		* bs is a 2-tuple of k-space basis vectors
		* amps is a 3-iterable of amplitudes.  amps[0] and amps[1] 
			correspond to bs[0] and bs[1], and amps[2] corresponds to
			-bs[1]-bs[2]
		* y is a 3-iterable of phases.  The correspondence is as for amps.
		* n is the wavevector cutoff
		The lattice Schrodinger equation is given explicitly by:
			i hbar dc(p)/dt = P^2/2m c(p) + Sum_{j=0}^{3} amps[j]/2 * (exp(1j*ys[j]) c(p-bs[j]) + exp(-1j*ys[j]) c(p+bs[j]))
		By default (M=True), a matrix is returned.  The matrix operates
		on vectors c[i] defined so that c[i] is the coefficient of 
		exp[ ((-n+(i mod(2n+1)))*bs[0]+q).r) + (((-n+floor(i/(2n+1)))*bs[1]+q).r) ]]
		in a Fourier expansion of the wavefunction.
		
		[Formerly, I had written 
		(-n+(i mod(2n+1)))exp((bs[0]+q).r) + (-n+floor(i/(2n+1)))exp((bs[1]+q).r)
		 instead of the above expression, but I think this was incorrect.]
		'''
	
	N=2*n+1						# The Hamiltonian matrix will be of size N^2 by N^2
	idx = arange(N**2)			# Indices for the vectors
	# dia is the kinetic part of the Hamiltonian
	dia = (q[0] + bs[0][0]*(-n+mod(idx,N))+bs[1][0]*(-n+floor(idx/N)))**2 + \
			(q[1] + bs[0][1]*(-n+mod(idx,N))+bs[1][1]*(-n+floor(idx/N)))**2
	
	# The M's below capture the potential part of the Hamiltonian
	d1 = ones(N,float); d2 = ones(N-1,float);
	Mup = sparse.diags([d1*amps[1]/2.*exp(-1.j*ys[1]),d2*amps[2]/2.*exp(1.j*ys[2])],[0,1])
	Mdn = sparse.diags([d1*amps[1]/2.*exp(1.j*ys[1]),d2*amps[2]/2.*exp(-1.j*ys[2])],[0,-1])
	M0  = sparse.diags([d2*amps[0]/2.*exp(-1.j*ys[0]),d2*amps[0]/2.*exp(1.j*ys[0])],[1,-1])
	Mkin = sparse.diags(dia,0)
	
	# Each k below corresponds to an M above, and will be used in a Kronecker product below
	kup = sparse.diags(d2,1)
	kdn = sparse.diags(d2,-1)
	k0 	= sparse.diags(d1,0)
	
	M = Mkin + sparse.kron(k0,M0) + sparse.kron(kup,Mup) + sparse.kron(kdn,Mdn)
	
	return M

def eigs2(q,bs,amps,nbands,ys=[0,0,0],n=None,returnM=False,wind=False):
	'''Returns nbands number of eigenvectors/values for quasimomentum q.
		bs are reciprocal lattice basis vectors (there should be 2), 
		amps are amplitudes (there should be three), and n (if supplied)
		is the wavevector cutoff (so eigenvectors have length 2n+1).  
		If not supplied, n is taken to be nbands.
		q may not be iterable.
		If returnM is True, then the lattice Hamiltonian matrix is also returned. 
		If wind is True, then the eigenvectors are wound into square matrices.
			Otherwise, they are (unwound) column vectors. 
		'''
	if n is None:
		n = nbands
	M = dia2(q,bs,amps,n,ys)			# Get the Hamiltonian matrix
	# Amin is the bottom of the lattice potential, and a lower bound on the eigenenergies.  This is needed for eigsh
	Amin = -abs(amps[0])-abs(amps[1])-abs(amps[2])
	eigvals,eigvecs = eigsh(M,nbands,sigma=Amin)
	s = argsort(eigvals)
	eigvals = (eigvals[s])[:nbands]
	eigvecs = (eigvecs[:,s])[:,:nbands]
	
	if wind:
		eigmats = zeros((nbands,2*n+1,2*n+1),dtype=complex)
		for i in range(nbands):
			eigmats[i] = reshape(eigvecs[:,i],(2*n+1,2*n+1),'F')
		eigvecs = eigmats
	if returnM:
		return eigvals,eigvecs,M
	else:
		return eigvals,eigvecs

def wind(c):
	'''Takes a vector of 2D momentum coefficients and turns them into 
		a matrix, or vice versa.'''
	
	if len(c.shape) == 1:		# This means we have a vector
		n = sqrt(len(c))
		return reshape(c,(n,n),'F')
	if len(c.shape) == 2:
		return ravel(c,'F')

################### Visualizers ##################

def getLat(ham,t,xs,ys):
	"""Returns lattice values at time t and points (x,y).  x and y can be arrays."""
	A = ham['A'](t); k = ham['k']; y = ham['y'](t); w = ham['w'](t);
	out = zeros(xs.shape)
	for i in (0,1,2):
		out += A[i]*cos(k[0,i]*xs + k[1,i]*ys - w[i]*t + y[i])
	return out

def plotLat(ham,t,xs=None,ys=None,N=2,fig=None):
	"""Plots lattice using getLat.
		xs is an array of points to plot.  Alternatively,
		N is a number of periods to plot (centered at zero).
		xs overrides N."""
	k = ham['k']
	K = min(norm(k[:,0]),norm(k[:,1]))
	if xs is None:
		xs = N*pi*(1./K)*linspace(-1,1+1./(50*N),N*50)
	if ys is None:
		ys = N*pi*(1./K)*linspace(-1,1+1./(50*N),N*50)
	x,y = meshgrid(xs,ys)
	z = getLat(ham,t,x,y)
	f = figure(fig);
	ax = f.add_subplot(111, projection='3d')
	ax.plot_surface(x,y,z)

def bars(coef,t=None,vert=True,fig=None,pv=None,ph=None,fn=abs):
	""" Plots 3d bar graph of coefficients in coef at time index t.
		If t is not supplied, coef is assumed to be 2D.  If t is 
		supplied, coef is assumed to be 3D.
		vert==True means plot coef[t,:,i] for each i.
		vert==False means plot coef[t,i,:] for each i.
		pv is the momentum value along the vertical axis
		ph is the momentum value along the horizontal axis
		fn is a function applied to the coef array before displaying.
		"""
	if t is not None:
		if not coef.ndim==3:
			raise ValueError("coef must have dimension 3 if t is provided.")
		c = fn(coef[t,:,:])
	else:
		if not coef.ndim==2:
			raise ValueError("coef must have dimension 2 if a time is not supplied.")
		c = fn(coef)
	
	if not vert:
		c = c.T
	if pv is None:		# If pv and ph are not supplied, we use integers centered at 0
		pv = arange(c.shape[0])-(c.shape[0]-1)/2
	if ph is None:
		ph = arange(c.shape[1])-(c.shape[1]-1)/2
	
	f = figure(fig)
	ax = f.add_subplot(111,projection='3d')
	for i in range(c.shape[1]):
		ax.bar(pv,c[:,i],ph[i],zdir='y',alpha=.8)
	ax.set_xlabel('Vertical k-index')
	ax.set_ylabel('Horizontal k-index')
	ax.set_zlabel('Coefficients')


############## Metrics and postprocessers #################

def pexpect(c,px,py,plt=False,out=True,fig=None):
	""" Gets expectation value of momentum from coefficient array c.
		c may be 2 or 3 dimensional; if 3 dimensional, first dimension
		is interpreted as time.  All of c, px, & py must have same shape.
	"""
	if not c.shape==px.shape==py.shape:
		raise ValueError('c, px, and py must have same shape.')
	
	c2 = abs(c)**2			# Probability array
	def expct(C2,PX,PY):
		""" Expectation for 2d arrays. """
		return array([sum(C2*PX),sum(C2*PY)])
	
	if c.ndim==2:
		return expct(c2,px,py)
	elif c.ndim==3:
		nt = c.shape[0]		# Number of times
		p = zeros((nt,2),dtype=float)
		for i in range(nt):
			p[i,:] = expct(c2[i],px[i],py[i])
		if plt:
			figure(fig)
			plot(p[:,0],label='Px')
			plot(p[:,1],label='Py')
			legend(loc='upper left')
		if out:
			return p
	else:
		raise ValueError('Inputs must have 2 or 3 dimensions.')

def dual(k,norm=2*pi):
	'''Constructs dual vectors to those in k[:,0:2].
		norm is the normalization, so 
		k[:,i].dot(kDual[:,j]) = norm*delta_{ij}'''
	a0 = array([-k[1,1],k[0,1]])
	a1 = array([-k[1,0],k[0,0]])
	a0 *= norm/a0.dot(k[:,0])
	a1 *= norm/a1.dot(k[:,1])
	return array([a0,a1]).T

def Dt(f,t0,dt=1.e-6):
	'''Time derivative of f(t) at t0.'''
	return (f(t0+dt) - f(t0-dt))/(2.*dt)


def LFrame(c,px,py,ham,t,T=None,mode=2):
	if T is not None:
		'''Providing T means t is an index, and T[t] is the actual time.'''
		c = c[t];px=px[t];py=py[t];
		t = T[t]
	if mode==1:
		''' Shifts c to lattice frame at time t.
		'''
		k = ham['k']; w = ham['w']; dwdt = Dt(w,t);
		kDual = dual(k)
		xtrans = (kDual[:,0]*w(t)[0] + k[:,1]*w(t)[1] )/(2*pi)
		c2 = c * exp(1j * (px*xtrans[0] + py*xtrans[1])*t)
		ktrans = .5*xtrans + .5*(kDual[:,0]*dwdt[0] + k[:,1]*dwdt[1])*t/(2*pi)
		px2 = px-ktrans[0]
		py2 = py-ktrans[1]
		return c2,px2,py2
	elif mode==2:
		''' Shifts c to lattice frame at time t via Galilean boost.
		'''
		k = ham['k']; w = ham['w']; dwdt = Dt(w,t);
		kDual = dual(k)
		v = kDual.dot((w(t) + dwdt*t)[:2])/2./pi; v2 = v.dot(v)	# Velocity of Galilean boost
		c2 = exp(1j*v2*t/4.)*exp(1j*(px*v[0]+py*v[1])*t)*c		# Translation part of boost
		px2 = px - .5*v[0]; py2 = py - .5*v[1]					# Momentum shift part of boost
		return c2,px2,py2

def qProj(c,px,py,ham,t,band=0,T=None):
	if T is not None:		# This indicates that t is an index, and the true time is T[t]
		c = c[t]; px = px[t]; py = py[t]; t = T[t]
	k = ham['k']; A = ham['A']; y = ham['y']; w = ham['w']; dwdt = Dt(w,t);
	kDual = dual(k)
	v = kDual.dot((w(t) + dwdt*t)[:2])/2./pi; v2 = v.dot(v)	# Velocity of Galilean boost
	c2 = exp(1j*v2*t/4.)*exp(1j*(px*v[0]+py*v[1])*t)*c		# Translation part of boost
	px2 = px - .5*v[0]; py2 = py - .5*v[1]					# Momentum shift part of boost
	
	N = c2.shape[0]; n=(N-1)/2
	q = array([px2[n,n],py2[n,n]])
	ph = y(t) - w(t)*t + k.T.dot(v)*t		# Phase of lattice in rest frame
	if not hasattr(band,'__len__'):
		band = array([band])
	evec = eigs2(q,k.T,A(t),amax(band)+1,ph,n=n,wind=True)[1][band]
	out = []
	for i in range(len(band)):
		out.append(abs(sum(evec[i].conj() * c2))**2)
	return array(out),ph,q,v,dwdt,w(t)

def SFrame(c,px,py,ham,t,T=None,band=None):
	if T is not None:
		c = c[t]; px = px[t]; py = py[t]; t = T[t];
	k = ham['k']; A = ham['A']; w = ham['w']; y = ham['y'];
	kDual = dual(k,1.)								# Note normalization is 1, not 2*pi
	B = lambda t: kDual.dot((w(t)*t-y(t))[:2])		# Need B as lambda to get dBdt
	dBdt = Dt(B,t)
	B = B(t)			# This is the only value of B we'll need now
	c2 = exp(1.j*(px*B[0]+py*B[1]))*c
	px2 = px - .5*dBdt[0]
	py2 = py - .5*dBdt[1]
	if band is None:
		return c2,px2,py2
	else:
		if not hasattr(band,'__len__'):
			band = array([band])
		N = c2.shape[0]; n = (N-1)/2
		q = [px2[n,n],py2[n,n]]
		evec = eigs2(q,k.T,A(t),amax(band)+1,n=n,wind=True)[1]
		out = []
		for i in band:
			out.append( abs(sum( evec[i].conj()*c2 ))**2 )
		out = array(out)
		return out

def SProj(c,px,py,ham,T,idx=slice(None),band=0,talk=False):
	if not c.shape==px.shape==py.shape and c.shape[0]==len(T):
		raise ValueError('c, px, py, and T do not have consistent shapes.')
	L = c.shape[0]
	if not hasattr(band,'__len__'):
		band = array([band])
	nb = band.shape[0]
	idx = range(L)[idx]
	L = len(idx)
	out = zeros((L,nb))
	for i in range(L):
		j = idx[i]
		if talk:
			print('step {} of {}'.format(i,L))
		out[i] = SFrame(c[j],px[j],py[j],ham,T[j],band=band)
	return out
	
"""
def qProj(c,px,py,ham,t=None,band=0,q=None,show=False):
	'''Projects onto lattice eigenstates for quasimomentum q.
		If q is not supplied, it is inferred from px and py.
		If t is supplied, it is assumed that the input values
		of c, px, and py are for the lab frame, and they are
		transformed automatically into the lattice frame. 
	'''
	k = ham['k']
	if t is not None:
		c,px,py = LFrame(c,px,py,t,ham)
	N = c.shape[0]
	n = (N-1)/2
	if q is None:
		q = array([px[n,n],py[n,n]])
	evec = eigs2(q,(k[:,0],k[:,1]),ham['A'](t),band+1,ys=ham['y'](t),n=n,wind=True)[1][band]
	if show:
		bars(evec)
		bars(c)
	return sum(evec.conj() * c)
=======
"""