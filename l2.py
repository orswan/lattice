#lattice.py
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
from scipy.integrate import quad

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

	sparams =  {'ks':sks,'Es':sEs,'grav':gSr,'wr':wSr0,'dtn':dtn0,'nbands':10}

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

def shift(A,n,axis=0):
	'''Shifts array A by n indices along axis 'axis', zeroing indices that "fall off".
		'''
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

def multirange(t):
	"""generates range of indices in multiple dimensions. t should be iterable."""
	return itertools.product(*[range(i) for i in t])

def ptPlot(pt,*args):
	"""Plots points pt"""
	plot(pt[0],pt[1],*args)

def restHam(**kwargs):
	""""""
	params = std1.copy()
	for i in kwargs.keys():
		params[i] = kwargs[i]

def BZCorner(v1,v2,norm=.5):
	""" Finds the vector v s.t. v.dot(v1)=norm*v1^2 and v.dot(v2) = norm*v2^2
		This is a way to find corners of a Brillouin zone."""
	w1 = array([-v1[1],v1[0]])		# Dual vectors, unnormalized
	w2 = array([-v2[1],v2[0]])
	return 	norm*(v1[0]**2 + v1[1]**2)/(w2[0]*v1[0]+w2[1]*v1[1]) * w2 +\
			norm*(v2[0]**2 + v2[1]**2)/(w1[0]*v2[0]+w1[1]*v2[1]) * w1

def dual(k,norm=2*pi):
	'''Constructs dual vectors to those in k[:,0:2].
		norm is the normalization, so 
		k[:,i].dot(kDual[:,j]) = norm*delta_{ij}'''
	a0 = array([-k[1,1],k[0,1]])
	a1 = array([-k[1,0],k[0,0]])
	a0 *= norm/a0.dot(k[:,0])
	a1 *= norm/a1.dot(k[:,1])
	return array([a0,a1]).T

def ptRange(pts,n,closed=True):
	"""Marks points along a polygonal path defined by pts.
		n points per segment.
		"""
	P = pts.shape[1]
	if closed:
		N = n*P
	else:
		N = n*(P-1)+1
	xs = zeros(N)			# x coordinates
	ys = zeros(N)			# y coordinates
	r = arange(n)/n
	for i in range(P-1):
		xs[i*n:(i+1)*n] = pts[0,i] + r*(pts[0,i+1]-pts[0,i])
		ys[i*n:(i+1)*n] = pts[1,i] + r*(pts[1,i+1]-pts[1,i])
	if closed:
		xs[n*(P-1):] = pts[0,-1] + r*(pts[0,0]-pts[0,-1])
		ys[n*(P-1):] = pts[1,-1] + r*(pts[1,0]-pts[1,-1])
	else:
		xs[-1] = pts[0,-1]
		ys[-1] = pts[1,-1]
	return array([xs,ys])

class Hamiltonian:
	def __init__(self,**kwargs):
		params = deepcopy(sparams)
		for i in kwargs.keys():				# Update standard parameters with user-supplied parameters
			if i in params.keys(): params[i] = kwargs[i]
		# The following are convenience parameters
		if 'phi' in kwargs.keys():		# Incoming laser angles
			phi = kwargs['phi']
			params['ks'] = array([[cos(phi),sin(phi)],[cos(phi),-sin(phi)],[-1.,0.]]).T
		if 'gm' in kwargs.keys():		# Scale gravitational strength by gm
			params['grav'] *= kwargs['gm']
		if 'amps' in kwargs.keys():		# Specify amplitudes directly
			A = kwargs['amps']
			params['Es'] = array([A[0]*A[2]/A[1],A[1]*A[0]/A[2],A[2]*A[1]/A[0]])
		if 'am' in kwargs.keys():		# Scale lattice depth by am
			params['Es'] *= sqrt(kwargs['am'])
		if 'Eratio' in kwargs.keys():	# Scale laser 3 strength by 1/Eratio
			params['Es'][2] /= kwargs['Eratio']
		
		# Reciprocal basis vectors
		ks = params['ks'].T			# The .T is just for convenience on the next line
		self.recip = array([ks[1]-ks[0],ks[2]-ks[1],ks[0]-ks[2]]).T
		self.b = self.recip			# Short alias
		self.ks = ks.T
		# Direct lattice basis vectors
		self.r = dual(self.b)
		# Cosine amplitudes
		Es = params['Es']
		self.amps = array([Es[1]*Es[0],Es[2]*Es[1],Es[0]*Es[2]])/(2*params['dtn'])
		self.A = self.amps			# Short alias
		self.Es = Es
		# Gravity
		self.grav = params['grav']
		# Brillouin zone properties
		self.dirs = empty((2,6))		# Brillouin zone directions - reciprocal vectors and their negatives
		self.dirs[:,:3] = self.b; self.dirs[:,3:] = -self.b
		self.args = mod(angle(self.dirs[0] + 1j*self.dirs[1]),2*pi)	# Angles of each direction
		self.ordr = argsort(self.args)
		self.bzCorners = BZCorner(self.dirs[:,self.ordr],self.dirs[:,roll(self.ordr,-1)])
	
	def getLat(self,xs,ys,grav=False):	
		"""Returns values of lattice potential at positions w/ x-coords xs and y-coords ys."""
		A = self.A; b = self.b
		V = A[0]*cos(b[0,0]*xs + b[1,0]*ys) + A[1]*cos(b[0,1]*xs + b[1,1]*ys) + A[2]*cos(b[0,2]*xs + b[1,2]*ys)
		if grav:
			return V + .5*self.grav*ys
		else: return V
	
	def plotLat(self,grav=False,npts=101,ncells=1,plt='surf'):
		m1,m0 = meshgrid(linspace(-ncells,ncells,npts),linspace(-ncells,ncells,npts))
		xs = m1*self.r[0,1] + m0*self.r[0,0]
		ys = m1*self.r[1,1] + m0*self.r[1,0]
		l1,l0 = meshgrid(arange(-ncells,ncells+1),arange(-ncells,ncells+1))
		rx = l0*self.r[0,0] + l1*self.r[0,1]
		ry = l0*self.r[1,0] + l1*self.r[1,1]
		V = self.getLat(xs,ys,grav)
		if plt=='surf':
			fig = figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.plot_surface(xs,ys,V)
		elif plt=='imshow':
			imshow(V)
		elif plt=='contourf':
			contourf(xs,ys,V)
			plot(rx,ry,'.',color='hotpink')
		elif plt=='pcolor':
			pcolor(xs,ys,V)
			plot(rx,ry,'.',color='hotpink')
	
	def eigs(self,q,nbands,**kwargs):
		return eigs2(q,self.b,self.A,nbands,**kwargs)
	
	def bandPath(self,qs,nbands,**kwargs):
		engs = zeros((qs.shape[1],nbands))
		for i in range(qs.shape[1]):
			engs[i] = self.eigs(qs[:,i],nbands,**kwargs)[0]
		return engs
	
	def bandArray(self,qx,qy,nbands,**kwargs):
		if not qx.shape==qy.shape: raise ValueError("qx and qy must have same shape.")
		engs = zeros(qx.shape+(nbands,))
		for i in multirange(qx.shape):
			engs[i] = self.eigs([qx[i],qy[i]],nbands,**kwargs)[0]
		return engs
	
	# The next few methods & attributes show structure of the Brillouin zone
	def showRLattice(self,n1,n2=None,fig=None):
		"""Plots the reciprocal lattice and first Brillouin zone.
			n1 and n2 are how many lattice points to show in the b[0] and b[1] directions.
			"""
		if n2 is None: n2 = n1
		b0 = self.b[:,0]; b1 = self.b[:,1]
		if fig is None:
			figr = figure()
		elif isinstance(fig,Figure):
			figr = fig
			#ax = fig.gca(projection='3d')
		else:
			figr = figure(fig)
		for i in range(-n1,n1+1):
			for j in range(-n2,n2+1):
				plot((b0*i+b1*j)[0],(b0*i+b1*j)[1],'bo')
		plot(0,0,'k*')
		c = self.bzCorners
		for i in range(6):
			plot([c[0,i-1],c[0,i]],[c[1,i-1],c[1,i]],'r')
	
	def showBandPath(self,path,nbands=4,fig=None,scenery=True,zero=True,**kwargs):
		"""Show bands along path.
			path is a 2 by n array of n points at which to evaluate the bands."""
		c = self.bzCorners
		b0 = self.b[:,0]; b1 = self.b[:,1]
		r = ptRange(c,5)
		if fig is None:
			fig = figure()
		elif isinstance(fig,Figure):
			figure(fig.number)
		else:
			fig = figure(fig)
		ax = fig.gca(projection='3d')
		bands = self.bandPath(path,nbands,**kwargs)
		if zero: bands -= amin(bands)
		for i in range(nbands):
			plot(path[0],path[1],bands[:,i])
		if scenery:
			plot(r[0],r[1],'r-*')
			for i in range(-1,2):
				for j in range(-1,2):
					plot((b0*i+b1*j)[0:1],(b0*i+b1*j)[1:2],[0.],'bo')
	
	def showbandary(self,npts=10,nbands=4,**kwargs):
		"""Show bands around boundary of Brillouin zone."""
		path = ptRange(self.bzCorners,npts)
		self.showBandPath(path,nbands,**kwargs)
	
	"""	Conversions between frames
		The 'L-frame' is the lab frame with H = P^2/2m + mgz + Sum_ij A_ij cos(k_ij (x-theta))
		The 'G-frame' is the frame with the gravity term unitaried into the p^2 kinetic term
		The 'P-frame' is the frame with gravity and lattice motion in the p^2 kinetic term
		The 'M-frame' is the frame with gravity and lattice motion as terms linear in p
		The 'S-frame' is the 'solid-state' frame with gravity and lattice motion as terms linear in x
		The 'A-frame' is the frame with gravity and lattice motion as a phase in the cosines
		
		Additionally, in each frame there is a k-basis of momentum states & a b-basis of Bloch states
		"""
	
	def getMomenta(self,q,n):
		""" Returns momenta px,py for a state with quasimomentum q and cutoff n."""
		m1,m2 = meshgrid(arange(-n,n+1),arange(-n,n+1))
		px = m1*self.b[0,1] + m2*self.b[0,0] + q[0]
		py = m1*self.b[1,1] + m2*self.b[1,0] + q[1]
		return px,py
	
	def Pbloch(self,q,nbands,n=None):
		""" Returns the matrix elements of the momentum operator in the Bloch basis,
			for quasimomentum q.  
			Specifically, the return is (P_x,P_y).
			"""
		if n is None: n = nbands
		engs,evecs = self.eigs(q,nbands,n=n,wind=True)
		px,py = self.getMomenta(q,n)					# These are momenta in p-basis
		
		PX = zeros((nbands,nbands),dtype=complex)		# These will be momenta in bloch-basis
		PY = zeros((nbands,nbands),dtype=complex)
		
		for i in range(nbands):
			for j in range(i):
				PX[i,j] = sum(evecs[i].conj()*px*evecs[j])
				PY[i,j] = sum(evecs[i].conj()*py*evecs[j])
			PX[i,i] = sum(evecs[i].conj()*px*evecs[i])/2.			# Breaking it up like this might give a slight speed boost
			PY[i,i] = sum(evecs[i].conj()*py*evecs[i])/2.
		PX += PX.T.conj()
		PY += PY.T.conj()
		return PX,PY
	
	def Hbloch(self,q,nbands,n=None):
		""" Returns matrix elements of lattice Hamiltonian H_0 in Bloch basis,
			for quasimomentum q.
			"""
		if n is None: n = nbands
		engs, evecs = self.eigs(q,nbands,n=n,wind=True)
		return diag(engs)
	
	def moveGrav(self,moveTo,moveFrom,c,px,py,t):
		""" Moves gravity from the moveFrom term to the moveTo term.  Operates on the k-basis.  t is time."""
		if moveFrom=='x1':
			if moveTo=='x1':
				return c,px,py
			elif moveTo=='p2':
				return c,px,py+.5*t*self.grav
			elif moveTo in {'p1','phase'}:		# Move to p2 and iterate
				return self.moveGrav(moveTo,'p2',c,px,py+.5*t*self.grav,t)
		elif moveFrom=='p2':
			if moveTo=='x1':
				return c,px,py-.5*t*self.grav
			elif moveTo=='p2':
				return c,px,py
			elif moveTo=='p1':
				return c*exp(.5j*self.grav**2*t**3/6.),px,py
			elif moveTo=='phase':
				return self.moveGrav(moveTo,'p1',c*exp(.5j*self.grav**2*t**3/6.),px,py,t)
		elif moveFrom=='p1':
			if moveTo=='x1':
				return self.moveGrav(moveTo,'p2',c*exp(-.5j*self.grav**2*t**3/6.),px,py,t)
			elif moveTo=='p2':
				return c*exp(-.5j*self.grav**2*t**3/6.),px,py
			elif moveTo=='p1':
				return c,px,py
			elif moveTo=='phase':
				return c*exp(-.5j*self.grav*t**2*py),px,py
		elif moveFrom=='phase':
			if moveTo in {'x1','p2'}:
				return self.moveGrav(moveTo,'p1',c*exp(.5j*self.grav*t**2*py),px,py,t)
			elif moveTo=='p1':
				return c*exp(.5j*self.grav*t**2*py),px,py
			elif moveTo=='phase':
				return c,px,py
	
	def moveMotionRight(self,moveTo,moveFrom,c,px,py,t,Phx,Phy):
		""" Moves lattice motion from the moveFrom term to the moveTo term.  
		Operates on the k-basis.  t is time, Ph is the lattice motion as a PHASE.
		(Taking lattice motion as a phase minimizes the number of integrals to be performed."""
		if moveFrom=='x1':
			if moveTo=='x1':
				return c,px,py
			elif moveTo=='p2':
				return c,px+.5*Dt(Phx,t),py+.5*Dt(Phy,t)
			elif moveTo in {'p1','phase'}:		# Move to p2 and iterate
				return self.moveMotionRight(moveTo,'p2',c,px+.5*Dt(Phx,t),py+.5*Dt(Phy,t),t,Phx,Phy)
		elif moveFrom=='p2':
			if moveTo=='x1':
				return c,px-.5*Dt(Phx,t),py-.5*Dt(Phy,t)
			elif moveTo=='p2':
				return c,px,py
			elif moveTo=='p1':
				Ph2int = quad(fDt2(Phx),0,t)+quad(fDt2(Phy,0,t))
				return c*exp(.25j*Ph2int),px,py
			elif moveTo=='phase':
				Ph2int = quad(fDt2(Phx),0,t)+quad(fDt2(Phy,0,t))
				return self.moveMotionRight(moveTo,'p1',c*exp(.25j*Ph2int),px,py,t,Phx,Phy)
		elif moveFrom=='p1':
			if moveTo=='x1':
				Ph2int = quad(fDt2(Phx),0,t)+quad(fDt2(Phy,0,t))
				return self.moveMotionRight(moveTo,'p2',c*exp(-.25j*Ph2int),px,py,t,Phx,Phy)
			elif moveTo=='p2':
				Ph2int = quad(fDt2(Phx),0,t)+quad(fDt2(Phy,0,t))
				return c*exp(-.25j*Ph2int),px,py
			elif moveTo=='p1':
				return c,px,py
			elif moveTo=='phase':
				return c*exp(-1j*(Phx(t)*px + Phy(t)*py)),px,py
		elif moveFrom=='phase':
			if moveTo in {'x1','p2'}:
				return self.moveMotionRight(moveTo,'p1',c*exp(1j*(Phx(t)*px + Phy(t)*py)),px,py,t,Phx,Phy)
			elif moveTo=='p1':
				return c*exp(1j*(Phx(t)*px + Phy(t)*py)),px,py
			elif moveTo=='phase':
				return c,px,py
	
	def moveMotionCheat(self,moveTo,moveFrom,c,px,py,t,Phx,Phy):
		""" Moves lattice motion from the moveFrom term to the moveTo term.  
		Operates on the k-basis.  t is time, Ph is the lattice motion as a PHASE.
		(Taking lattice motion as a phase minimizes the number of integrals to be performed."""
		if moveFrom=='x1':
			if moveTo=='x1':
				return c,px,py
			elif moveTo=='p2':
				return c,px+.5*Dt(Phx,t),py+.5*Dt(Phy,t)
			elif moveTo in {'p1','phase'}:		# Move to p2 and iterate
				return self.moveMotionCheat(moveTo,'p2',c,px+.5*Dt(Phx,t),py+.5*Dt(Phy,t),t,Phx,Phy)
		elif moveFrom=='p2':
			if moveTo=='x1':
				return c,px-.5*Dt(Phx,t),py-.5*Dt(Phy,t)
			elif moveTo=='p2':
				return c,px,py
			elif moveTo=='p1':
				return c,px,py
			elif moveTo=='phase':
				return self.moveMotionCheat(moveTo,'p1',c,px,py,t,Phx,Phy)
		elif moveFrom=='p1':
			if moveTo=='x1':
				return self.moveMotionCheat(moveTo,'p2',c,px,py,t,Phx,Phy)
			elif moveTo=='p2':
				return c,px,py
			elif moveTo=='p1':
				return c,px,py
			elif moveTo=='phase':
				return c*exp(-1j*(Phx(t)*px + Phy(t)*py)),px,py
		elif moveFrom=='phase':
			if moveTo in {'x1','p2'}:
				return self.moveMotionCheat(moveTo,'p1',c*exp(1j*(Phx(t)*px + Phy(t)*py)),px,py,t,Phx,Phy)
			elif moveTo=='p1':
				return c*exp(1j*(Phx(t)*px + Phy(t)*py)),px,py
			elif moveTo=='phase':
				return c,px,py
	
	def tobBasis(self,c,q,nbands=7,phase=[0.,0.]):
		n = (c.shape[1]-1)//2
		ys = [-self.b[0,i]*phase[0]-self.b[1,i]*phase[1] for i in range(3)]
		mats = self.eigs(q,nbands,n=n,ys=ys,wind=True)[1]
		return array([ sum(mats[i].conj()*c) for i in range(nbands)])
	
	def tokBasis(self,c,q,phase=[0.,0.],n=10):
		ys = [-self.b[0,i]*phase[0]-self.b[1,i]*phase[1] for i in range(3)]
		mats = self.eigs(q,len(c),n=n,ys=ys,wind=True)[1]
		return tensordot(c,mats,(0,0))
	
	def convert(self,fin,bin,fout,bout,c,Phx,Phy,t,px=None,py=None,q=None,nbands=7,n=10,cheat=True):
		""" Converts wavefunction from one frame & basis to another.
			Internally frame conversion is always done in the k-basis.
			
			Gravity and lattice motion can be in several places:
				m x ... 	+ (p-...)^2/2m	- p ... 	+ Sum_ij A_ij cos(k_ij (x - ...))
					 ^			  ^				 ^									 ^
					 x1			  p2			 p1								   phase
						<------->	 <--------->	<---------------------------->
						exp(ix...)	  exp(it...)			exp(ip...)
			
			The 2's and 1's after p or x indicate the degree of the term.  The ...'s are not
			all the same.  The arrows and text below them indicate the unitary transformations
			to move terms around from one place to another. 
			"""
		
		Gstart  = {'L':'x1'	,'G':'p2'	,'P':'p2'	,'M':'p1'	,'S':'x1'	,'A':'phase'}[fin]		# Where gravity starts out
		Gtarget = {'L':'x1'	,'G':'p2'	,'P':'p2'	,'M':'p1'	,'S':'x1'	,'A':'phase'}[fout]		# Where gravity should end up
		Mstart  = {'L':'phase','G':'phase','P':'p2'	,'M':'p1'	,'S':'x1'	,'A':'phase'}[fin]		# Where lattice motion starts
		Mtarget = {'L':'phase','G':'phase','P':'p2'	,'M':'p1'	,'S':'x1'	,'A':'phase'}[fout]		# Where lattice motion should end up
		
		if cheat: 
			moveMotion = self.moveMotionCheat
		else: moveMotion = self.moveMotionRight
		
		if bin=='b':
			if q is None: raise ValueError("q must be supplied to convert from b-basis")
			phasex = 0.; phasey = 0.;
			dpx = 0.; dpy = 0.; 
			dqx = 0.; dqy = 0.;
			if Mstart=='phase':
				phasex += Phx(t)
				phasey += Phy(t)
			elif Mstart=='p2':
				dpx += Dt(Phx,t)
				dpy += Dt(Phy,t)
			elif Mstart=='p1':
				pass
			elif Mstart=='x1':
				dqx += Dt(Phx,t)
				dqy += Dt(Phy,t)
			
			if Gstart=='phase':
				phasey += .5*t**2*self.grav
			elif Gstart=='p2':
				dpy += .5*self.grav*t
			elif Gstart=='p1':
				pass
			elif Gstart=='x1':
				dqy += .5*self.grav
			
			print(dpx,dpy,dqx,dqy,phasex,phasey)
			c = self.tokBasis(c,[q[0]-dpx-dqx,q[1]-dpy-dqy],phase=[phasex,phasey],n=n)
			px,py = self.getMomenta([q[0]-dqx,q[1]-dqy],n)
		elif bin=='k':
			if px is None:
				if q is None: raise ValueError("Either q or px & py must be supplied")
				px,py = self.getMomenta([q[0],q[1]],n)
		c,px,py = self.moveGrav(Gtarget,Gstart,c,px,py,t)
		c,px,py = moveMotion(Mtarget,Mstart,c,px,py,t,Phx,Phy)
		
		if bout=='k':
			return c,px,py
		elif bout=='b':
			n = (px.shape[1]-1)//2
			q = [px[n,n],py[n,n]]
			phasex = 0.; phasey = 0.;
			dpx = 0.; dpy = 0.; 
			dqx = 0.; dqy = 0.;
			if Mtarget=='phase':
				phasex += Phx(t)
				phasey += Phy(t)
			elif Mtarget=='p2':
				dpx += Dt(Phx,t)
				dpy += Dt(Phy,t)
			elif Mtarget=='p1':
				pass
			elif Mtarget=='x1':
				dqx += Dt(Phx,t)
				dqy += Dt(Phy,t)
			
			if Gtarget=='phase':
				phasey += .5*t**2*self.grav
			elif Gtarget=='p2':
				dpy += .5*self.grav*t
			elif Gtarget=='p1':
				pass
			elif Gtarget=='x1':
				dqy += .5*self.grav
			
			print(dpx,dpy,dqx,dqy,phasex,phasey)	
			c = self.tobBasis(c,[q[0]-dqx-dpx,q[1]-dqy-dpy],nbands=nbands,phase=[phasex,phasey])
			return c
		

def Dt(f,t,dt=1.e-6):
	return (f(t+dt)-f(t-dt))/(2.*dt)

def D2t(f,t,dt=1.e-6):
	return (f(t+dt)-2.*f(t)+f(t-dt))/(2.*dt)

def fDt2(f,dt=1.e-6):
	return lambda t : ((f(t+dt)-f(t-dt))/(2.*dt))**2

def emp(a,b,tol=1.e-14):
	"""Equal mod phase.  Tests near equality of arrays a and b modulo overall phase difference."""
	if not (hasattr(a,'shape') and hasattr(b,'shape') and a.shape==b.shape): return False
	idx = argmax(abs(a))
	ph = b.flatten()[idx]/a.flatten()[idx]
	return tol>amax(abs(b-a*ph))
	
################### 2D band structure ############################
""" Taken from bands.py """

def LHam(q,bs,amps,n,ys=[0,0,0],DP=[0.,0.],M=True):
	'''Returns a k-space matrix Hamiltonian for an optical lattice.
	NOTE: THIS VERSION HAS A DIFFERENT CONVENTION FOR bs THAN bands.py
		* q is a quasimomentum vector in the 1st Brillioun zone.
		* bs is a 2 by 2 array k-space basis COLUMN vectors
			- can also be 2 by n, in which case only bs[:,:2] is used
		* amps is a 3-iterable of amplitudes.  amps[0] and amps[1] 
			correspond to bs[0] and bs[1], and amps[2] corresponds to
			-bs[1]-bs[2]
		* y is a 3-iterable of phases.  The correspondence is as for amps.
		* DP is a momentum shift, so the kinetic term of the Hamiltonian
			becomes (p-DP)**2/2m, or (p-DP)**2 in our units.
		* n is the wavevector cutoff
		The lattice Schrodinger equation is given explicitly by:
			i hbar dc(p)/dt = P^2/2m c(p) + Sum_{j=0}^{3} amps[j]/2 * (exp(1j*ys[j]) c(p-bs[j]) + exp(-1j*ys[j]) c(p+bs[j]))
		By default (M=True), a matrix is returned.  The matrix operates
		on vectors c[i] defined so that c[i] is the coefficient of 
		exp[ ((-n+(i mod(2n+1)))*bs[0]+q).r) + (((-n+floor(i/(2n+1)))*bs[1]+q).r) ]]
		in a Fourier expansion of the wavefunction.
		'''
	
	N=2*n+1						# The Hamiltonian matrix will be of size N^2 by N^2
	idx = arange(N**2)			# Indices for the vectors
	# dia is the kinetic part of the Hamiltonian
	dia = 	(q[0] + bs[0,0]*(-n+mod(idx,N))+bs[0,1]*(-n+floor(idx/N)) - DP[0])**2 + \
			(q[1] + bs[1,0]*(-n+mod(idx,N))+bs[1,1]*(-n+floor(idx/N)) - DP[1])**2
	
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

def eigs2(q,bs,amps,nbands,ys=[0,0,0],n=None,returnM=False,wind=False,DP=[0.,0.]):
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
	M = LHam(q,bs,amps,n,ys,DP)			# Get the Hamiltonian matrix
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

################## Time evolution ################################

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

def bevolve(ham,motion,T,q,nbands=None,c0=None,n=None,talk=False,plt='path'):
	"""Evolves state from initial condition c0 via Hamiltonian ham+motion*p,
		through times in T.
		motion should be a function of time returning a 2-iterable.
		The Hamiltonian being solved is:
			p^2/2m + motion*p + (Sum_i A_i cos(k_i * x))
		"""
	if c0 is None:
		if nbands is None: raise ValueError('c0 and nbands cannot both be unspecified.')
		c0 = zeros(nbands,dtype=complex)
		c0[0] = 1.0
	else: nbands = len(c0)
	if n is None: n = nbands
	
	PX,PY = ham.Pbloch(q,nbands,n=n)
	H = ham.Hbloch(q,nbands,n=n)
	def Dt(c,t): 
		m = motion(t)
		return -1j * (H + m[0]*PX + m[1]*PY).dot(c)
	
	c = zeros((len(T),len(c0)),dtype=complex)			# Matrix of wavefunction coefficients
	c[0] = c0
	
	tol = 1.e-7							# Error tolerance
	refine = 1.5						# How much to improve resolution each iteration
	for i in range(1,len(T)):
		cPrev = c[i-1]
		dc = Dt(cPrev,T[i])
		mdc = amax(abs(dc))
		nsteps = int(ceil(max(2,(T[i]-T[i-1])*mdc*10)))
		cNew = midpoint(cPrev,Dt,T[i-1],T[i],nsteps)
		
		err = 2*tol
		while err>tol:
			nsteps = int(ceil(nsteps*refine))
			cOld = cNew
			cNew = midpoint(cPrev,Dt,T[i-1],T[i],nsteps)
			err = amax(abs(cOld-cNew))
		c[i] = cNew
		if talk:
			print('Completed step {} of {}'.format(i,len(T)))
	
	if plt=='path':
		path = qpath(q,T,motion,False).T
		engs = ham.bandPath(path,nbands,n=n)
		figure()
		plot(T,abs(c)**2,'o')
		plot(T,engs)
		fig = figure()
		ham.showRLattice(2,fig=fig)
		plot(path[0],path[1],'x',color='limegreen')
		
	return c

def motion(vel=0.,accel=0.,grav=None,gm=1.,ham=None,theta=None):
	"""Makes a function describing gravity and the motion of the lattice, for
		use in b-frame time evolution.
		"""
	if grav is None and ham is not None:
		grav = ham.grav
	elif grav is None:
		grav = gSr
	grav *= gm
	
	if theta is None:
		u = lambda t: [vel + a*t,grav*t]
	else:
		def u(t):
			th = theta(t)
			return [vel + a*t + th[0],grav*t + th[1]]

def qpath(q0,T,motion,zero=True):
	"""Returns the path followed by quasimomentum starting at q0 and evolving in
		the s-frame under motion."""
	q = zeros((len(T),2))
	for i in range(len(T)):
		q[i] = motion(T[i])
	q *= .5							# This is because mass m=.5 in the units used here
	if zero: q -= q[0]
	q += reshape(q0,(2,))
	return q

