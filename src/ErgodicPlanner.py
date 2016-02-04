import numpy as np
#import GP 
#import sys
#import trep
#from trep import tx, ty, tz, rx, ry, rz
from scipy.integrate import trapz
from numpy import dot
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.interpolate import interp1d
#import math
#from math import sin, cos
#from math import pi as mpi
#import trep.visual as visual
#from PyQt4.QtCore import Qt, QRectF, QPointF
#from PyQt4.QtGui import QColor
from scipy.integrate import quad
import matplotlib.pyplot as plt
from numpy.linalg import pinv as inverse
from matplotlib import cm
import time

def matmult(*x):
    """
    Shortcut for standard matrix multiplication.
    matmult(A,B,C) returns A*B*C.
    """
    return reduce(np.dot, x)

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))"""

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def plot_all(X):
    tj=X.T
    for ln in range(0,tj.shape[0]):
        plt.plot(tj[ln])
    plt.show()
    
def uniform_pdf(X):
        return np.ones(X[0].shape)
        
def gaussian_pdf(X):
    mux = .25
    muy = .3
    mux2 = .24
    muy2=.4
    varx = .1
    vary = .1
    
    pd= np.exp(-((X[0] - mux)**2/( 2*varx**2)) -
               ((X[1] - muy)**2/(2*vary**2))) + \
               0*np.exp(-((X[0] - mux)**2/( 2*varx**2)) -
               ((X[1] - muy)**2/(2*vary**2))) + \
               0*np.exp(-((X[0] - mux2)**2/( 2*varx**2)) -
                      ((X[1] - muy2)**2/(2*vary**2)))
    return pd

class pdf(object):
    def __init__(self,res = 400,wlimit=float(1),  dimw=2):
        self.dimw = dimw # workspace dimension. current implementation only works for 2D
        self.dim=dim
        self.wlimit = wlimit
        self.res = res
    
class lqrsolver(object):
    def __init__(self, nx,nu,Nfourier=10,res = 400, dim=2, wlimit=float(1),  dimw=2,barrcost=1,contcost=.05,ergcost=10):
        # self.A=system.dfdx
        # self.B=system.dfdu
        self.nx=nx
        self.nu=nu
        self.barrcost=barrcost
        self.ergcost=ergcost
        self.contcost=contcost*np.eye(self.nu)
        self.P1= np.eye(nx) #* dt
        # self.r1= cost.dmdx #* dt
        self.Qn=np.eye(self.nx)
        self.Rn=np.eye(self.nu)
        # self.z0=np.zeros(system.nx)
        self.Qk=np.eye(self.nx)
        self.Rk=np.eye(self.nu)
        self.Nfourier=Nfourier
        # self.cost=cost
        self.dimw = dimw # workspace dimension. current implementation only works for 2D
        self.dim=dim
        self.wlimit = wlimit
        self.res = res
        # Check that the number of coefficients matches the workspace dimension
        if type(self.wlimit) == float:
            self.wlimit = [self.wlimit] * self.dimw
        elif len(self.wlimit) != self.dimw:
            raise Exception(
                "dimension of xmax_workspace \
                does not match dimension of workspace")
        if type(self.Nfourier) == int:
        # if a single number is given, create tuple of length dim_worksapce
            self.Nfourier = (self.Nfourier,) * self.dimw  # if tuples
        elif len(self.Nfourier) != self.dimw:
            raise Exception("dimension of Nfourier \
            does not match dimension of workspace")
                    # self.dimw=wdim
        # setup a grid over the workspace
        #xgrid=[]
        xgrid=[np.linspace(0, wlim, self.res) for wlim in self.wlimit]

        # just this part needs to be fixed for 2d vs 1d vs 3d
        self.xlist = np.meshgrid(xgrid[0], xgrid[1], indexing='ij')

        # set up a grid over the frequency
        klist = [np.arange(kd) for kd in self.Nfourier]
        klist = cartesian(klist)

        # do some ergodic stuff
        s = (float(self.dimw) + 1)/2;
        self.Lambdak = 1/(1 + np.linalg.norm(klist,axis=1)**2)**s
        self.klist = klist/self.wlimit * np.pi
        self.hk=np.zeros(self.Nfourier).flatten()
        for index,val in np.ndenumerate(self.hk):
            hk_interior=1
            for n_dim in range(0,self.dimw):
                integ=quad(lambda x: (np.cos(x*self.klist[index][n_dim]))**2,
                           0, float(self.wlimit[n_dim]))
                hk_interior=hk_interior*integ[0]
            self.hk[index]=np.sqrt(hk_interior)
    def set_pdf(self, pdf):
        self.pdf = pdf
        self.normalize_pdf()
        self.calculate_uk(self.pdf)
        pass

    def normalize_pdf(self):
        sz=np.prod([n/float(self.res) for n in self.wlimit])
        summed = sz * np.sum(self.pdf.flatten())
        self.pdf= self.pdf/summed        
    def calculate_uk(self, pdf):
        self.pdf=pdf
        self.uk=np.zeros(self.Nfourier).flatten()
        for index,val in np.ndenumerate(self.uk):
            uk_interior=1/self.hk[index] * pdf
            for n_dim in range(0,self.dimw):
                basis_part = np.cos(self.klist[index][n_dim] \
                                    * self.xlist[n_dim])
                uk_interior = self.wlimit[n_dim]/self.res \
                              * uk_interior*basis_part
            self.uk[index] = np.sum(uk_interior) #sum over # XXX:
        pass
    
    def calculate_ergodicity(self):
        self.erg = np.sum(self.Lambdak * (self.ck - self.uk)**2)
        return self.erg

    def config_to_workspace(self,X):
        # goes from q to the workspace config
        # e.g. from joint angles to XY position on a table top
        XT=X.T
        W=np.array([XT[0],XT[1]])
        return W.T

    def DWDX(self,xk):
        x_workspace=np.array([[1,0,0,0],[0,1,0,0]])
        return (x_workspace).T

    def barrier(self,xk):
        barr_cost=np.zeros(xk.shape[0])
        
        xk=self.config_to_workspace(xk)
        xk=xk.T

        for n in range(0,self.dimw):
            too_big = xk[n][np.where(xk[n]>self.wlimit[n])]
            barr_cost[np.where(xk[n]>self.wlimit[n])]+=np.square(too_big-self.wlimit[n])
            too_small = xk[n][np.where(xk[n]<0)]
            barr_cost[np.where(xk[n]<0)]+=np.square(too_small-0)
            #barr_cost+=(too_small)**2
        barr_cost=trapz(barr_cost,self.time)
        return barr_cost
    
    def Dbarrier(self,xk):
        xk=self.config_to_workspace(xk)
        xk=xk.T
        dbarr_cost=np.zeros(xk.shape)
        # for n in range(0,self.dimw):
        #     if xk[n]>self.wlimit[n]:
        #         dbarr_cost[n]=2*(xk[n]-self.wlimit[n])
        #     if xk[n]<0:
        #         dbarr_cost[n]=2*(xk[n]-0)
        # dbarr_cost=np.zeros(xk.shape[0])
        for n in range(0,self.dimw):
            too_big = xk[n][np.where(xk[n]>self.wlimit[n])]
            dbarr_cost[n,np.where(xk[n]>self.wlimit[n])]=2*(too_big-self.wlimit[n])
            too_small = xk[n][np.where(xk[n]<0)]
            dbarr_cost[n,np.where(xk[n]<0)]=2*(too_small-0)
            #barr_cost+=(too_small)**2
        return dbarr_cost.T
    
    def ckeval(self):
        X=self.X_current
        time=self.time
        T=time[-1]
        # change coordinates from configuration to ergodic workspace
        W = self.config_to_workspace(X).T
        self.ck=np.zeros(self.Nfourier).flatten()
        #xlist = tj.T
        for index,val in np.ndenumerate(self.ck):
            ck_interior=1/self.hk[index]* 1/(float(T))
            for n_dim in range(0,self.dimw):
                basis_part = np.cos(self.klist[index][n_dim] * W[n_dim])
                ck_interior = ck_interior*basis_part
                self.ck[index]=trapz(ck_interior,time)#np.sum(self.dt*ck_interior)

    def akeval(self):
        X=self.X_current
        time=self.time
        T=time[-1]
        xlist = X.T
        outerchain = 2 * 1/self.hk * 1/(float(T)) * self.Lambdak \
                     * (self.ck-self.uk)
        x_in_w=self.config_to_workspace(X).T
        ak = []
        for index,val in np.ndenumerate(outerchain):
            DcDX = []
            # these are chain rule terms, get added
            for config_dim in range(0,self.dim):
                Dcdx=0
                for term_dim in range(0,self.dimw):
                    term=outerchain[index]
                    for prod_dim in range(0,self.dimw):
                        if term_dim == prod_dim:
                            basis_part=-self.klist[index][prod_dim]*np.sin(
                                self.klist[index][prod_dim] * x_in_w[prod_dim])\
                                *self.DWDX(xlist)[config_dim][prod_dim]
                        else:
                            basis_part = np.cos(self.klist[index][prod_dim] \
                                            * x_in_w[prod_dim])
                        term*=basis_part
                    Dcdx=Dcdx+term
                DcDX.append(Dcdx)
            ak.append(DcDX)
            
        summed_ak=np.sum(np.array(ak),axis=0)
        
        self.ak = summed_ak.T#self.workspace_to_config(summed_ak).T
        return  self.ak

    def peqns(self,t,pp,Al,Bl,Rn,Qn):
        pp=pp.reshape(self.nx,self.nx)
        matdiffeq=(matmult(pp,Al(t)) + matmult(Al(t).T,pp) -
                   matmult(pp,Bl(t),inverse(Rn),Bl(t).T,pp) + Qn)
        return matdiffeq.flatten()

    def reqns(self,t,rr,Al,Bl,a,b,Psol,Rn,Qn):
        t=self.time[-1]-t
        matdiffeq=(matmult(Al(t)-matmult(Bl(t),inverse(Rn),Bl(t).T,Psol(t)).T,rr.T)
                   +a(t)-matmult(Psol(t),Bl(t),inverse(Rn),b(t))) 
        return matdiffeq.flatten()

    def veqns(self,zz,Al,Bl,a,b,Psol,Rsol,Rn,Qn):
        vmatdiffeq=(matmult(-inverse(Rn),Bl.T,Psol,zz) - matmult(inverse(Rn),Bl.T,Rsol) - 
                   matmult(inverse(Rn),b))
        return vmatdiffeq
    
    def zeqns(self,t,zz,Al,Bl,a,b,Psol,Rsol,Rn,Qn):
        vmateq=self.veqns(zz,Al(t),Bl(t),a(t),b(t),Psol(t),Rsol(t),Rn,Qn)
        matdiffeq=matmult(Al(t),zz) + matmult(Bl(t),vmateq)
        return matdiffeq.flatten()
  
    def Ksol(self, X, U):
        time=self.time
        P1 = np.eye(X.shape[1]).flatten()
        solver = ode(self.peqns).set_integrator('dopri5')
        solver.set_initial_value(P1,time[0]).set_f_params(self.A_interp,
                                                          self.B_interp,
                                                          self.Rk,
                                                          self.Qk)
        k = 0
        t=time
        soln = [P1]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            soln.append(solver.y)

        # Convert the list to a numpy array.
        psoln = np.array(soln).reshape(time.shape[0],X.shape[1],X.shape[1])
        K=np.empty((time.shape[0],X.shape[1],X.shape[1]))
        for tindex,t in np.ndenumerate(time):
            K[tindex,:,:]=matmult(inverse(self.Rk),self.B_current[tindex].T,psoln[tindex])
        self.K=K
        return K

    def Psol(self,X, U,time):

        P1 = np.eye(X.shape[1]).flatten()
        solver = ode(self.peqns).set_integrator('dopri5')
        solver.set_initial_value(P1,time[0]).set_f_params(self.A_interp,self.B_interp,
                                                          self.Rn,self.Qn)
        k = 0
        t=time
        soln = [P1]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            soln.append(solver.y)

        soln = np.array(soln)
        return soln.reshape(time.shape[0],X.shape[1],X.shape[1])

    def Rsol(self,X, U,P_interp,time):
        rinit2 = np.zeros(X.shape[1])
        Qn = np.eye(X.shape[1])
        Rn = np.eye(U.shape[1])
        solver = ode(self.reqns).set_integrator('dopri5')
        solver.set_initial_value(rinit2,time[0]).set_f_params(self.A_interp,self.B_interp,self.a_interp,self.b_interp,P_interp,
                                                          Rn,Qn)
        k =0
        t=time
        soln = [rinit2]
        while solver.successful() and solver.t < t[-1]:# 
            k +=1
            solver.integrate(t[k])
            soln.append(solver.y)
           
        soln.reverse()
        soln = np.array(soln)
        return soln
        
    # pointwise dynamics linearizations
    def A(self,x,u):
        xdim=x.shape[0]
        return np.zeros((xdim,xdim))
    
    def B(self,x,u):
        udim=u.shape[0]
        return np.eye(udim)

    def dfdx(self):
        X=self.X_current
        U=self.U_current
        time=self.time
        dfdxl=np.empty((time.shape[0],X.shape[1],X.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dfdxl[tindex,:,:]=self.A(X[tindex],U[tindex])
        self.A_current=dfdxl
        return dfdxl

    def dfdu(self):
        X=self.X_current
        U=self.U_current
        time=self.time
        dfdul=np.empty((time.shape[0],U.shape[1],U.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dfdul[tindex,:,:]=self.B(X[tindex],U[tindex])
        self.B_current=dfdul
        return dfdul
    # pointwise cost linearizations

    def controlcost_point(self,x,u):
        udim=u.shape[0]
        R=self.contcost
        #print u
        return .5*matmult(u.T,R,u)
    
    def controlcost(self,X,U):
        #print u
        cost=np.empty(self.time.shape[0])
        for tindex,t in np.ndenumerate(self.time):
            #print U[tindex]
            cost[tindex]=self.controlcost_point(X[tindex],U[tindex])
        #print cost
        intcost=trapz(cost,self.time)
        return intcost

    def cost(self):
        X=self.X_current
        U=self.U_current
        #print U
        barr_cost=self.barrcost*self.barrier(X)
        erg_cost=self.ergcost*self.calculate_ergodicity()
        cont_cost=self.controlcost(X,U)
        #print "barrcost=", barr_cost
        #print "contcost=", cont_cost
        #print "ergcost=", erg_cost
        #print "J=", barr_cost+erg_cost+cont_cost
        return barr_cost+erg_cost+cont_cost
            
    def dldu_point(self,x,u):
        udim=u.shape[0]
        R=self.contcost
        return matmult(R,u)
    
    def dldx_point(self,x,u):
        xdim=x.shape[0]
        Qtrack=np.zeros(xdim)
        return 0*matmult(x.T,Qtrack,x)  

    def dldx(self):
        X=self.X_current
        U=self.U_current
        time=self.time
        dldxl=np.empty((time.shape[0],X.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dldxl[tindex,:]=self.dldx_point(X[tindex],U[tindex])
        self.a_current=dldxl+self.ergcost*self.ak+self.barrcost*self.Dbarrier(X)  #
        return self.a_current

    def dldu(self):
        X=self.X_current
        U=self.U_current
        time=self.time
        dldul=np.empty((time.shape[0],U.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dldul[tindex,:]=self.dldu_point(X[tindex],U[tindex])
        self.b_current=dldul
        return dldul

    def dcost(self,descdir):
        dX=descdir[0]
        dU=descdir[1]
        time=self.time
        dc=np.empty(time.shape[0])
        for tindex,t in np.ndenumerate(time):
            dc[tindex]=matmult(self.a_current[tindex],dX[tindex])+matmult(self.b_current[tindex],dU[tindex])
        intdcost=trapz(dc,time)
        return intdcost
    
    def descentdirection(self):
        X=self.X_current
        U=self.X_current
        time=self.time
        Ps=self.Psol(X, U,time)
        #plot_all(Ps)
        self.P_current=Ps
        P_interp = interp1d(time, Ps.T)
        Rs=self.Rsol(X, U,P_interp,time)
        self.R_current=Rs
        r_interp = interp1d(time, Rs.T)
        zinit = np.zeros(X.shape[1])
        #print 'here'
        #Qn = np.eye(X.shape[1])
        #Rn = np.eye(U.shape[1])
        # # initialize the 4th order Runge-Kutta solver
        solver = ode(self.zeqns).set_integrator('dopri5')
        # # initial value
        solver.set_initial_value(zinit,time[0]).set_f_params(self.A_interp, self.B_interp,
                                                             self.a_interp, self.b_interp,
                                                             P_interp, r_interp,
                                                             self.Rn, self.Qn)
        #ppsol = odeint(pkeqns,P1,time,args=(A_interp,B_interp))
        k = 0
        t=time
        zsoln = [zinit]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            zsoln.append(solver.y)

        # Convert the list to a numpy array.
        zsoln = np.array(zsoln)
        zsoln=zsoln.reshape(time.shape[0],X.shape[1])
        vsoln=np.empty(U.shape)
        for tindex,t in np.ndenumerate(time):
            vsoln[tindex]=self.veqns(zsoln[tindex],self.A_current[tindex],
                                     self.B_current[tindex],self.a_current[tindex],
                                     self.b_current[tindex],Ps[tindex],Rs[tindex],self.Rn,self.Qn)
        return np.array([zsoln,vsoln])

    def fofx(self,t,X,U):
        return U(t)
    def fofxpoint(self,X,U):
        return U
    
    def proj(self,t,X,K,mu,alpha):
        # print U(t)
        # print K(t)
        # print alpha(t)
        uloc =mu(t) +  matmult(K(t),(alpha(t).T - X.T))
        self.fofxpoint(X,uloc)
        return uloc

    def projcontrol(self,X,K,mu,alpha):
        uloc =mu +  matmult(K,(alpha.T - X.T))
        return uloc
    
    def simulate(self,X0,U,time):
        U_interp = interp1d(time, U.T)
        # # initialize the 4th order Runge-Kutta solver
        solver = ode(self.fofx).set_integrator('dopri5')
        # # initial value
        solver.set_initial_value(X0,time[0]).set_f_params(U_interp)
        #ppsol = odeint(pkeqns,P1,time,args=(A_interp,B_interp))
        k = 0
        t=time
        xsoln = [X0]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            xsoln.append(solver.y)

        # Convert the list to a numpy array.
        xsoln = np.array(xsoln)
        return xsoln
    
    def project(self,X0,traj,time):
        alpha=traj[0]
        mu=traj[1]
        Ks=self.Ksol(alpha, mu)
        K_interp = interp1d(time, Ks.T)
        mu_interp = interp1d(time, mu.T)
        alpha_interp = interp1d(time, alpha.T)
        solver = ode(self.proj).set_integrator('dopri5')
        # # initial value
        solver.set_initial_value(X0,time[0]).set_f_params(K_interp,mu_interp,alpha_interp)
        #ppsol = odeint(pkeqns,P1,time,args=(A_interp,B_interp))
        k = 0
        t=time
        soln = [X0]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            soln.append(solver.y)

        # Convert the list to a numpy array.
        xsoln = np.array(soln)
        usoln=np.empty(mu.shape)
        for tindex,t in np.ndenumerate(time):
            usoln[tindex,:]=self.projcontrol(xsoln[tindex],Ks[tindex],mu[tindex],alpha[tindex])
        return np.array([xsoln,usoln])
        
    def update_traj(self,X,U,time):
        self.time=time
        self.X_current=X
        self.U_current=U
        self.X_interp = interp1d(time, X.T)
        self.U_interp = interp1d(time, U.T)
        self.ckeval()
        self.akeval()
        self.dfdx()
        self.dfdu()
        self.dldx()
        self.dldu()
        self.A_interp = interp1d(time, self.A_current.T)
        self.B_interp = interp1d(time, self.B_current.T)
        self.a_interp = interp1d(time, self.a_current.T)
        self.b_interp = interp1d(time, self.b_current.T)

#Bl=LQ.dfdu(X,U,time)
#time horizon
#dim=2
#udim=2
#t0=0
#tf=10
#dt=100

#sample trajectories
#Xd = np.array([np.linspace(t0,tf,100),np.linspace(t0,tf,100)]).T

#X

def ergoptimize(solver,pdf,state_init,
                control_init=np.array([0,0]),
                t0=0, tf=10, dt=100,
                plot=True,
                maxsteps=20):
    plt.close("all")
    #initialize system
    xdim=2
    udim=2

    solver.set_pdf(pdf)
    time=np.linspace(t0,tf,dt)
    
    #state_init=np.array(state_init)
    U0 = np.array([np.linspace(control_init[0],control_init[0],dt),np.linspace(control_init[1],control_init[1],dt)]).T
    X0 =solver.simulate(state_init,U0,time)
    solver.update_traj(X0,U0,time)

    #set up some containers
    costs=[solver.cost()]
    trajlist=[np.array([X0,U0])]
    descdirlist=[]
    dcosts=[]

    #linesearch parameters
    alpha=.001
    beta=.5

    #print "init ck=", solver.ck
    #print "uk=", solver.uk
    #print "init J", costs[0]
    # Setup Plots
    if plot==True:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2,aspect=maxsteps/costs[0])
        ax3 = fig.add_subplot(1, 3, 3,aspect=abs(maxsteps/12.0))
        ax2.set_xlim([0,maxsteps])
        ax2.set_title('J')
        ax2.set_ylim([0,costs[0]])
        ax3.set_xlim([0,maxsteps])
        ax3.set_ylim([-10.0,2])
        ax3.set_title('DJ')
        #pdf=LQ.pdf
        #wlim=LQ.wlimit
        q=solver.X_current.T
        ax1.imshow(np.squeeze(solver.pdf.T),extent=(0, solver.wlimit[0], 0, solver.wlimit[1]),
               cmap=cm.binary, aspect='equal',origin="lower")
        ax1.plot(q[0], q[1], '-c', label='q',lw=1.5)
        ax2.plot(costs, '*', label='q',lw=1.5)
        ax3.plot(dcosts, '-', label='q',lw=1.5)

        plt.draw()
        plt.pause(.01)
    k=0
    for k in range(0,maxsteps,1):
        print "*****************k=", k
        descdir=solver.descentdirection()
        newdcost=solver.dcost(descdir)
        print "DJ=", newdcost
        gamma=1
        newtraj=solver.project(state_init,trajlist[k]+gamma*descdir,time)
        solver.update_traj(newtraj[0],newtraj[1],time)
        newcost=solver.cost()
        while newcost > (costs[k] + alpha*gamma*newdcost) and gamma>.00000000001:
            gamma=beta*gamma
            print gamma
            newtraj=solver.project(state_init,trajlist[k]+gamma*descdir,time)
            solver.update_traj(newtraj[0],newtraj[1],time)
            newcost=solver.cost()
        print "gamma=", gamma
        print "new J=", newcost
        if plot==True:
            q=newtraj[0].T
            ax1.plot(q[0], q[1],  label='q',
                     lw=1,alpha=0.4)
            ax2.plot(costs, '-k', label='q', lw=1)
            ax2.plot(k,costs[-1], '*', label='q', lw=1)
            ax3.plot(dcosts, '-', label='q',lw=1.5)
            plt.draw()
            plt.pause(.01)
    
        costs.append(newcost)
        descdirlist.append(descdir)
        dcosts.append(np.log10(np.abs(newdcost)))
        trajlist.append(np.array([solver.X_current,solver.U_current]))
    if plot==True:
        q=newtraj[0].T
        ax1.plot(q[0], q[1],  label='q',
                 lw=3,alpha=0.4)
    return solver.X_current

X = .2*np.array([[-4.61611719, -6.00099547],
              [4.10469096, 5.32782448],
              [0.00000000, -0.50000000],
              [-6.17289014, -4.6984743],
              [1.3109306, -6.93271427],
              [-5.03823144, 3.10584743],
              [-2.87600388, 6.74310541],
              [5.21301203, 4.26386883]])
# xdim=2
# udim=2
# LQ=lqrsolver(xdim,udim, Nfourier=10, res=100,barrcost=50,contcost=.1,ergcost=10)


# #gaussian_pdf(LQ.xlist)
# GP=GP.GaussianProcessModel(res=LQ.res)
# GP.update_GP(X)
# pdf=GP.uncertainty

# xinit=np.array([0.501,.1])
# U0=np.array([0,0])
# pdf=uniform_pdf(LQ.xlist)
# #trajtotal=np.array([])
# for j in range (1,10,1):
#     traj=ergoptimize(pdf, xinit,control_init=U0,maxsteps=20)
#     if j>1:
#         trajtotal=np.concatenate((trajtotal,traj),axis=0)
#     else:
#         trajtotal=traj    
#     GP.update_GP(trajtotal)
#     pdf=GP.uncertainty
#     xinit=trajtotal[-1]
