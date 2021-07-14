import numpy as np
from matplotlib import pyplot as plt

def BackwardPass(variations,trajectory,ctrl,mu=0.0):
    N = trajectory.shape[0]
    minControlParam = {}
    minControlParam['k'] = []
    minControlParam['K'] = []
    minControlParam['deltaV'] = []
    minControlParam['V'] = []
    numStates = trajectory.shape[1]
    numControls = ctrl.shape[1]
    V = 0
    Vx = None
    Vxx = None
    zipTC = list(zip(trajectory[:-1],ctrl))
    for trajElem,ctrlElem in reversed(zipTC):
        L = variations['L'](trajElem,ctrlElem)
        Lx = variations['Lx'](trajElem,ctrlElem)
        Lu = variations['Lu'](trajElem,ctrlElem)
        Lxx = variations['Lxx'](trajElem,ctrlElem)
        Luu = variations['Luu'](trajElem,ctrlElem)
        Lux = variations['Lux'](trajElem,ctrlElem)
        Fx = variations['Fx'](trajElem,ctrlElem)
        Fu = variations['Fu'](trajElem,ctrlElem)
        Fxx = variations['Fxx'](trajElem,ctrlElem)
        Fuu = variations['Fuu'](trajElem,ctrlElem)
        V = L + V
        if Vx is None:
            Vx = Lx
            Vxx = Lxx
        Qx = Lx + np.dot(Fx.T,Vx)
        Qu = Lu + np.dot(Fu.T,Vx) 
        Qxx = Lxx + np.dot(np.dot(Fx.T,Vxx),Fx)
        Quu = Luu + np.dot(np.dot(Fu.T,Vxx),Fu)
        for i in range(numStates):
            Qxx += Fxx[i]*Vx[i,0]
            Quu += Fuu[i]*Vx[i,0]
        Quu = Quu + mu*np.identity(Fu.shape[1])
        Qux = Lux + np.dot(np.dot(Fu.T,Vxx),Fx)

        Vx = Qx - np.dot(np.dot(Qux.T,np.linalg.inv(Quu)),Qu)
        Vxx = Qxx - np.dot(np.dot(Qux.T,np.linalg.inv(Quu)),Qux)
        deltaV = np.squeeze(-0.5*np.dot(np.dot(Qu.T,np.linalg.inv(Quu)),Qu))
        k = -np.dot(np.linalg.inv(Quu),Qu)
        K = -np.dot(np.linalg.inv(Quu),Qux)
        minControlParam['k'].insert(0,k)
        minControlParam['K'].insert(0,K)
        minControlParam['deltaV'].insert(0,deltaV)
        minControlParam['V'].insert(0,V)
    return minControlParam

def ForwardPass(model,minControlParam,trajectory,control,alpha=0.01):
    newTrajectory = []
    newControl = []
    newTrajectory.append(trajectory[0,:])
    N = trajectory.shape[0]
    k = minControlParam['k']
    K = minControlParam['K']
    numStates = trajectory.shape[1]
    numControls = control.shape[1]
    Xt = trajectory[0,:].reshape(numStates,1)
    for i in range(N-1):
        Xprev  = trajectory[i,:].reshape(numStates,1)
        Ut = control[i].reshape(numControls,1) + alpha*k[i] + np.dot(K[i],(Xt - Xprev))
        Xt = model(Xt,Ut)
        newTrajectory.append(np.squeeze(Xt,-1))
        newControl.append(np.squeeze(Ut,-1))
    return np.array(newTrajectory),np.array(newControl)

def Solve(model,variations,traj0,ctrl0,N):
    trajPrev = traj0
    ctrlPrev = ctrl0
    traj = [trajPrev]
    ctrl = [ctrlPrev]
    Vs = []
    DeltaV = []
    for i in range(N):
        minControlParam = BackwardPass(variations,trajPrev,ctrlPrev)
        trajNext,ctrlNext = ForwardPass(model,minControlParam,trajPrev,ctrlPrev)
        trajPrev = trajNext
        ctrlPrev = ctrlNext
        traj.append(trajNext)
        ctrl.append(ctrlNext)
        Vs.append(minControlParam['V'])
        DeltaV.append(minControlParam['deltaV'])
        print(minControlParam['V'][0])
    return traj,ctrl,Vs,DeltaV