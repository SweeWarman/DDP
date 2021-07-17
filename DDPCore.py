import numpy as np
from matplotlib import pyplot as plt

def BackwardPass(variations,trajectory,ctrl,mu=10):
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
        Fux = variations['Fux'](trajElem,ctrlElem)
        V = L + V
        if Vx is None:
            Vx = Lx
            Vxx = Lxx
        Qx = Lx + np.dot(Fx.T,Vx)
        Qu = Lu + np.dot(Fu.T,Vx) 
        Qxx = Lxx + np.dot(np.dot(Fx.T,Vxx),Fx)
        Quu = Luu + np.dot(np.dot(Fu.T,Vxx),Fu)
        Qux = Lux + np.dot(np.dot(Fu.T,Vxx),Fx)
        for i in range(numStates):
            Qxx += Fxx[i]*Vx[i,0]
            Quu += Fuu[i]*Vx[i,0]
            Qux += Fux[i]*Vx[i,0]
        #Quu += mu*np.eye(Quu.shape[0])

        Quu_p = Luu + np.dot(np.dot(Fu.T,Vxx + mu*np.eye(Vxx.shape[0])),Fu)
        Qux_p = Lux + np.dot(np.dot(Fu.T,Vxx + mu*np.eye(Vxx.shape[0])),Fx)
        for i in range(numStates):
            Quu_p += Fuu[i]*Vx[i,0]
            Qux_p += Fux[i]*Vx[i,0]

        # If Quu_p is not positive definite, increase mu and try again.
        if np.min(np.linalg.eigvals(Quu_p)) <= 0:
            return BackwardPass(variations,trajectory,ctrl,mu*10)

        k = -np.dot(np.linalg.inv(Quu_p),Qu)
        K = -np.dot(np.linalg.inv(Quu_p),Qux_p)
 
        #Vx = Qx - np.dot(np.dot(Qux.T,np.linalg.inv(Quu)),Qu)
        #Vxx = Qxx - np.dot(np.dot(Qux.T,np.linalg.inv(Quu)),Qux)
        #deltaV = np.squeeze(-0.5*np.dot(np.dot(Qu.T,np.linalg.inv(Quu)),Qu))
        deltaV = [np.squeeze(0.5*np.dot(np.dot(k.T,Quu),k)) , np.squeeze(np.dot(k.T,Qu))]
        Vx  = Qx + np.dot(np.dot(K.T,Quu),k) + np.dot(K.T,Qu) + np.dot(Qux.T,k)
        Vxx = Qxx + np.dot(np.dot(K.T,Quu),K) + np.dot(K.T,Qux) + np.dot(Qux.T,K)
        
        minControlParam['k'].insert(0,k)
        minControlParam['K'].insert(0,K)
        minControlParam['deltaV'].insert(0,deltaV)
        minControlParam['V'].insert(0,V)
    return minControlParam

def ForwardPass(model,minControlParam,trajectory,control,alpha=1,L=None):
    newTrajectory = []
    newControl = []
    newTrajectory.append(trajectory[0,:])
    N = trajectory.shape[0]
    k = minControlParam['k']
    K = minControlParam['K']
    numStates = trajectory.shape[1]
    numControls = control.shape[1]
    Xt = trajectory[0,:].reshape(numStates,1)
    StageCost = []
    for i in range(N-1):
        Xprev  = trajectory[i,:].reshape(numStates,1)
        Ut = control[i].reshape(numControls,1) + alpha*k[i] + np.dot(K[i],(Xt - Xprev))
        if L is not None:
            StageCost.append(L(np.squeeze(Xt),np.squeeze(Ut)))
        Xt = model(Xt,Ut)
        newTrajectory.append(np.squeeze(Xt,-1))
        newControl.append(np.squeeze(Ut,-1))
    return np.array(newTrajectory),np.array(newControl),StageCost

def Solve(model,variations,traj0,ctrl0,N,alpha=1):
    trajPrev = traj0
    ctrlPrev = ctrl0
    traj = [trajPrev]
    ctrl = [ctrlPrev]
    Vs = []
    DeltaV = []
    convergence = False
    while not convergence: 
        FPAccepted = False
        minControlParam = BackwardPass(variations,trajPrev,ctrlPrev)
        alpha = 1
        while not FPAccepted:
            trajNext,ctrlNext,StageCost = ForwardPass(model,minControlParam,trajPrev,ctrlPrev,alpha,variations['L'])
            deltaV = np.array(minControlParam['deltaV'])
            deltaVAlpha = np.sum(alpha**2*deltaV[:,0] + alpha*deltaV[:,1])
            Vold = minControlParam['V'][1]
            Vnew = np.sum(StageCost[1:])
            if np.fabs(deltaVAlpha) > 1e-5:
                improvement = (Vnew - Vold)/deltaVAlpha
                if  improvement > 0:
                    FPAccepted = True
                else:
                    alpha = alpha*0.9
            elif np.fabs(deltaVAlpha) > 0:
                FPAccepted = True
            else:
                alpha = alpha*0.9
        if np.fabs(Vold - Vnew) < 1e-3:
            convergence = True
        trajPrev = trajNext
        ctrlPrev = ctrlNext
        traj.append(trajNext)
        ctrl.append(ctrlNext)
        Vs.append(minControlParam['V'])
        #DeltaV.append(minControlParam['deltaV'])
        print(minControlParam['V'][0])
        #convergence = True
    return traj,ctrl,Vs,DeltaV