from matplotlib import pyplot as plt
import numpy as np
import DDPCore as DDP

def Model(X,U,dt=0.1):
    Ax = np.array([[1.0,dt],[0.0,1.0]])
    Bx = np.array([[0.0,0.0],[dt,0.0]])
    Ay = Ax
    By = np.array([[0.0,0.0],[0.0,dt]])
    Au  = np.hstack([Ax,np.zeros((2,2))])
    Al  = np.hstack([np.zeros((2,2)),Ay])
    A   = np.vstack([Au,Al])
    B   = np.vstack([Bx,By])
    Xnew = np.dot(A,X.reshape(4,1)) + np.dot(B,U.reshape(2,1))
    return Xnew

def GenerateInitialTrajectory(X0):
    traj = [X0] 
    control = []
    X = np.array(X0).reshape(4,1)
    while (X[0]**2 + X[2]**2) > 1e-3:
        if X[0] > 3:
            U = np.array([-1.0,0.0])
        else:
            U = np.array([0.0,-1.0])
        U = U.reshape(2,1)
        Xnew = Model(X,U)
        traj.append(np.squeeze(Xnew,-1))
        control.append(np.squeeze(U,-1))
        X = Xnew
        if X[2] < 0:
            break
    return np.array(traj),np.array(control)
    

variations = {}
variations['L'] = lambda X,U: (X[0]**2 + X[1]**2 + X[2]**2 + X[3]**2 + U[0]**2 + U[1]**2)*0.5 
variations['Lx'] = lambda X,U: np.array([[X[0]],[X[1]],[X[2]],[X[3]]])
variations['Lxx'] = lambda X,U: np.eye(4)
variations['Lu'] = lambda X,U: np.array([[U[0]],[U[1]]])
variations['Luu'] = lambda X,U: np.eye(2)
variations['Lux'] = lambda X,U: np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
variations['Fx'] = lambda X,U:np.array([[1.0, 0.1, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.1],
                                        [0.0, 0.0, 0.0, 1.0]])
variations['Fu'] = lambda X,U:np.array([[0.0,0.0],
                                        [0.1,0.0],
                                        [0.0,0.0],
                                        [0.0,0.1]])
variations['Fxx'] = lambda X,U: [np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))] 
variations['Fuu'] = lambda X,U: [np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))] 

trajectory0,control0 = GenerateInitialTrajectory([10.0,5.0,10.0,0.0])
traj,ctrl,Vs,DeltaV = DDP.Solve(Model,variations,trajectory0,control0,150)

N = len(traj)
plt.figure(1)
for i in range(0,N,2):
    plt.plot(traj[i][:,0],traj[i][:,2])

plt.figure(2)
for i in range(N):
    plt.plot(np.arange(ctrl[i].shape[0]),ctrl[i])

plt.figure(3)
for i in range(N-1):
    plt.plot(np.arange(len(Vs[i])),Vs[i],'o-')

plt.figure(4)
plt.plot(np.arange(len(DeltaV[0])),DeltaV[0])
plt.show()