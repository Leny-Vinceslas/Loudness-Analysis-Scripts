import numpy as np 
from scipy.optimize import least_squares

#alternative fitting method ---> Optimized loudness-function estimation for categorical loudness scaling data

def polyfit_with_fixed_points(n, x, y, xf, yf) :
            mat = np.empty((n + 1 + len(xf),) * 2)
            vec = np.empty((n + 1 + len(xf),))
            x_n = x**np.arange(2 * n + 1)[:, None]
            yx_n = np.sum(x_n[:n + 1] * y, axis=1)
            x_n = np.sum(x_n, axis=1)
            idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
            mat[:n + 1, :n + 1] = np.take(x_n, idx)
            xf_n = xf**np.arange(n + 1)[:, None]
            mat[:n + 1, n + 1:] = xf_n / 2
            mat[n + 1:, :n + 1] = xf_n.T
            mat[n + 1:, n + 1:] = 0
            vec[:n + 1] = yx_n
            vec[n + 1:] = yf
            params = np.linalg.solve(mat, vec)
            return params[:n + 1]

def delta_fit(F_L,Ri_L):
    delta=np.zeros(len(F_L))
    for i,F_Li in enumerate(F_L):
        if F_Li < 0 or Ri_L[i] < 0 :
            delta[i]=0
        elif F_Li > 50 or Ri_L[i] == 50:
            delta[i]=50
        else:
            delta[i]= np.abs(Ri_L[i] - F_Li)
    return delta
        
def linearFunc(x,a,b):
    return a*x+b
    
def CUforGivenL(a,L_cut,CU_cut,y):
    #y=a*x+b
    b=CU_cut-(a*L_cut)
    return (y-b)/a


def x2y_lin(x, x0, y0, m):
    return y0 + m*(x-x0)

def y2x_lin(y, x0, y0, m):
    return (y-y0)/m + x0

#try fitting quadratic function as well  
     
# def loudnessFunc(m_l,m_h,L_cut,L,Ri_L):
def loudnessFunc(x,L,Ri_L):
    m_l=x[0]
    m_h=x[1]
    L_cut=x[2] 

    L_15=y2x_lin(15, L_cut, 25, m_l)
    L_35=y2x_lin(35, L_cut, 25, m_h)

    F_L=np.zeros(len(L))
    for i,Li in enumerate(L):
        if Li<=L_15:
            F_L[i]=25+m_l*(Li-L_cut)
        if Li>=L_35:
            F_L[i]=25+m_h*(Li-L_cut)
    for i,Li in enumerate(L):
        if L_15<Li<L_35:
            F_L[i]=bezSmoothing(Li,L_cut,L_15,L_35)
            # F_L[i]=0

    # if less than 4 point >35 the compute from UCL Pascoe / HTL
    L_50=y2x_lin(50, L_cut, 25, m_h)
    d=np.sqrt((50 - 50)**2 + (L_50 - 100)**2) #constrain on the last data point which has to be [100 50]

    return delta_fit(F_L,Ri_L) + d*.013 #+(m_l-(m_h))*0.05 #try square the error


    
def bezSmoothingInv(F,L_cut,L_15,L_35):
    y=F
    x0, y0 = 15, L_15
    x1 = 2 * L_cut - 2 * L_15
    y1 = 2 * 25 - 2 * 15
    x2 = L_15 - 2 * L_cut + L_35
    y2 = 15 - 2 * 2 * 25 + 35 
    t = y / y1 - y0/ y1
    
    return x2 * ( t + x1 / (2 * x2))**2 - ((x1**2) / (4 * 2**2)) + x0
    
def bezSmoothing(L,L_cut,L_15,L_35):
    x0, y0 = L_15, 15
    x1 = 2 * L_cut - 2 * L_15
    y1 = 2 * 25 - 2 * 15
    x2 = L_15 - 2 * L_cut + L_35
    y2 = 15 - 2 * 25 + 35 
    xk, yk = x1, y1
    xa, ya = x0, y0
    xb, yb = x2, y2
    if ((yk-ya)/(xk-xa)) < ((yk-yb)/(xk-xb)):
        t = -( x1 / ( 2*x2 )) - np.sqrt((( L - x0 ) / x2) + ( x1**2 / ( 4 * x2**2 )))
        # print('1')
    elif ((yk-ya)/(xk-xa)) > ((yk-yb)/(xk-xb)):
        t = -( x1 / ( 2*x2 )) + np.sqrt(( L - x0 ) / x2 + ( x1**2 / ( 4 * x2**2 )))
        # print('2')
    
    return y0 + y1 * t + y2 * t**2

class Pwlf_bez:
    
    def __init__(self):

        self.block=[]
        self.maxCU=50
        self.maxLevel=100
        self.minCU=0
        self.minLevel=0
        
        
    def __str__(self):
        return 'piecewise linear fitting'
        
    def Pwlf_bez_BrandT_true(self,x,y):
        
        x0= np.array([0.4, 0.6,80])
        bounds=([0.2, 0.2,60], [5, 5 ,np.inf])
    
        res_lsq = least_squares(loudnessFunc, x0, args=(x,y),bounds=bounds)
        
        return res_lsq.x,res_lsq

    