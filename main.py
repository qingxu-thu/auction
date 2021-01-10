import numpy as np
import math
#q;v;mu;lda

def N(q,v):
    return g*q-a_3/3-a_1*v*v*q+b*v*q+1/2*q*q*v

def N_v(q,v):
    return (-2*a_1*q*v+v*q)

def inverse_NI(mu,v,rho):
    delta = v*v+4*a_3*(g-a*v*v+b*v+2/rho*a*v-1/rho*b-mu)
    if delta>0:
        x = (-v+math.sqrt(delta))/(2*a_3)
        if x<0:
            print("no solution")
            return -1000
        else:
            return x
    else:
        print("no solution")
        return -1000

def I_cal(q,v,alpha,lda,rho):
    if alpha<lda:
        I = N(q,v)-N_v(q,v)/rho-alpha*q
    else:
        I = N(q,v)-N_v(q,v)/rho
    return I

def inverse_I(mu,alpha,lda,v,rho):
    if alpha<lda:
        return inverse_NI(mu+alpha,v,rho)
    else:
        return inverse_NI(mu,v,rho)

def allocation(N,lda,mu_list,alpha_list,v_list,B_list,q_0,rho_list):
    for itr,mu in enumerate(mu_list):
        q = []
        q_acc = 0
        demand_acc = 0
        for i in range(N):
            if inverse_I==-1000:
                x = 0
            else:
                x = inverse_I(mu,alpha_list[i],lda,v_list[i],rho_list[i])
            q.append(x)
            q_acc += q[-1]
        if q_acc>q_0:
            break
    for i in range(N):
        if alpha_list[i]<lda:
            demand_acc = q[i]+B_list[i]
    return demand_acc, q, mu, q_acc


def demand_satisfy(N,lda_list,mu_list,alpha_list,v_list,B_list,q_0,d,rho_list,vmax,vmin,flag):
    for lda in lda_list:
        v_list = rounding(N,alpha_list,lda,flag,v_list,interval,vmax,vmin):
        demand, q, mu, q_acc = allocation(N,lda,mu_list,alpha_list,v_list,B_list,q_0,rho_list)
        if demand>d:
            R = revenue_cal(N,q,alpha_list,v_list,lda,rho_list)
            return q,lda,R

def revenue_cal(N,q,alpha_list,v_list,lda,rho_list):
    I = 0
    for i in range(N):
        I+=I_cal(q[i],v_list[i],alpha_list[i],lda,rho_list[i]) 
    return I

def rounding(N,alpha_list,lda,flag,v_list,interval,vmax,vmin):
    if flag==0:
        return v_list
    else:
        v = [None]*N
        for i in range(N):
            if alpha_list[i]<lda:
                v[i] = (int((v_list[i]-vmin[i])/interval)+1)*interval
            else:
                v[i] = v_max-(int((v_max[i]-v_list[i])/interval)+1)*interval
        return v

def rho_cal(N,vmax,vmin,v_list):
    rho = [None]*N
    for i in range(N):
        rho[i] = (1/((v_max[i]-v_min[i])))/[(v_max[i]-v_list[i])/(v_max[i]-v_min[i])]
    return rho

if if __name__ == "__main__":
    q,lda,R = demand_satisfy()