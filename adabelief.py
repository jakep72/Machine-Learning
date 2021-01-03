import math

#stepsize
alpha = .005
 
#exponential decay rates for moment estimates
beta1 = 0.9
beta2 = 0.999

eps = 1e-8

def func(x):
    return ((x**2)-(4*x)+4)

def grad_func(x):
    return (2*x-4)

theta0 = 0
mt = 0
vt = 0
t = 0

while (1):
    t+=1
    gt = grad_func(theta0)
    mt = (beta1*mt)+(1-beta1)*gt
    vt = (beta2*vt)+((1-beta2)*((gt-mt)**2))+eps
    mhat = mt/(1-(beta1**t))
    vhat = vt/(1-(beta2**t))
    theta_prior = theta0
    print(theta_prior)
    theta0 = theta0 - ((alpha*mhat)/((vhat**.5)+eps))
    if (theta_prior == theta0):
        break
