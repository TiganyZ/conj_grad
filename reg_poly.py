import numpy as np
import random
import matplotlib.pyplot as plt

def append_vector(T, t_vec):
    if len(T) > 1:
        T = np.append(T, t_vec).reshape( (T.shape[0] + 1, T.shape[1]) )
    else:
        T = np.array([t_vec])
    return T  


def cost_function(t,y,a,lam=0.0):
    cf = 0.0
    if len(t) > 1:
        cf += np.sum( (t-y)**2 + lam * a**2  )
    else:
        cf += (t-y)**2 + lam * a**2
    return cf
    
def reg_polyf(xi, yi, n_deg, lam=0.0):
    X  = np.array([ xi**n for n in reversed(range(n_deg + 1))]).T
    print(X.shape, len(yi))
    a = np.linalg.inv( X.T.dot(X) + lam * np.eye(X.shape[1]) ).dot( X.T.dot(yi) )
    print("Regularised Polynomial Coefficients")
    for i in reversed(range(n_deg+ 1)):
        print("x^%s:    a_%s = %s " %(i, i, a[i]))
    return a

def poly(a, x):
    f = 0.0
    for i in reversed(range(len(a))):
        f += a[len(a) - i -1] * x**i
    return f
    
def test_f(x):
    n_deg = 6
    a = np.array([1,2,0.5,4,5,6,7])
    f = poly(a, x) +  np.random.normal(0, 2)
    return f


# x = np.linspace(-5, 5, 15)
# y = np.zeros(x.shape)
# n_deg = 6
# l = 0.0
# for i, xi in enumerate(x):
#     y[i] = test_f(xi)
# a  = reg_polyf(x, y, n_deg, lam=l)
# yp = poly(a, x)

# fig = plt.figure()
# ax  = fig.add_subplot(111)
# ax.plot(x, y, 'bo', label="Data ")
# ax.plot(x, yp, 'r--', label="Reg poly fit")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_title("Regularised polyfit")
# ax.legend()
# plt.show()
