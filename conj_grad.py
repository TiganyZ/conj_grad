import numpy as np




def x_d_eps(x1, pm, eps=1e-8):
    return x1 + pm * eps


def eval_grad(f, x1, f1=False, evaltype="forward"):

    if  f1 == False:
        f1  = f(x1)
    
    if   evaltype == "forward" :
        ##  Differentiation with forward finite difference met
        x2 = x_d_eps( x1, 1. )
        f2 = f(x2)
        g  = (f2 - f1) / eps

    elif evaltype == "backward":
        ##  Do differentiation with backward finite difference method
        x2 = x_d_eps( x1, -1. )
        f2 = f(x2)
        g  = (f1 - f2) / eps

    elif evaltype == "central" :
        ##  Do differentiation with central finite difference method        
        x2p = x_d_eps( x1,  1. )
        x2m = x_d_eps( x1, -1. )        
        f2p = f(x2p)
        f2m = f(x2m)
        g   = (f2p -  f2m) / (2 * eps)
        
    return g
       

def grad_f(f1, x1, f2, x2):
    """
    This calculates the gradient of the function. 
    f is a vector of function evaluations fi at the matrix of vector points x
    """
    if type(x1) == float or type(x2) == float:
        ##  If x1 or x2 is just a scalar
        return (f2 - f1) / (x2 - x1)
    else:
        if shape.x1 != shape.x2:
            print("The shapes of the input vectors are not equal!\n    Exiting...")
            raise ValueError

        print("\n GRAD_F \n Evaluating the gradient at point \n x0 = %.8f \n  "%( (x2 + x1)/2. ) )

        g_f     = np.zeros(x1.shape)
        deltax  =  x2 - x1
        df      =  f2 - f1

        for ind, dx in enumerate(deltax):
            g_f[ind] = df / dx

        return g_f


def line_search_wolfe(f0, a_max):
    ##  This is a line search algorithm to find the best step length a_k_best
    


def zoom(f, aj, alo=0., ahi=10.,
            f0   = False, df0   = False,
            f_aj = False, f_alo = False,
            c1   = 0.25,  c2    = 0.5,  eps=1e-8, maxit=100):
    """
    Interpolate (using quadratic, cubic, or bisection) to find
    a trial step length a_j between a_lo and a_hi
    """
    if  f0    == False:
        f0    =  f(0)
    if  df0   == False:
        df0   =  eval_grad( f,  0, f1=f0  )
    if  f_aj  == False:
        f_aj  =  f(aj)
    if  f_alo == False:
        f_alo =  f(alo)

    ast = False
    for i in range(maxit):
        cond = (f_aj  > f0  +  c1 * aj * df0) or ( f_aj > f_alo )

        if cond:
            ahi = aj
        else:
            df_aj = eval_grad(f,aj,f1=f_aj)
            if abs(df_aj) <= -c2 * df0:
                ast = aj
            if df_aj * (ahi - alo) >= 0.:
                ahi = alo
            alo = aj
        if ast != False:
            break
        if i == maxit - 1:
            print("Warning: Maximum number of iterations for finding step length has been exceeded.")
            print("Setting a_s to current a_j:\n    a_s = %s" %(aj))
            ast = aj
    return ast


    

def conj_grad(p0, g0, mx_it):

    ## h0 == g0 initially

    h = g0
    a_k = - ( np.dot(r_k, p_k) / np.dot(pk, ( np.dot(A, p_k) )))
 
    for i in range(mx_it):
        hnp1 =


        
    """
############     From BOP   ##########
            
            gg = sum(ftot(:,:nd)*ftot(:,:nd))

            if ((abs(ggold) < 1.0e-6_dp) .or. (mod(iter,5) == 0)) then
               gamma = 0.0_dp
            else
               gamma = gg/ggold
            endif
            
            if (.not. quiet) write(9,'("gamma = ",g12.5)') gamma
            ggold = gg
            
            do
               cg(:,:nd) = gamma * cg(:,:nd) + ftot(:,:nd) 
               newad(:,:nd) = ad(:,:nd)
               
               call safemin(newad,cg,ein,eout,status)

               if ((status /= 1) .or. abs(ggold) <= 1.0e-6_dp) exit
               
               ggold = 0.0_dp               
               gamma = 0.0_dp
               
            end do

"""
