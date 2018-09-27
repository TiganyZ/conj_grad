import numpy as np

def grad_f(f1, x1, f2, x2):
    """
    This calculates the gradient of the function. 
    f is a vector of function evaluations fi at the matrix of vector points x
    """
    if shape.x1 != shape.x2:
        print("The shapes of the input vectors are not equal!\n    Exiting...")
        raise ValueError

    print("\n GRAD_F \n Evaluating the gradient at point \n x0 = %s \n  "%( (x2 + x1)/2. ) )

    g_f     = np.zeros(x1.shape)
    deltax  =  x2 - x1
    df      =  f2 - f1
    
    for ind, dx in enumerate(deltax):
        g_f[ind] = df / dx

    return g_f



def zoom()
    """
    Interpolate (using quadratic, cubic, or bisection) to find
    a trial step length
    """

def conj_grad(p0, g0, mx_it):

    ## h0 == g0 initially

    h = g0

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
