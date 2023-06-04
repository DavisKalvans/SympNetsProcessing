import numpy as np
import copy
### Implements the eight order verlet symplectic intergator, using multiple trimple jump methods

# Triple jump coefficients (Hairier 4.4)
def gen_gamma(p):
    gamma = np.zeros(3)
    gamma[0] = 1/(2-2**(1/(p+1)))
    gamma[1] = -(2**(1/(p+1))) /(2-2**(1/(p+1)))
    gamma[2] = 1/(2-2**(1/(p+1)))

    return gamma

# Fourth order method (3 jumps)
def fourth_order(x0, tau, D, gamma, problem, extraParams):
    tau_half = tau/2

    # First time
    p = x0[D:(2*D)] - gamma[0]*tau_half*problem.H_p(x0[0:D], extraParams)
    Q = x0[0:D] +gamma[0]*tau*problem.H_q(p, extraParams)
    P = p - gamma[0]*tau_half*problem.H_p(Q, extraParams)

    # Second time
    p = P - gamma[1]*tau_half*problem.H_p(Q, extraParams)
    Q = Q + gamma[1]*tau*problem.H_q(p, extraParams)
    P = p - gamma[1]*tau_half*problem.H_p(Q, extraParams)

    # Third time
    p = P - gamma[2]*tau_half*problem.H_p(Q, extraParams)
    Q = Q + gamma[2]*tau*problem.H_q(p, extraParams)
    P = p - gamma[2]*tau_half*problem.H_p(Q, extraParams)

    return np.concatenate((Q, P))

# Sixth order method (9 jumps)
def sixth_order(x0, tau, D, gamma2, gamma4, problem, extraParams):
    # First time
    tau1 = tau*gamma4[0]
    x1 = fourth_order(x0, tau1, D, gamma2, problem, extraParams)

    #Second time
    tau2 = tau*gamma4[1]
    x2 = fourth_order(x1, tau2, D, gamma2, problem, extraParams)

    # Third time
    tau3 = tau*gamma4[2]
    x3 = fourth_order(x2, tau3, D, gamma2, problem, extraParams)

    return x3

# Eight order method (27 jumps)
def eight_order(x0, tau, D, gamma2, gamma4, gamma6, problem, extraParams):
    # First time
    tau1 = tau*gamma6[0]
    x1 = sixth_order(x0, tau1, D, gamma2, gamma4, problem, extraParams)

    tau2 = tau*gamma6[1]
    x2 = sixth_order(x1, tau2, D, gamma2, gamma4, problem, extraParams)

    tau3 = tau*gamma6[2]
    x3 = sixth_order(x2, tau3, D, gamma2, gamma4, problem, extraParams)

    return x3

# Returns one step result for arbitrary tau, using much smaller step sizes to guarantee accuraccy
def eight_orderPrecise(x0, tau, D, problem, extraParams):
    TAU = 0.01
    tau_new = copy.deepcopy(tau) # Need to do this, since otherwise it changes the outside variable tau
    #X = copy.deepcopy(x0)
    steps = 1
    gamma2 = gen_gamma(2)
    gamma4 = gen_gamma(4)
    gamma6 = gen_gamma(6)


    while tau_new > TAU: # Constantly divide timestep by 2 and multiple required steps by 2
        tau_new /= 2
        steps *= 2

    for i in range(steps):
        x0 = eight_order(x0, tau_new, D, gamma2, gamma4, gamma6, problem, extraParams)

    return x0
