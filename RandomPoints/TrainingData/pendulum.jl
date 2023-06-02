#module pendulum

using DifferentialEquations
using DiffEqPhysics
using Plots

# Pendulum equations
function Pendulum!(du, u, t)
    du[1] = u[2]
    du[2] = -sin(u[1])
end

function pendulum_H(p, q, params)
    a = params
    H = a*p^2/2 -cos(q)
    return H
end

function pendulum_H_multi(p, q)
    H = p.^2/2 -cos.(q)
    return H
end


function pend_sol(q, p, tau, M, Tend)
    y = [q1, p1]
    tm = LinRange(0, Tend, M+1)
    tspan = (0, Tend)
    prob = ODEProblem(Pendulum!, y0, tspan)
    sol = solve(prob, VCABM(), reltol=1e-12, abstol=1e-16, saveat=tm)

    return 1
end

q0 = 0.8
p0 = 0.5
tau = 1
tau_smol = 0.01
params = 1
M = 1000
Tend = tau*M
tm = LinRange(0, Tend, M+1)
tspan = (0, Tend)

prob = HamiltonianProblem(pendulum_H, p0, q0, tspan, params)
sol = solve(prob, KahanLi8(), reltol=1e-20, abstol=1e-20, dt = tau_smol, saveat=tm)

plot(sol[1, :], sol[2, :])
energy = pendulum_H_multi(sol[1, :], sol[2, :])
energy0 = energy[1]
energy_error = abs.(energy.-energy0)
plot(tm, energy_error)


