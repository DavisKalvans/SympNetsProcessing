using DifferentialEquations
using Plots


# Harmonic oscillator equation
function HarmOsc!(du, u, p, t)
    omega = p
    du[1] = u[2]
    du[2] = -omega^2*u[1]
end


omega = 0.5
q1 = 0.3
p1 = 0.5
y0 = [q1, p1]
tau = 1
M = 1000
Tend = tau*M
tm = LinRange(0, Tend, M+1)
tspan = (0, Tend)

prob = ODEProblem(HarmOsc!, y0, tspan, omega)
sol = solve(prob, VCABM(), reltol=1e-12, abstol=1e-16, saveat=tm)

plot(sol[1, :], sol[2, :])
