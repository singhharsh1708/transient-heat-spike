# Minimal reproducer: InterpolatingAdjoint(checkpointing = true) re-solves its checkpoint intervals adaptively, so
# with a fixed-step forward solve (adaptive = false, dt) the gradient carries the re-solve's tolerance error, which
# does not shrink with dt: at the default tolerances it stalls near 0.7 %, and only tighter solve tolerances bring
# back first order. Without checkpointing the same adjoint converges at first order in dt at either setting.
# Same root cause as SciMLSensitivity#1705 (the fixed dt is not passed to the re-solve), where fixed-step-only
# methods raise an error; with ImplicitEuler there is no error.
#
#   julia --project=backends/julia repro/sciml_checkpoint_resolve.jl
#
# Problem: 1D heat equation u_t = p u_xx on (0, 1), u(0) = u(1) = 0, second-order finite differences, u0 = sin(pi x);
# loss = sum(u(T)^2). Gradient d loss / dp from Zygote with each sensealg, against a central difference of the same
# fixed-step forward solve.

using OrdinaryDiffEqSDIRK: ImplicitEuler
using OrdinaryDiffEqCore: ODEProblem, solve
using SciMLSensitivity, Zygote

const N = 50
const h = 1 / (N + 1)
heat!(du, u, p, t) = (for i in 1:N
    du[i] = p[1] * ((i > 1 ? u[i - 1] : 0.0) - 2u[i] + (i < N ? u[i + 1] : 0.0)) / h^2
end; nothing)
const T = 0.1
u0 = sin.(pi .* (1:N) .* h)
prob = ODEProblem(heat!, u0, (0.0, T), [1.0])

loss(p, dt, sensealg; tols...) = sum(abs2, solve(prob, ImplicitEuler(); p = p, dt = dt, adaptive = false,
    saveat = [T], save_start = false, sensealg = sensealg, tols...).u[end])

for (label, tols) in (("default tolerances", (;)), ("abstol = 1e-12, reltol = 1e-10", (abstol = 1e-12, reltol = 1e-10)))
    println("\n", label)
    println("steps   central FD        checkpointing=true     checkpointing=false")
    for steps in (100, 1000, 10000)
        dt = T / steps
        eps = 1e-6
        fd = (loss([1 + eps], dt, nothing; tols...) - loss([1 - eps], dt, nothing; tols...)) / (2eps)
        errs = map((true, false)) do ck
            sensealg = InterpolatingAdjoint(checkpointing = ck, autojacvec = ReverseDiffVJP())
            g = Zygote.gradient(p -> loss(p, dt, sensealg; tols...), [1.0])[1][1]
            abs(g / fd - 1)
        end
        println(rpad(steps, 8), rpad(round(fd; sigdigits = 10), 18),
            rpad("rel err $(round(errs[1]; sigdigits = 3))", 23), "rel err $(round(errs[2]; sigdigits = 3))")
    end
end
