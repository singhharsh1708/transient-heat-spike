# Julia backend: the same Q1 FEM and backward Euler as spike/problem.py, time-stepped by OrdinaryDiffEq
# (ImplicitEuler on the mass-matrix ODE M dT/dt = -K(rho) T, fixed steps), gradients by SciMLSensitivity adjoints.
#
#   julia --project=backends/julia backends/julia/heat.jl --nx 32 --ny 16 --steps 1000 --strategy interp --out r.json
#
# Strategies (continuous adjoints, each solved by the same integrator with the same fixed step):
#   interp      InterpolatingAdjoint(checkpointing = true): forward states kept only at checkpoints
#   quadrature  QuadratureAdjoint: dense forward solution kept, parameter gradient by quadrature
#   gauss       GaussAdjoint: dense forward solution kept, parameter gradient accumulated in a callback
#   backsolve   BacksolveAdjoint: forward state re-integrated backwards with the adjoint (ill-posed for diffusion)
# All use EnzymeVJP for the vector-Jacobian products; the solver Jacobian is the analytic -K(rho).

using OrdinaryDiffEqSDIRK: ImplicitEuler
using OrdinaryDiffEqCore: ODEFunction, ODEProblem, solve
using SciMLSensitivity, Zygote, Enzyme, SparseArrays, LinearAlgebra, JSON3, Random

args = Dict{String, String}()
let i = 1
    while i <= length(ARGS)
        args[ARGS[i][3:end]] = i < length(ARGS) && !startswith(ARGS[i + 1], "--") ? ARGS[i + 1] : "true"
        i += startswith(get(ARGS, i + 1, "--"), "--") ? 1 : 2
    end
end
nx, ny, steps = parse(Int, get(args, "nx", "32")), parse(Int, get(args, "ny", "16")), parse(Int, get(args, "steps", "1000"))
strategy = get(args, "strategy", "interp")

const Lx, Ly, k0, lam_t_end = 2.0, 1.0, 1.0, 1.0
lam = k0 * (pi / (2Lx))^2
t_end = lam_t_end / lam
exact_J = Lx * Ly / 2 * exp(-2lam * t_end)
exact_dJ_uniform = -2lam * t_end * exact_J

# --- mesh and element matrices (as spike/problem.py) ---
hx, hy = Lx / nx, Ly / ny
kx = [2 -2 -1 1; -2 2 1 -1; -1 1 2 -2; 1 -1 -2 2]
kyy = [2 1 -1 -2; 1 2 -2 -1; -1 -2 2 1; -2 -1 1 2]
const KE = hy / (6hx) .* kx .+ hx / (6hy) .* kyy
const ME = hx * hy / 36 .* [4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4]
nid(ix, iy) = iy * (nx + 1) + ix + 1
cells = [(nid(ix, iy), nid(ix + 1, iy), nid(ix + 1, iy + 1), nid(ix, iy + 1)) for iy in 0:(ny - 1) for ix in 0:(nx - 1)]
xcoord = [ix * hx for iy in 0:ny for ix in 0:nx]
free = findall(>(0.5hx), xcoord)              # all nodes except x = 0
dof = zeros(Int, length(xcoord)); dof[free] .= 1:length(free)
nd, ncells = length(free), length(cells)
const CELLDOF = [ntuple(a -> dof[c[a]], 4) for c in cells]   # 0 = Dirichlet node

function assemble(vals)
    I, J, V = Int[], Int[], Float64[]
    for (e, cd) in enumerate(CELLDOF), a in 1:4, b in 1:4
        (cd[a] == 0 || cd[b] == 0) && continue
        push!(I, cd[a]); push!(J, cd[b]); push!(V, vals(e, a, b))
    end
    return sparse(I, J, V, nd, nd)
end
const M = assemble((e, a, b) -> ME[a, b])
const KPATTERN = assemble((e, a, b) -> 1.0)

struct HeatRHS end
function (::HeatRHS)(du, u, p, t)
    fill!(du, zero(eltype(du)))
    @inbounds for e in 1:length(CELLDOF)
        cd = CELLDOF[e]
        for a in 1:4
            cd[a] == 0 && continue
            s = zero(eltype(du))
            for b in 1:4
                cd[b] == 0 && continue
                s += KE[a, b] * u[cd[b]]
            end
            du[cd[a]] -= k0 * p[e] * s
        end
    end
    return nothing
end
function heat_jac!(Jm, u, p, t)     # analytic Jacobian -K(p) into the sparsity pattern
    fill!(nonzeros(Jm), 0.0)
    for (e, cd) in enumerate(CELLDOF), a in 1:4, b in 1:4
        (cd[a] == 0 || cd[b] == 0) && continue
        Jm[cd[a], cd[b]] -= k0 * p[e] * KE[a, b]
    end
    return nothing
end

u0 = sin.(pi .* xcoord[free] ./ (2Lx))
const DT = Ref(t_end / steps)
fun = ODEFunction{true}(HeatRHS(); mass_matrix = M, jac = heat_jac!, jac_prototype = copy(KPATTERN))
prob = ODEProblem(fun, u0, (0.0, t_end), ones(ncells))
SENSE = Dict(
    "interp" => InterpolatingAdjoint(checkpointing = true, autojacvec = EnzymeVJP()),
    "quadrature" => QuadratureAdjoint(autojacvec = EnzymeVJP()),
    "gauss" => GaussAdjoint(autojacvec = EnzymeVJP()),
    "backsolve" => BacksolveAdjoint(autojacvec = EnzymeVJP()),
)[strategy]

function J(p; sensealg = SENSE)
    sol = solve(prob, ImplicitEuler(); p = p, dt = DT[], adaptive = false, saveat = [t_end], save_start = false,
        sensealg = sensealg)
    u = sol.u[end]
    return dot(u, M * u)
end

res = Dict{String, Any}("backend" => "julia-ordinarydiffeq", "strategy" => strategy, "nx" => nx, "ny" => ny,
    "steps" => steps, "n_dofs" => nd, "exact_J" => exact_J, "exact_dJ_uniform" => exact_dJ_uniform)
try
    p = ones(ncells)
    # Warm-up at 10 steps compiles everything; its peak is the baseline, so peak_rss_mb - baseline_rss_mb is what
    # the measured run needed on top of a compiled process.
    DT[] = t_end / 10
    res["compile_s"] = @elapsed (get(args, "mode", "grad") == "forward" ? J(p) : Zygote.gradient(J, p))
    GC.gc()
    res["baseline_rss_mb"] = Sys.maxrss() / 2^20
    DT[] = t_end / steps
    if get(args, "mode", "grad") == "forward"
        res["first_s"] = @elapsed (res["J"] = J(p))
        res["warm_s"] = res["first_s"]
    else
        res["first_s"] = @elapsed (g = Zygote.gradient(J, p)[1])
        res["warm_s"] = res["first_s"]
        res["J"] = J(p)
        res["dJ_uniform"] = sum(g)
        if haskey(args, "fd")
            d = randn(MersenneTwister(0), ncells); d ./= norm(d)
            eps = 1e-4
            res["fd_dir"] = (J(p .+ eps .* d) - J(p .- eps .* d)) / (2eps)
            res["grad_dir"] = dot(g, d)
        end
    end
    res["ok"] = true
catch e
    res["ok"] = false
    res["error"] = first(sprint(showerror, e), 500)
end
res["peak_rss_mb"] = Sys.maxrss() / 2^20
println(JSON3.write(res))
haskey(args, "out") && open(io -> JSON3.write(io, res), args["out"], "w")
