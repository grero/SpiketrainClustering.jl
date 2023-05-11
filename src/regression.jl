using Zygote
using Flux
using Flux: Zygote, Optimise
using LinearAlgebra
using ProgressMeter

function do_regression(y::Vector{Float64}, sp::Vector{Vector{Float64}};niter=20)
    rule = Optimise.Adam()
    params = (σ=[1.0], τ=[1.0], σn=[1.0])
    opt_state = Flux.setup(rule, params)
    L = fill(0.0, niter+1)
    L[1] = regression_loss(params.σ[1], params.τ[1], params.σn[1], sp, y)
    prog = Progress(niter, dt=1.0)
    for i in 1:niter
        g =  gradient(params->-SpiketrainClustering.regression_loss(params.σ[1], params.τ[1], params.σn[1], sp, y), params)
        Flux.update!(opt_state, params, g[1])
        L[1+i] = regression_loss(params.σ[1], params.τ[1], params.σn[1], sp, y)
        next!(prog, showvalues=[(:posterior, L[i+1])])
    end
    params,L
end
