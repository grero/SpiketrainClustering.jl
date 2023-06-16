using Zygote
using ForwardDiff
using Flux
using Flux: Zygote, Optimise
using LinearAlgebra
using ProgressMeter
using Random

function kernel_creator(ps)
    return SchoenbergKernel(ps[1], SpikeKernel(ps[2]))
end

function predict(x_test, x_train, y_train, ps)
    pps = exp.(ps) 
    k = kernel_creator(pps[1:2])
    return kernelmatrix(k, x_test, x_train)*((kernelmatrix(k, x_train)+pps[3]*I)\y_train)
end

"""
Get the appropriate kernel reconstruction function
"""
get_kernel_function(sp::Any) = error("Not implemented")

"""
A single kernel for trials of spiketrains from a single neuron
"""
function get_kernel_function(sp::Vector{Vector{Float64}})
    params, kernelc = Flux.destructure(SpiketrainClustering.SchoenbergKernel(SpiketrainClustering.SpikeKernel([1.0]), [1.0]))
end

function get_kernel_function(sp::Vector{Spiketrain})
    params, kernelc = Flux.destructure(SpiketrainClustering.SchoenbergKernel(SpiketrainClustering.SpikeKernel([1.0]), [1.0]))
end

"""
Product kernel for trials of spiketrains from multiple neurons
"""
function get_kernel_function(sp::Vector{PopulationSpiketrain{N}}) where N
    # first make sure that each trial has the same number of neurons
    nn = length.(sp)
    all(nn .== nn[1]) || error("All trials should have the same number of neurons")
    n = nn[1]
    params, kernelc = Flux.destructure(SpiketrainClustering.ProductKernel(([SpiketrainClustering.SchoenbergKernel(SpiketrainClustering.SpikeKernel([1.0]), [1.0]) for k in 1:n]...,)))
end

"""
```
function do_regression(y::Vector{Float64}, sp::Vector{Vector{Float64}};niter=20, opt=Optimise.Adam(),rel_tol=sqrt(eps(Float64)))
```
Regress the values in `y` using the spike train in `sp`.

The kernel regression is carried out using Schoenberg outer kernel and a spike inner kernel, i.e.

\$\\kappa(x,y) = \\exp\\left( -\\frac{1}{\\sigma}(\\kappa'(x,x) - 2\\kappa'(x,y) + \\kappa'(y,y))\\right)\$,

with \$\\kappa'\$ given by

\$\\kappa'(x,y) = sum_{i,j}\\exp\\left( -\\frac{1}{\\tau} | x_i - y_i| \\right)\$.

"""
function do_regression(y::Vector{Float64}, sp;niter=20, opt=Optimise.Adam(),rel_tol=sqrt(eps(Float64)), r_train=0.8,p0::Union{Nothing, Vector{T}}=nothing) where T<: Real
    # prep the kernel
    params,kernelc = get_kernel_function(sp)

    nparams = length(params)

    #separate into training and testing
    nt = length(sp)
    ntrain = round(Int64, r_train*nt)
    ntest = nt - ntrain
    sp_train = sp[1:ntrain]
    sp_test = sp[ntrain+1:end]
    ytrain = y[1:ntrain]
    ytest = y[ntrain+1:end]
    function f(x, x_train, y_train, ps)
        k = kernelc(ps[1:nparams])
        return kernelmatrix(k, x, x_train) * ((kernelmatrix(k, x_train) + (ps[nparams+1]) * I) \ y_train)
    end 

    function loss(θ)
        ŷ = f(sp_test, sp_train, ytrain, exp.(θ))
        return norm(ytest - ŷ) + exp(θ[end]) * norm(ŷ)
    end

    if p0 === nothing
        ps = [params;1.0]
    else
        ps = p0[1:nparams+1]
    end

    L = fill(NaN, niter+1)
    L[1] = loss(ps)
    prog = ProgressThresh(rel_tol, dt=1.0,showspeed=true)
    stop_i = niter+1
    for i in 1:niter
        grads = ForwardDiff.gradient(loss, ps)
        Optimise.update!(opt, ps, grads)
        L[1+i] = loss(ps)
        rr = abs(L[1+i]-L[i])/L[i] 
        if  rr <= rel_tol
            stop_i = i+1
            finish!(prog)
            break
        end
        ProgressMeter.update!(prog, rr, showvalues=[(:iter, i),(:loss, L[1+i]),(:rel_loss, (L[1]-L[1+i])/L[1])])
    end
    kernelc(exp.(ps[1:nparams])), f(sp,sp,y,exp.(ps)), L[1:stop_i]
end
