using Zygote
using Flux
using Flux: Zygote, Optimise
using LinearAlgebra
using ProgressMeter

function kernel_creator(ps)
    return SchoenbergKernel(ps[1], SpikeKernel(ps[2]))
end

function predict(x_test, x_train, y_train, ps)
    pps = exp.(ps) 
    k = kernel_creator(pps[1:2])
    return kernelmatrix(k, x_test, x_train)*((kernelmatrix(k, x_train)+pps[3]*I)\y_train)
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
function do_regression(y::Vector{Float64}, sp::Vector{Vector{Float64}};niter=20, opt=Optimise.Adam(),rel_tol=sqrt(eps(Float64)))
    # prep the kernel
    params, kernelc = Flux.destructure(SpiketrainClustering.SchoenbergKernel2(SpiketrainClustering.SpikeKernel([1.0]), [1.0]))

    function f(x, x_train, y_train, ps)
        k = kernelc(ps[1:2])
        return kernelmatrix(k, x, x_train) * ((kernelmatrix(k, x_train) + (ps[3]) * I) \ y_train)
    end 

    function loss(θ)
        ŷ = f(sp, sp, y, exp.(θ))
        return norm(y - ŷ) + exp(θ[3]) * norm(ŷ)
    end

    ps = [1.0, 1.0, 1.0]
    L = fill(NaN, niter+1)
    L[1] = loss(ps)
    prog = Progress(niter, dt=1.0)
    for i in 1:niter
        grads = only((Zygote.gradient(loss, ps)))
        Optimise.update!(opt, ps, grads)
        L[1+i] = loss(ps)
        if abs(L[1+i]-L[i])/L[i] <= rel_tol
            finish!(prog)
            break
        end
        next!(prog, showvalues=[(:posterior, L[i+1])])
    end
    kernelc(exp.(ps[1:2])), f(sp,sp,y,exp.(ps)), L
end
