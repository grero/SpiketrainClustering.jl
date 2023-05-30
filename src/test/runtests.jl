using SpiketrainClustering
using KernelFunctions
using Flux
using Flux: Optimise, Zygote
using Distributions
using Random
using LinearAlgebra
using Test


@testset "Basic" begin
    n = 5 # five spikes
    nt = 100
    g1 = Gamma(0.3, 0.5)
    sp1 = [cumsum(rand(g1, 5)) for i in 1:nt]
    y = [sp[4] for sp in sp1] + 0.05randn(nt)

    ps = [1.0, 1.0, 1.0]
    params, kernelc = Flux.destructure(SpiketrainClustering.SchoenbergKernel([1.0], [1.0]))

    function f(x, x_train, y_train, ps)
        k = kernelc(ps[1:2])
        return kernelmatrix(k, x, x_train) * ((kernelmatrix(k, x_train) + (ps[3]) * I) \ y_train)
    end 

    function loss(θ)
        ŷ = f(sp1, sp1, y, exp.(θ))
        return norm(y - ŷ) + exp(θ[3]) * norm(ŷ)
    end

    # initial loss
    L = fill(0.0,31)
    L[1] = loss(ps)
    opt = Optimise.ADAGrad(0.5)
    for i in 1:length(L)-1
        grads = only((Zygote.gradient(loss, ps)))
        Optimise.update!(opt, ps, grads)
        L[i+1] = loss(ps)
    end
    @show L
end


@testset "Basic2" begin
    n = 5 # five spikes
    nt = 100
    g1 = Gamma(0.3, 0.5)
    sp1 = [cumsum(rand(g1, 5)) for i in 1:nt]
    y = [sp[4] for sp in sp1] + 0.05randn(nt)

    ps = [1.0, 1.0, 1.0]
    params, kernelc = Flux.destructure(SpiketrainClustering.SchoenbergKernel2(SpiketrainClustering.SpikeKernel([1.0]), [1.0]))
    function f(x, x_train, y_train, ps)
        k = kernelc(ps[1:2])
        return kernelmatrix(k, x, x_train) * ((kernelmatrix(k, x_train) + (ps[3]) * I) \ y_train)
    end 

    function loss(θ)
        ŷ = f(sp1, sp1, y, exp.(θ))
        return norm(y - ŷ) + exp(θ[3]) * norm(ŷ)
    end

   @time grads = only((Zygote.gradient(loss, ps)))
   # initial loss
   L = fill(0.0,31)
   L[1] = loss(ps)
   opt = Optimise.ADAGrad(0.5)
   for i in 1:length(L)-1
       grads = only((Zygote.gradient(loss, ps)))
       Optimise.update!(opt, ps, grads)
       L[i+1] = loss(ps)
   end
   @show L
end

@testset "Product kernel" begin
    ps = [1.0, 1.0, 1.0]
    params, kernelc = Flux.destructure(SpiketrainClustering.SchoenbergKernel2(SpiketrainClustering.SpikeKernel([1.0]), [1.0])) 
    kernel1 = kernelc(ps[1:2])
    kernel2 = kernelc(ps[1:2])
    pkernel = SpiketrainClustering.ProductKernel([kernel1, kernel2])
    
    # two trials, each containing spike trains for two cells
    sp = [[[0.1, 0.3, 0.4],[0.3, 0.5]], [[0.2,0.4],[0.5,0.7]]]
    q = pkernel(sp[1], sp[2])
    @test q ≈ 0.18684577370816283
end