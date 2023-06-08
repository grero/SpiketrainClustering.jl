using SpiketrainClustering
using SpiketrainClustering: Spiketrain, PopulationSpiketrain
using KernelFunctions
using Flux
using Flux: Optimise, Zygote
using Distributions
using Random
using LinearAlgebra
using Test

@testset "basic" begin
    x = Spiketrain([0.1, 0.3, 0.4])
    y = x .+ 0.01
    @test y.spikes ≈ [0.11, 0.31, 0.41]

    @test x[1] ≈ 0.1
    x[1] = 0.15
    @test x[1] ≈ 0.15

    y = Spiketrain([0.3, 0.5])
    xx = PopulationSpiketrain([x,y])

    @test xx[1] ≈ x
    @test xx[1,1] ≈  0.15
end

@testset "kernels" begin
    xx = PopulationSpiketrain([[0.1, 0.3],[0.2,0.4,0.45]])
    @test length.(xx) == [2,3]
    params,kernel = SpiketrainClustering.get_kernel_function([xx])
    K = kernel(params)  
    @test typeof(K) <: SpiketrainClustering.ProductKernel
end

@testset "mean" begin
    x = [Spiketrain([0.1, 0.3, 0.4]), Spiketrain([0.2,0.4])]
    y = Spiketrain.([[0.3, 0.5], [0.5,0.6, 0.7]])
    kernel = SpiketrainClustering.SchoenbergKernel(SpiketrainClustering.SpikeKernel([1.0]), [1.0])
    μ = mean(kernel, x, y)
    @test μ ≈ [0.15358134219644462, 0.37404874973981617]
end

@testset "Divergence" begin
    x = Spiketrain.([[0.1, 0.3, 0.4], [0.2,0.4]])
    y = Spiketrain.([[0.3, 0.5, 0.7], [0.5,0.7]])
    kernel = SpiketrainClustering.SchoenbergKernel(SpiketrainClustering.SpikeKernel([1.0]), [1.0])
    d = SpiketrainClustering.divergence(kernel, x, y)
    # TODO: Check that this actually is correct
    @test d ≈ 0.9772709154550234
end


@testset "Regression" begin
    n = 5 # five spikes
    nt = 100
    g1 = Gamma(0.3, 0.5)
    sp1 = [SpiketrainClustering.Spiketrain(cumsum(rand(g1, 5))) for i in 1:nt]
    y = [sp[4] for sp in sp1] + 0.05randn(nt)

    ps = [1.0, 1.0, 1.0]
    params, kernelc = Flux.destructure(SpiketrainClustering.SchoenbergKernel(SpiketrainClustering.SpikeKernel([1.0]), [1.0]))
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
   opt = Optimise.AdaGrad(0.5)
   for i in 1:length(L)-1
       grads = only((Zygote.gradient(loss, ps)))
       Optimise.update!(opt, ps, grads)
       L[i+1] = loss(ps)
   end
   @show L
end

@testset "Product kernel" begin
    ps = [1.0, 1.0, 1.0]
    params, kernelc = Flux.destructure(SpiketrainClustering.ProductKernel([SpiketrainClustering.SchoenbergKernel(SpiketrainClustering.SpikeKernel([1.0]), [1.0]),
                                                                              SpiketrainClustering.SchoenbergKernel(SpiketrainClustering.SpikeKernel([1.0]), [1.0])]))
    pkernel = kernelc([1.0, 1.0, 1.0, 1.0])
    
    # two trials, each containing spike trains for two cells
    sp = [[[0.1, 0.3, 0.4],[0.3, 0.5]], [[0.2,0.4],[0.5,0.7]]]
    q = pkernel(sp[1], sp[2])
    @test q ≈ 0.18684577370816283

    # construct kernel matrix over all trials
    K = kernelmatrix(pkernel, sp, sp)
    @test K ≈ [1.0 q;q 1.0]
end

@testset "Clustering" begin
    # Two sets of spike trains, elicited by different stimuli
    g1 = Gamma(0.3, 0.5)
    g2 = Gamma(1.5, 0.5)
    nt = 5
    sp1 = [cumsum(rand(g1, 5)) for i in 1:nt]
    sp2 = [cumsum(rand(g2, 5)) for i in 1:nt]
end