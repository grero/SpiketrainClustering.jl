module SpiketrainClustering
using LinearAlgebra

gaussian_kernel(t1::Real, t2::Real, τ) = exp(-(t1-t2)^2/(2τ^2))
spike_kernel(t1::Real, t2::Real, τ) = exp(-abs(t1-t2)/τ)

function gaussian_kernel(t1::Vector{T}, t2::Vector{T}, τ) where T <: Real
    n1 = length(t1)
    n2 = length(t2)
    s = 0.0
    for i in 1:n1
        for j in 1:n2
            s += gaussian_kernel(t1[i],t2[j],τ)
        end
    end
    s /= n1*n2
    s
end

function schoenberg_kernel(t1::Vector{T}, t2::Vector{T}, τ,σ) where T <: Real
    σ² = σ^2
    n1 = length(t1)
    n2 = length(t2)
    k11 = 0.0
    k12 = 0.0
    k22 = 0.0
    for i in 1:n1
        for j in 1:n2
            k11 += spike_kernel(t1[i],t1[j],τ)
            k22 += spike_kernel(t2[i],t2[j],τ)
            k12 += spike_kernel(t1[i],t2[j],τ)
        end
    end
    exp(-(k11 - 2*k12 + k22)/σ²)
end

function posterior(K::Matrix{Float64}, y::Vector{Float64}, σn)
    σn² = σn^2
    K*inv(K+σn²*I)*y
end

function regression_loss(σ::Real, τ::Real,σn::Real, spiketrains::Vector{Vector{Float64}}, y::Vector{Float64})
    n = size(y,1)
    #K = fill(0.0, n,n)
    #for i in 1:n
    #    for j in 1:n
    #        K[j,i] = schoenberg_kernel(spiketrains[i], spiketrains[j], τ,σ)
    #    end
    #end
    K = [schoenberg_kernel(spiketrains[i], spiketrains[j], τ, σ) for i in 1:n, j in 1:n]
    sum(posterior(K,y, σn))
end

#TODO: Maximize regression loss i.e. posterior

include("regression.jl")
end # module SpiketrainClustering
