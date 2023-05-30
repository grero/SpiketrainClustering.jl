module SpiketrainClustering
using LinearAlgebra
using KernelFunctions
using KernelFunctions: Distances
using Functors
using Zygote
using Zygote: ChainRules, Tangent

#TODO: Implement product kernel to deal with multiple neurons
struct SpikeKernel{T<:Real} <: KernelFunctions.Kernel
    τ::Vector{T}
end

# this doesn't work with Zygote
#(k::SpikeKernel)(x,y) = sum(exp.(-KernelFunctions.pairwise(Distances.Cityblock(), x,y)./k.τ))
function (k::SpikeKernel)(x::AbstractVector{Float64},y::AbstractVector{Float64})
    τ = k.τ[1]
    sum(exp.(-abs.(broadcast(-, x, permutedims(y)))/τ))
end

function spike_kernel(x::AbstractVector{T}, y::AbstractVector{T},τ::Real) where T <: Real
    q = 0.0
    for _x in x
        for _y in y
            q += exp(-abs(_x - _y)/τ)
        end
    end
    q
end

function spike_kernel2(x::AbstractVector{T}, y::AbstractVector{T}, τ::Real) where T <: Real
    sum(exp.(-abs.(broadcast(-, x, permutedims(y)))/τ))
end
#function ChainRules.rrule(k::SpikeKernel, x, y)
#    output = k(x,y) 
#    function SpikeKernel_pullback(Δy)
#        q1 = 0.0
#        q2 = 0.0
#        q3 = 0.0
#        for _x in x
#            for _y in y
#                d = abs(_x-_y)
#                q1 += exp(-d/k.τ)*d
#                if _x > _y
#                    q2 += -exp(-d/k.τ)
#                    q3 += exp(-d/k.τ)
#                else
#                    q2 += exp(-d/k.τ)
#                    q3 += -exp(-d/k.τ)
#            end
#        end
#        q/k.τ
#    end
#        q1 /= k.τ^2
#        q2 /= k.τ
#        q3 /= k.τ
#        Tangent{SpikeKernel}(;τ=q1*Δy), q2*Δy, q3*Δy
#    end
#    output, SpikeKernel_pullback
#end

Functors.@functor SpikeKernel
struct SchoenbergKernel{T2<:Real} <: KernelFunctions.Kernel
    τ::Vector{T2}
    σ::Vector{T2}
end

(k::SchoenbergKernel)(x,y) = exp(-(spike_kernel2(x,x,k.τ[1]) - 2*spike_kernel2(x,y,k.τ[1]) + spike_kernel2(y,y,k.τ[1]))/k.σ[1]^2)

Functors.@functor SchoenbergKernel

struct SchoenbergCombinator{T<:Real}
    σ::Vector{T}
end

(k::SchoenbergCombinator)(xx,xy,yy) = exp(-(xx - 2xy + yy)/k.σ[1]^2)
#function Functors.functor(::Type{<:SchoenbergKernel}, x::SchoenbergKernel)
#    function reconstruct_schoenbergkernel(xs)
#        return SchoenbergKernel(xs.σ, xs)
#    end
#end

#struct KernelCombinator{T<}

struct SchoenbergKernel2{T<:Real, T2 <: KernelFunctions.Kernel} <: KernelFunctions.Kernel
    skernel::T2
    σ::Vector{T}
end

(k::SchoenbergKernel2)(x,y) = exp(-(k.skernel(x,x) - 2*k.skernel(x,y) + k.skernel(y,y))/k.σ[1]^2)

Functors.@functor SchoenbergKernel2

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
