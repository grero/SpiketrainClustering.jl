module SpiketrainClustering
using LinearAlgebra
using KernelFunctions
using KernelFunctions: Distances
using Functors
using Zygote
using Zygote: ChainRules, Tangent
using StatsBase

struct Spiketrain <: AbstractVector{Float64}
    spikes::Vector{Float64}
end

Base.size(x::Spiketrain) = size(x.spikes)
Base.getindex(x::Spiketrain, inds::Vararg{Int,1}) = x.spikes[inds...]
Base.setindex!(x::Spiketrain,val, inds::Vararg{Int,1}) = x.spikes[inds...] = val

Base.BroadcastStyle(::Type{<:Spiketrain}) = Broadcast.ArrayStyle{Spiketrain}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Spiketrain}}, ::Type{ElType}) where ElType
    Spiketrain(similar(Array{ElType}, axes(bc)))
end

KernelFunctions.dim(x::Spiketrain) = 0 # always pass
KernelFunctions.dim(x::AbstractVector{Spiketrain}) = 0 # always pass

KernelFunctions.kernelmatrix(k::KernelFunctions.Kernel, x::Spiketrain, y::Spiketrain) = k.(x.spikes, permutedims(y.spikes))

function StatsBase.mean(k::KernelFunctions.Kernel, x, y)
    K = kernelmatrix(k, x,y)
    vec(mean(K,dims=2))
end

struct ProductKernel{T<:KernelFunctions.Kernel} <: KernelFunctions.Kernel
    kernels::Vector{T}
end

function (k::ProductKernel)(x::AbstractVector{T},y::AbstractVector{T})  where T <: Vector{T2} where T2 <: Real
    q = 1.0
    for (kk,_x,_y) in zip(k.kernels, x, y)
        q *= kk(_x,_y)
    end
    q
end

Functors.@functor ProductKernel
struct SpikeKernel{T<:Real} <: KernelFunctions.Kernel
    τ::Vector{T}
end

# this doesn't work with Zygote
#(k::SpikeKernel)(x,y) = sum(exp.(-KernelFunctions.pairwise(Distances.Cityblock(), x,y)./k.τ))
function (k::SpikeKernel)(x::AbstractVector{Float64},y::AbstractVector{Float64})
    τ = k.τ[1]
    sum(exp.(-abs.(broadcast(-, x, permutedims(y)))/τ))
end

function (k::SpikeKernel)(x::Spiketrain, y::Spiketrain)
    τ = k.τ[1]
    sum(exp.(-abs.(broadcast(-, x.spikes, permutedims(y.spikes)))/τ))
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

struct SchoenbergKernel{T<:Real, T2 <: KernelFunctions.Kernel} <: KernelFunctions.Kernel
    skernel::T2
    σ::Vector{T}
end

(k::SchoenbergKernel)(x,y) = exp(-(k.skernel(x,x) - 2*k.skernel(x,y) + k.skernel(y,y))/k.σ[1]^2)

Functors.@functor SchoenbergKernel

function divergence(k::KernelFunctions.Kernel, x, y)
    n = length(x)
    m = length(y)
    Kxx = kernelmatrix(k,x,x)
    Kyy = kernelmatrix(k,y,y)
    Kxy = kernelmatrix(k,x,y)
    (1/n^2)*sum(Kxx) + (1/m^2)*sum(Kyy) - (2/(m*n))*sum(Kxy)
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
include("kernelpca.jl")
end # module SpiketrainClustering
