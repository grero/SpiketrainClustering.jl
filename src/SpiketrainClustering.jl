module SpiketrainClustering
using LinearAlgebra
using KernelFunctions
using KernelFunctions: Distances
using Functors
using Zygote
using Zygote: ChainRules, Tangent
using StatsBase
using StaticArrays

struct Spiketrain{T<:Real} <: AbstractVector{T}
    spikes::Vector{T}
end

Base.size(x::Spiketrain) = size(x.spikes)
Base.getindex(x::Spiketrain, inds::Vararg{Int,1}) = x.spikes[inds...]
Base.setindex!(x::Spiketrain,val, inds::Vararg{Int,1}) = x.spikes[inds...] = val

Base.BroadcastStyle(::Type{<:Spiketrain}) = Broadcast.ArrayStyle{Spiketrain}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Spiketrain}}, ::Type{ElType}) where ElType
    Spiketrain(similar(Array{ElType}, axes(bc)))
end

struct PopulationSpiketrain{N} <: AbstractVector{Spiketrain}
    spikes::SVector{N, Spiketrain}
end

get_ncells(x::PopulationSpiketrain{N}) where N = N

PopulationSpiketrain(x::Vector{Spiketrain}) = PopulationSpiketrain(SVector(x...))
PopulationSpiketrain(x::Vector{Vector{Float64}}) = PopulationSpiketrain(Spiketrain.(x))

Base.size(x::PopulationSpiketrain) = size(x.spikes)
Base.getindex(x::PopulationSpiketrain, ii) = x.spikes[ii]
Base.getindex(x::PopulationSpiketrain, ii,jj) = x.spikes[ii][jj]
Base.setindex!(x::PopulationSpiketrain,val, ii) = x.spikes[ii] = val
Base.setindex!(x::PopulationSpiketrain,val, ii,jj) = x.spikes[ii][jj] = val
Base.BroadcastStyle(::Type{<:PopulationSpiketrain}) = Broadcast.ArrayStyle{PopulationSpiketrain}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PopulationSpiketrain}}, ::Type{ElType}) where ElType
    Spiketrain(similar(Array{ElType}, axes(bc)))
end

KernelFunctions.dim(x::Spiketrain) = 0 # always pass
KernelFunctions.dim(x::AbstractVector{Spiketrain{T}}) where T <: Real = 0 # always pass

KernelFunctions.kernelmatrix(k::KernelFunctions.Kernel, x::Spiketrain, y::Spiketrain) = k.(x.spikes, permutedims(y.spikes))

function StatsBase.mean(k::KernelFunctions.Kernel, x, y)
    K = kernelmatrix(k, x,y)
    vec(mean(K,dims=2))
end

struct ProductKernel{T<:KernelFunctions.Kernel,N} <: KernelFunctions.Kernel
    kernels::NTuple{N,T}
end

function (k::ProductKernel{T,N})(x::PopulationSpiketrain{N},y::PopulationSpiketrain{N}) where N where T <: KernelFunctions.Kernel
    q = 1.0
    for i in 1:N
        q *= k.kernels[i](x[i], y[i])
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
    if isempty(x) || isempty(y)
        return 0.0
    end
    τ = k.τ[1]
    sum(exp.(-abs.(broadcast(-, x.spikes, permutedims(y.spikes)))/τ))
end

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

#TODO: Maximize regression loss i.e. posterior

include("regression.jl")
include("kernelpca.jl")
end # module SpiketrainClustering
