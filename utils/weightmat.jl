"""
This file is for creating the structure of Weight matrix W in the paper by using Flux.
This way, we may be able to use NN in the future work.
"""

#import Flux: functor, @functor
#using ForwardDiff, Zygote

"""
PredictionMat(in::Integer, out::Integer)

Creates a Matrix with parameter `W`.

y = W * x

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.

```julia
julia> d = PredictionMat(5, 2)
PredictionMat(5, 2)

julia> d(rand(5))
Tracked 2-element Array{Float32,1}:
0.47773355f0
-0.8844353f0 
"""
struct PredictionMat{S}
    W::S
end

"""
glorot_uniform borrowed from Flux.jl https://github.com/FluxML/Flux.jl/blob/master/src/utils.jl

duplicated to avoid adding Flux package to this project
"""
nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) # In case of convolution kernels
glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(nfan(dims...)))


function PredictionMat(in::Integer, out::Integer; initW = glorot_uniform)
    return PredictionMat(initW(out, in))
end

#@functor PredictionMat

function (a::PredictionMat)(x::AbstractArray)
    W = a.W
    W*x
end

function Base.show(io::IO, l::PredictionMat)
    print(io, "PredictionMat(", size(l.W, 2), ", ", size(l.W, 1))
    print(io, ")")
end

(a::PredictionMat{W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
invoke(a, Tuple{AbstractArray}, x)

(a::PredictionMat{W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
a(T.(x))


