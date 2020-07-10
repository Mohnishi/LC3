"""
This file is for creating RFF structure and for fast evaluations at given inputs.
"""

using Random

# A set of random fourier functions
struct RandomFourierFunctions{T<:AbstractFloat}
	directions::Matrix{T} # random direction
    offsets::Vector{T} # random shift in [0,2*pi]
    sigma::T

    function RandomFourierFunctions{T}(bandwidth, input, num_functions) where T<:AbstractFloat
        w = randn(T, num_functions, input) / T(bandwidth)
        # draw random offsets from a uniform distribution in [-pi,pi]
        b = rand(T, (num_functions,)) * T(2pi) .- T(pi)
        new{T}(w, b, T(bandwidth))
    end
end

function (rff::RandomFourierFunctions)(x::AbstractArray)
    W, b = rff.directions, rff.offsets
	nfeat = size(W,1)
    sD = eltype(b)(sqrt(2/nfeat))
    sD .* cos.(W*x .+ b)
end
(rff::RandomFourierFunctions{T})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}} = invoke(rff, Tuple{AbstractArray}, x)
(rff::RandomFourierFunctions{T})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}} = rff(T.(x))

function (rff::RandomFourierFunctions{T})(z::AbstractArray{T, N},
                                          x::AbstractArray{T, N}) where {T <: Union{Float32,Float64}, N}
    nfeat = size(rff.directions,1)
    sD = T(sqrt(2/nfeat))
   
    W, b = rff.directions, rff.offsets
    mul!(z, W, x)
    z .+= b
    z .= sD .* cos.(z)
    z
end

