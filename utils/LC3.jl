using Base.Iterators: partition
using ElasticArrays
using LinearAlgebra
using LyceumAI

# data buffer
struct DataBuffers{T<:AbstractFloat}
    phixudatabuf::ElasticArray{T}      # phi(x_t,u_t)
    xudatabuf::ElasticArray{T}         # x_t,u_t
    ydatabuf::ElasticArray{T}          # x_t+1
    xdatabuf::ElasticArray{T}          # x_t
    udatabuf::ElasticArray{T}          # u_t
    rewarddatabuf::ElasticArray{T}     # reward_t
    sizeofbuf::ElasticArray{Int}       # size of the buffer
    function DataBuffers{T}(ns::Int, no::Int, nu::Int, nf::Int) where T<:AbstractFloat
        new(ElasticArray{T}(undef,nf,0),
            ElasticArray{T}(undef,no+nu,0),
            ElasticArray{T}(undef,ns,0),
            ElasticArray{T}(undef,no,0),
            ElasticArray{T}(undef,nu,0),
            ElasticArray{T}(undef,1,0),
            ElasticArray{Int}(undef,1,0)
           )
    end
end

# LC3 structure
struct LC3{E, F, K, M}
    envsampler::E
    featurize::F
    PredictMat::K
    Hmax::Int      # maximum horizon
    N::Int         # number of trajectories
    ctrlrange::AbstractArray
    covW::AbstractArray
    Amat::AbstractArray
    covscale::AbstractFloat
    controller::M
    numfeatures::Int
    buffers::DataBuffers    # replay buffer sort of..
    function LC3(
                 env_tconstructor,
                 featurize,
                 PredictMat,
                 numfeatures,
                 ns,
                 no,
                 nu,
                 ctrlrange,
                 ridgereg,
                 covscale,
                 controller;

                 Hmax = 400, 
                 N = 400,
                 prebuffer = nothing,
                )

        # check errors
        0 < Hmax <= N || throw(ArgumentError("Hmax must be in interval (0, N]"))
        0 < N || throw(ArgumentError("N must be > 0"))
        # find common type of PredictionMat
        #KT = eltype(PredictMat.W)
        #DT = LyceumAI.promote_modeltype(PredictMat)
        #if !isconcretetype(DT)
        #    DTnew = Shapes.default_datatype(DT)
        #    @warn "Could not infer model element type. Defaulting to $DTnew"
        #    DT = DTnew
        #end
        envsampler = env_tconstructor 

        new{
            typeof(envsampler),
            typeof(featurize),
            typeof(PredictMat),
            typeof(controller)
           }(
             envsampler,
             featurize,
             PredictMat,
             Hmax,
             N,
             ctrlrange,
             Matrix(I(numfeatures)/ridgereg),
             zeros(numfeatures,no),
             covscale,
             controller,
             numfeatures,
             prebuffer == nothing ? DataBuffers{Float32}(ns,no,nu,numfeatures) : prebuffer,
            )
    end
end


#Get coherent data from multiple trajectory data
function getshifteddata(trajectory::AbstractArray)
    data1 = trajectory[:,1:end-1]
    data2 = trajectory[:,2:end]
    return (data1, data2)
end

#Clamp control inputs to the given range
function clampctrl!(actions::AbstractArray, ctrlrange::AbstractArray)
    nu = size(actions, 1)
    @inbounds for u=1:size(ctrlrange, 2)
        @simd for a=1:nu
            actions[u,a] = clamp(actions[u,a], ctrlrange[1,u], ctrlrange[2,u])
        end
    end
end

#Storing data and plot
function storedata!(batch::NamedTuple, buffers::DataBuffers, ctrlrange::AbstractArray, featurize)
    # Storing data into arrays : getshifteddata deals with multiple trajecotries
    xydata = getshifteddata(batch.observations)
    uvdata = getshifteddata(batch.actions)
    xdata  = first(xydata) 
    udata  = first(uvdata) 
    rewarddata = batch.rewards[1:end-1] 
    clampctrl!(udata, ctrlrange)
    xudata = vcat(xdata, udata)
    PhiXUdata = featurize(xudata)
    ydata  = last(xydata) - xdata 
    # save data into data buffers
    append!(buffers.phixudatabuf, PhiXUdata)
    append!(buffers.xudatabuf, xudata)
    append!(buffers.ydatabuf, ydata) 
    append!(buffers.xdatabuf, xdata) 
    append!(buffers.udatabuf, udata)
    append!(buffers.rewarddatabuf, rewarddata) 
    append!(buffers.sizeofbuf, [size(xdata,2)])  # data size 
    return xudata, ydata, PhiXUdata
end

#Iterator for PredictionMat Learning
function Base.iterate(featlearn::LC3{DT}, i = 1) where {DT}
    @unpack envsampler, featurize, PredictMat = featlearn
    @unpack Hmax, N, ctrlrange = featlearn
    @unpack covW, Amat, covscale, controller, numfeatures, buffers = featlearn

    if (i == 1) || (mod(i, 50) == 0); @info "Iterations:" i; end;

    #------------------Reset MPPI struct------------#
    reset!(controller)
    randreset!(envsampler)

    #-------------------Roll out--------------------#
    elapsed_sample = @elapsed begin
        opt = ControllerIterator((action, state, obs) -> getaction!(action, obs, controller),
                                 envsampler; T = Hmax, plotiter = 9999)
        for _ in opt # runs iterator
        end
        batch = opt.trajectory 
    end


    #-------------------Store data------------------#
    newxu, newy, newPhiXU = storedata!(batch, buffers, ctrlrange, featurize)
    if (i == 1) || (mod(i, 50) == 0); @info "rewards" sum(batch.rewards); end;

    sizeofbuf = sum(buffers.sizeofbuf)


    #------------------Weight matrix----------------#
    # matrix inversion lemma / update covariance and compute mean model
    covW .= covW .- covW * newPhiXU * pinv(I+newPhiXU'*covW*newPhiXU) * newPhiXU' * covW
    Amat .+= newPhiXU * newy'
    meanW = covW * Amat
    
    prederr = norm(newy - meanW' * newPhiXU)/(norm(meanW' * newPhiXU) + 1e-6)  
    #@info "Mean-model error" prederr

    # when covscale is set to zero, we use mean model
    if covscale == 0.
        # update weight matrix
        PredictMat.W .= meanW'
    # otherwise, we do Thompson sampling.  Each row is independent.
    else
        TestSampleMat = zeros(size(PredictMat.W,1),size(PredictMat.W,2))
        for row = 1:size(newy,1)
            samplerMat = MvNormal(meanW[:,row], Matrix(Hermitian(covW * covscale)))
            TestSampleMat[row,:] = rand(samplerMat)
        end
        # update weight matrix
        PredictMat.W .= TestSampleMat[:,:,1]
    end
    # update weight matrix in multiple learnedenvs for multi-threading
    for k in controller.envs 
        k.PredictMat .= PredictMat.W
    end
   
    
    #------------------updating state---------------#
    result = (
              iter = i,
              elapsed_sampled = elapsed_sample,
              traj_reward = sum(batch.rewards),
              traj_length = mean(length, batch),
              prederr = prederr
             )

    return result, i + 1
end
