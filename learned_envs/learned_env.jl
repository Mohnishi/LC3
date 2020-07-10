"""
This file is for creating a virtual environment reflecting the learned model.
For our work, we use obs -> obs as a prediction and use obs for reward computation etc.
So, statespace and obsspace are the same; however, we can easily make this to be the case for state -> state
prediction while obs is still used for other purposes.
"""

using LyceumBase
using Shapes
using Base: @propagate_inbounds

# we assume the data stored in state, obs, action are vectors of some float type
struct LearnedEnv{D,E,K,P,
    SP,OP,AP,RP} <: AbstractEnvironment
    state::D
    obs::D
    action::D
    time::D
    mjenv::E 
    PredictMat::K
    phi::P
    features::D # scratch space
    obsact::D
    nextobs::D
    sp::SP
    op::OP
    ap::AP
    rp::RP
    function LearnedEnv{T}(PredictMat, phi, mjenv) where T<:AbstractFloat
        nobs = length(obsspace(mjenv))
        naction = length(actionspace(mjenv))
        sp = VectorShape(T, nobs)
        op = VectorShape(T, nobs)
        ap = VectorShape(T, naction)
        rp = ScalarShape(T)
        s = zeros(T, nobs)
        o = zeros(T, nobs)
        a = zeros(T, naction)
        env = new{typeof(s), typeof(mjenv), typeof(PredictMat), typeof(phi),
        typeof(sp), typeof(op), typeof(ap), typeof(rp)}(s,o,a,zeros(T,1),
                                                        mjenv,
                                                        copy(PredictMat), phi,
                                                        zeros(T, size(PredictMat, 2)),
                                                        zeros(T, nobs + naction),
                                                        zeros(T, nobs),
                                                        sp,op,ap,rp)
        reset!(env)
    end
end

function LyceumBase.tconstruct(::Type{LearnedEnv}, PredictMat, phi, mjenvs, n)
    @assert length(mjenvs) >= n
    Tuple(LearnedEnv{Float64}(PredictMat, phi, mjenv) for mjenv in mjenvs)
end
LearnedEnv(PredMat, phi, mjenv) = first(tconstruct(LearnedEnv, PredMat, phi, [mjenv], 1))



@inline LyceumBase.statespace(env::LearnedEnv) = env.sp
@inline LyceumBase.obsspace(env::LearnedEnv) = env.op
@inline LyceumBase.actionspace(env::LearnedEnv) = env.ap
@inline LyceumBase.rewardspace(env::LearnedEnv) = env.rp

@inline function LyceumBase.getstate!(state, env::LearnedEnv) 
    state .= 0.0 
    copyto!(state, env.state)
end
@inline function LyceumBase.setstate!(env::LearnedEnv, state) 
    if length(env.state) >= length(state)
        env.state .= state
    else
        nstate = length(env.state) # drops warmstart
        for i=1:nstate
            env.state[i] = state[i]
        end
    end
end

@inline LyceumBase.getaction!(action, env::LearnedEnv) = action .= env.action
@inline LyceumBase.setaction!(env::LearnedEnv, action) = env.action .= action

@propagate_inbounds function LyceumBase.isdone(state, action, obs, env::LearnedEnv)
    isdone(state, action, obs, env.mjenv)
end
@propagate_inbounds function LyceumBase.getobs!(obs, env::LearnedEnv)
    copyto!(env.obs, env.state)
    obs .= env.obs
end
@propagate_inbounds function LyceumBase.getreward(s, a, o, env::LearnedEnv) 
    getreward(s, a, o, env.mjenv) # uses original env's reward function
end
@propagate_inbounds function LyceumBase.geteval(s, a, o, env::LearnedEnv)
    geteval(s, a, o, env.mjenv)
end
@propagate_inbounds function LyceumBase.reset!(env::LearnedEnv)
    reset!(env.mjenv)
    getobs!(env.obs, env.mjenv)
    nstate = length(env.state) # drops warmstart
    copyto!(env.state, 1:nstate, env.obs, 1:nstate)
    env.time[1] = 0.0
    env
end
@propagate_inbounds function LyceumBase.randreset!(env::LearnedEnv)
    randreset!(env.mjenv)
    getobs!(env.obs, env.mjenv)
    nstate = length(env.state) # drops warmstart
    copyto!(env.state, 1:nstate, env.obs, 1:nstate)
    env.time[1] = 0.0
    env
end

@propagate_inbounds function LyceumBase.step!(env::LearnedEnv)
    # the code performs the following in a memory allocation free way
    # state += PredictMat * phi(vcat(obs, act))
    obsact, obs, act = env.obsact, env.obs, env.action
    copyto!(obsact, obs)                   # copy obs to start of vector
    r = (length(obs)+1):length(obsact)
    copyto!(obsact, r, act, 1:length(act)) # copy act to end of vector
    env.phi(env.features, obsact)
    mul!(env.nextobs, env.PredictMat, env.features)
    env.state .+= env.nextobs
    setstate!(env, env.state)
    env.time[1] += timestep(env)
    env
end



@inline LyceumBase.timestep(env::LearnedEnv) = timestep(env.mjenv)
@inline Base.time(env::LearnedEnv) = env.time[1]
