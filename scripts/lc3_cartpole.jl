"""
This file is for executing Gym cartpole task with LC^3.
"""

#------------------------Packages-------------------#
using LinearAlgebra, Random, Statistics
using UnicodePlots, JLSO
using LyceumBase.Tools, LyceumBase, LyceumAI, UniversalLogger, Shapes
using Distributions, Parameters
using UnsafeArrays
using Distances

include("../utils/weightmat.jl")
include("../utils/LC3.jl")
include("../gym/cartpole.jl")
include("../learned_envs/learned_env.jl")
include("../planner/MPPIClamp.jl")

#-----------------------Constants -------------------#
numfeatures = 200      # d_phi
T           = 200      # task horizon
lambda_reg  = 0.0005   # prior covariance is I/lambda_reg
TS_scale    = 0.0001   # posterior reshaping constant for Thompson sampling

@info "Cartpole Task"
@info "Seed number" seednum
Random.seed!(seednum)
#----------------------Environment-------------------#
mjenvs = tconstruct(CartpoleGym, Threads.nthreads());
env = CartpoleGym()
dobs, dact = length(obsspace(env)), length(actionspace(env))
dstate = length(obsspace(env)) #output  (In our work, output is also obs instead of state)
PredictMat  = PredictionMat(numfeatures, dstate) # generates random matrix
ctrlrange = env.ctrlrange


@info "Dimensions of observations/actions/outputs" dobs dact dstate
@info "Simulation timestep:" LyceumBase.timestep(env)

#------------------Define Features-------------------#
include("../utils/rff.jl")
rffbandwidth = 1.5
const rff = RandomFourierFunctions{Float64}(rffbandwidth, dobs+dact, numfeatures)

#------------------Strategy Struct-------------------#
env_tconstructor = n -> tconstruct(CartpoleGym, n)
mppi = MPPIClamp(
            env_tconstructor = n -> tconstruct(LearnedEnv, PredictMat.W, rff, mjenvs, n),
            covar = Diagonal(0.2 ^2 * I, size(actionspace(env), 1)),
            lambda = 0.1,
            H = 50,
            K = 128,
            gamma = 1.,
            clamps = ctrlrange
           )

gt_mppi = MPPIClamp(
            env_tconstructor = n -> tconstruct(CartpoleGym, n),   #This is for GT-MPPI
            covar = Diagonal(0.2 ^2 * I, size(actionspace(env), 1)),
            lambda = 0.1,
            H = 50,
            K = 128,
            gamma = 1.,
            clamps = ctrlrange
           )

#------------------Strategy Struct-------------------#
randreset!(env)
lc3 = LC3(
        env,
        rff,
        PredictMat,
        numfeatures,
        dstate,
        dobs,
        dact,
        ctrlrange,
        lambda_reg,
        TS_scale,
        mppi;
        Hmax = T,
        N = T,
        );

#------------------Running the algo-------------------#
function cpg_lc3(lc3::LC3, plot::Bool; NITER=1000)
    # save data to the following file
    exper = Experiment("log/cpg/cpg.jlso", overwrite = false)

    lg = ULogger()
    for (i, state) in enumerate(lc3)
        if i >= NITER
            # save some constants here (Weight matrix is at the terminal episode)
            exper[:feat] = lc3.featurize
            exper[:timestep] = LyceumBase.timestep(env)
            exper[:PredictMat] = lc3.PredictMat
            exper[:mppistr] = lc3.controller
            break
        end
        # save all the log data
        push!(lg, :algstate, filter_nt(state, exclude = (:elapsed_sampled)))
        if plot && mod(i, 50) == 0
            x = lg[:algstate]
            # show reward curve
            display(expplot(
                            Line(x[:traj_reward], "reward evaluations"),
                            title = "reward evaluations, Iter=$i",
                            width = 60, height = 7,
                           ))
            # show prediction error plots
            display(expplot(
                            Line(x[:prederr], "prediction error"),
                            title = "model prediction error, Iter=$i",
                            width = 60, height = 7,
                           ))
            @info "Heatmap of Weight Matrix"
            display(heatmap(lc3.PredictMat.W))
        end
    end
    exper, lg
end

exper, lg = cpg_lc3(lc3, true; NITER=1000);

exper[:logs] = get(lg)
finish!(exper); # flushes everything to disk


