using Revise

envnames = ["ab", "cpg", "mc"]

# make directories to store log files
_ckmkdir(f) = isdir(f) == false && mkdir(f)
_ckmkdir("log")
for envname in envnames
    _ckmkdir("log/"*envname)
end

#--------------- for 4 seed numbers ---------------------# 
seednum = 1234
include("scripts/lc3_acrobot.jl")
include("scripts/lc3_cartpole.jl")
include("scripts/lc3_mountaincar.jl")
seednum = 2345
include("scripts/lc3_acrobot.jl")
include("scripts/lc3_cartpole.jl")
include("scripts/lc3_mountaincar.jl")
seednum = 3456
include("scripts/lc3_acrobot.jl")
include("scripts/lc3_cartpole.jl")
include("scripts/lc3_mountaincar.jl")
seednum = 4567
include("scripts/lc3_acrobot.jl")
include("scripts/lc3_cartpole.jl")
include("scripts/lc3_mountaincar.jl")


#--------------- compute the last results ---------------# 

for envname in envnames
    # reading data
    d1 = read("log/"*envname*"/"*envname*".jlso", JLSOFile)
    d2 = read("log/"*envname*"/"*envname*"_1.jlso", JLSOFile)
    d3 = read("log/"*envname*"/"*envname*"_2.jlso", JLSOFile)
    d4 = read("log/"*envname*"/"*envname*"_3.jlso", JLSOFile)
    # extracting reward data to vector
    reward_1 = Vector(d1[:logs][:algstate][:traj_reward])
    reward_2 = Vector(d2[:logs][:algstate][:traj_reward])
    reward_3 = Vector(d3[:logs][:algstate][:traj_reward])
    reward_4 = Vector(d4[:logs][:algstate][:traj_reward])
    lenalg = length(reward_1)
    movemean = zeros(lenalg); movestdm = zeros(lenalg)
    # 5000 window average and across four random seeds
    timestep = 200
    for i in 1:lenalg
        p = 0.
        for j in max(1,(i-Int(5000/timestep)+1)):i
            p += 1.
            movemean[i] += reward_1[j]; movemean[i] += reward_2[j]; movemean[i] += reward_3[j]; movemean[i] += reward_4[j];
            movestdm[i] += reward_1[j]^2; movestdm[i] += reward_2[j]^2; movestdm[i] += reward_3[j]^2; movestdm[i] += reward_4[j]^2;
        end
        movemean[i] /= (p * 4)
        movestdm[i] /= (p * 4)
    end
    reward_m = movemean; reward_s = sqrt.(movestdm - movemean.^2)
    finalm = reward_m[end]; finals = reward_s[end]

    println("LC3 for $envname -> final mean: $finalm  final std: $finals")
end
