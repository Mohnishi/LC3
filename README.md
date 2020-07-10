## Readme

This code is meant to reproduce the results found in **Information Theoretic Regret Boundsfor Online Nonlinear Control** by Sham Kakade, Akshay Krishnamurthy, Kendall Lowrey, Motoya Ohnishi, and Wen Sun.
We include the benchmark task environments that do not require external licenses, namely the mountain-car, acrobot, and cartpole environments.

## Setup & Install

This code has been tested on Ubuntu 18.04, but should also work on different platforms (MacOS, Windows, FreeBSD) if the instructions are adapted.

The process to bring up this repo is as follows:
1. Download and install [Julia](https://julialang.org/)
2. Navigate to project and instantiate 
3. Run

The following is an example of installing Julia for Ubuntu 18.04.
```bash
cd ~/Downloads
wget https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.2-linux-x86_64.tar.gz
tar xvf julia-1.4.2-linux-x86_64.tar.gz

# the following exports can be added to your bashrc.
export JULIA_BINDIR=~/Downloads/julia-1.4.2/bin
export PATH=$JULIA_BINDIR:$PATH
export JULIA_NUM_THREADS=12

cd $directory_you_extracted_code
julia
```

One you start Julia, regardless of platform, the following instructions may proceed:
```julia
julia> ]
(@v1.4) pkg> activate .                        # activates this project
(LC3) pkg> instantiate   # the built in package manager downloads, installs dependences
(LC3) pkg> ctrl-c

julia> include("main.jl")                      # to run all the environments and generate results, or...
julia> seednum = 1234
julia> include("scripts/lc3_acrobot.jl")       # to run individual environments.
```

## Notes

The results in the paper were generated with **Julia 1.4.2**, with **12 Julia threads**. This is critical to reproducibility, but not necessary for running the included algorithm; one should adapt these settings to their compute.

## Code Structure

```bash
.
├── gym                    # Environment details and functions
│   ├── acrobot.jl
│   ├── cartpole.jl
│   └── mountaincar.jl
├── learned_envs           # Wrapper around environments to allow for learned models
│   └── learned_env.jl
├── log                    # Data store
│   ├── ab
│   ├── cpg
│   └── mc
├── main.jl                # Generates results from paper
├── Manifest.toml          # Julia Manifest file for all dependencies
├── planner
│   └── MPPIClamp.jl
├── Project.toml           # Julia Project file for top level dependencies
├── README.md              # This file
├── scripts                # Environment Hyper-Parameters and configuration
│   ├── lc3_acrobot.jl
│   ├── lc3_cartpole.jl
│   └── lc3_mountaincar.jl
└── utils                  # Algorithm and support code
    ├── LC3.jl
    ├── rff.jl
    └── weightmat.jl
```

## Code Maintenance

The codes are maintained by the authors of **Information Theoretic Regret Boundsfor Online Nonlinear Control** (https://arxiv.org/abs/2006.12466).
The project page can be found at https://sites.google.com/view/lc3algorithm/

