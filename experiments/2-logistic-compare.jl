using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random, LinearAlgebra, SparseArrays, Printf
using Plots, LaTeXStrings
using OpenML, Tables, JLD2
using COSMO, OSQP, GeNIADMM
using IterativeSolvers, LinearMaps
using JuMP, MosekTools
using CSV, DataFrames, Statistics
include(joinpath(@__DIR__, "utils.jl"))
using GeNIOS

const DATAPATH = joinpath(@__DIR__, "data")
const DATAFILE_SPARSE = joinpath(DATAPATH, "real-sim.jld2")
const DATAFILE_SPARSE_2 = joinpath(DATAPATH, "news20.jld2")
const SAVEPATH = joinpath(@__DIR__, "saved")
const SAVEFILE = "2-logistic-compare-june2025-new"
const FIGS_PATH = joinpath(@__DIR__, "figures")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA_SPARSE = true
const HAVE_DATA_SPARSE2 = true
const RAN_TRIALS = false
const TEST_MODE = false

function run_trial(; type, n=100, verbose=false)
    if type == "test"
        # real-sim dataset
        A_full, b_full = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE, dataset_id=1578)
        A_full = A_full[1:n, 1:2n]
        b_full = b_full[1:n]

        # normalize A_full
        A_full = A_full ./ norm(A_full, Inf)

    elseif type == "sparse"
        # real-sim dataset
        A_full, b_full = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE, dataset_id=1578)
        # normalize A_full
        A_full = A_full ./ norm(A_full, Inf)

    elseif type == "sparse2"
        # news20 dataset
        A_full, b_full = load_sparse_data(file=DATAFILE_SPARSE_2, have_data=HAVE_DATA_SPARSE2, dataset_id=1594)
        # make into two classes:
        b_full[b_full .<= 10] .= -1.0
        b_full[b_full .> 10] .= 1.0

        # normalize A_full
        A_full = A_full ./ norm(A_full, Inf)
    else
        error("Unknown type: $type")
    end
    @info "Starting type = $type"

    # Reguarlization parameters
    λ1_max = norm(A_full'*b_full, Inf)
    λ1 = 0.1*λ1_max
    λ2 = 0.0

    A = Diagonal(b_full) * A_full
    b = zeros(size(A, 1))

    # For compilation
    solve!(
        GeNIOS.LogisticSolver(λ1, λ2, A, b); 
        options=GeNIOS.SolverOptions(
            use_dual_gap=true,
            max_iters=2,
            verbose=false, 
            precondition=true
    ))

    # GeNIOS
    @info "  -- GeNIOS --"
    solver = GeNIOS.LogisticSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        eps_abs=1e-4,
        eps_rel=1e-4,
        verbose=verbose,
        precondition=true,
        ρ0=10.0,
        max_iters=5000,
        max_time_sec=1800.
    )
    GC.gc()
    result = solve!(solver; options=options)
    time_genios = result.log.solve_time + result.log.setup_time
    @info "    time: $(round(time_genios, digits=3))"

    # COSMO (indirect)
    @info "  -- COSMO (indirect) --"
    model = construct_jump_model_logistic(A_full, b_full, λ1)
    set_optimizer(model, COSMO.Optimizer)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_optimizer_attribute(model, "eps_rel", 1e-4)
    set_optimizer_attribute(model, "kkt_solver", CGIndirectKKTSolver)
    set_optimizer_attribute(model, "verbose", verbose)
    set_time_limit_sec(model, 600.0)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "COSMO (indirect) did not solve the problem"
    time_cosmo_indirect = solve_time(model)
    @info "    time: $(round(time_cosmo_indirect, digits=3))"


    # COSMO (direct)
    @info "  -- COSMO (direct) --"
    model = construct_jump_model_logistic(A_full, b_full, λ1)
    set_optimizer(model, COSMO.Optimizer)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_optimizer_attribute(model, "eps_rel", 1e-4)
    set_optimizer_attribute(model, "verbose", verbose)
    set_time_limit_sec(model, 600.0)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "COSMO (direct) did not solve the problem"
    time_cosmo_direct = solve_time(model)
    @info "    time: $(round(time_cosmo_direct, digits=3))"

    # Mosek
    @info "  -- Mosek --"
    model = construct_jump_model_logistic(A_full, b_full, λ1)
    set_optimizer(model, Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-4)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-4)
    set_time_limit_sec(model, 600.0)
    if !verbose
        set_silent(model)
    end
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "Mosek did not solve the problem"
    time_mosek = solve_time(model)
    @info "    time: $(round(time_mosek, digits=3))"


    # FISTA
    @info "  -- FISTA --"
    prob = GeNIADMM.LogisticSolver(A_full, b_full, λ1)
    GC.gc()
    result_fista = GeNIADMM.solve!(
        prob; indirect=true, relax=false, max_iters=5_000, tol=1e-4, logging=true,
        precondition=false, verbose=verbose, print_iter=100, agd_x_update=true,
        rho_update_iter=10_000, multithreaded=true
    )
    time_fista = result_fista.log.solve_time + result_fista.log.setup_time
    @info "    time: $(round(time_fista, digits=3))"


    savefile = joinpath(SAVEPATH, SAVEFILE*"-$type.jld2")
    save(savefile, 
        "time_genios", time_genios,
        "time_cosmo_indirect", time_cosmo_indirect,
        "time_cosmo_direct", time_cosmo_direct,
        "time_mosek", time_mosek,
        "time_fista", time_fista
    )

    return nothing
end


if !RAN_TRIALS
    # compile
    run_trial(type="test")
    @info "Finished compiling"
    if !TEST_MODE
        types = ["sparse", "sparse2"]
        for type in types
            run_trial(type=type, verbose=true)
            @info "Finished with type=$type"
        end
    end
end
@info "Finished with all trials"


println("\\begin{tabular}{@{}lrrr@{}}")
println("\\toprule")
println("Dataset & GeNIOS & COSMO (indirect) & COSMO (direct) & Mosek & FISTA \\\\")
println("\\midrule")
for type in ["sparse", "sparse2"]
    savefile = joinpath(SAVEPATH, SAVEFILE*"-$type.jld2")
    time_genios, 
    time_cosmo_indirect, 
    time_cosmo_direct, 
    time_mosek,
    time_fista = 
        load(savefile, 
            "time_genios", 
            "time_cosmo_indirect", 
            "time_cosmo_direct", 
            "time_mosek", 
            "time_fista"
        )
    
    println(
        "$type & $(@sprintf("%.3f", time_genios)) & $(@sprintf("%.3f", time_cosmo_indirect)) & $(@sprintf("%.3f", time_cosmo_direct)) & $(@sprintf("%.3f", time_mosek)) & $(@sprintf("%.3f", time_fista)) \\\\"
    )
end
println("\\bottomrule")
println("\\end{tabular}")
