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
const DATAFILE_DENSE = joinpath(DATAPATH, "YearPredictionMSD.txt")
const DATAFILE_SPARSE = joinpath(DATAPATH, "real-sim.jld2")
const DATAFILE_SPARSE_2 = joinpath(DATAPATH, "news20.jld2")
const SAVEPATH = joinpath(@__DIR__, "saved")
const SAVEFILE = "1-elastic-net-compare-june2025-full"
const FIGS_PATH = joinpath(@__DIR__, "figures")

# Set this to false if you have not yet downloaded the real-sim dataset
const HAVE_DATA_SPARSE = true
const HAVE_DATA_SPARSE2 = true
const RAN_TRIALS = false
const TEST_MODE = false

function load_all_datasets()
    load_sparse_data(file=DATAFILE_SPARSE, have_data=false, dataset_id=1578)
    load_sparse_data(file=DATAFILE_SPARSE_2, have_data=false, dataset_id=1594)
end

function run_trial(; type, m=10_000, n=20_000, verbose=false)
    if type == "sparse"
        # real-sim dataset
        A, b = load_sparse_data(file=DATAFILE_SPARSE, have_data=HAVE_DATA_SPARSE, dataset_id=1578)
        # n = size(A, 2)
        # A = A[1:n, :]
        # b = b[1:n]
    elseif type == "sparse2"
        # news20 dataset
        A, b = load_sparse_data(file=DATAFILE_SPARSE_2, have_data=HAVE_DATA_SPARSE2, dataset_id=1594)
        m = size(A, 1)
    elseif type == "dense"
        A, b = get_augmented_data(m, n, DATAFILE_DENSE)
    else
        error("Unknown type: $type")
    end
    @info "Starting type = $type,\t(m, n) = $(size(A))"

    # Reguarlization parameters
    λ1_max = norm(A'*b, Inf)
    λ1 = 0.1*λ1_max
    λ2 = λ1


    # GeNIOS
    @info "  -- GeNIOS --"
    solver = GeNIOS.ElasticNetSolver(λ1, λ2, A, b)
    options = GeNIOS.SolverOptions(
        relax=true,
        α=1.6,
        eps_abs=1e-4,
        eps_rel=1e-4,
        verbose=verbose,
        precondition=true,
        sketch_update_iter=1000,    # We know that the Hessian AᵀA does not change
        ρ0=10.0,
        rho_update_iter=1000,
        max_iters=5000,
        max_time_sec=600.
    )
    GC.gc()
    result = solve!(solver; options=options)
    time_genios = result.log.solve_time + result.log.setup_time
    @info "    time: $(round(time_genios, digits=3))"



    @info "  -- OSQP --"
    model = construct_jump_model_elastic_net(A, b, λ1, λ2)
    set_optimizer(model, OSQP.Optimizer)
    set_optimizer_attribute(model, "eps_abs", 1e-4)
    set_optimizer_attribute(model, "eps_rel", 1e-4)
    set_optimizer_attribute(model, "verbose", verbose)
    set_time_limit_sec(model, 600.0)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.:OPTIMAL && @warn "OSQP did not solve the problem"
    time_osqp = solve_time(model)
    @info "    time: $(round(time_osqp, digits=3))"


    # COSMO (indirect)
    @info "  -- COSMO (indirect) --"
    model = construct_jump_model_elastic_net(A, b, λ1, λ2)
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
    model = construct_jump_model_elastic_net(A, b, λ1, λ2)
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
    model = construct_jump_model_elastic_net(A, b, λ1, λ2)
    set_optimizer(model, Mosek.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-4)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-4)
    set_time_limit_sec(model, 600.0)
    !verbose && set_silent(model)
    GC.gc()
    optimize!(model)
    termination_status(model) != MOI.OPTIMAL && @warn "Mosek did not solve the problem"
    time_mosek = solve_time(model)
    @info "    time: $(round(time_mosek, digits=3))"


    # FISTA
    @info "  -- FISTA --"
    prob = GeNIADMM.LassoSolver(A, b, λ1; μ=λ2)
    GC.gc()
    result_fista = GeNIADMM.solve!(
        prob; indirect=true, relax=false, max_iters=5_000, tol=1e-4, logging=true,
        precondition=false, verbose=verbose, print_iter=100, agd_x_update=true,
        rho_update_iter=10_000, multithreaded=true
    )
    time_fista = result_fista.log.solve_time + result_fista.log.setup_time
    @info "    time: $(round(time_fista, digits=3))"

    n < 1000 && return nothing
    savefile = joinpath(SAVEPATH, SAVEFILE*"-$type.jld2")
    save(savefile, 
        "time_genios", time_genios,
        "time_osqp", time_osqp,
        "time_cosmo_indirect", time_cosmo_indirect,
        "time_cosmo_direct", time_cosmo_direct,
        "time_mosek", time_mosek,
        "time_fista", time_fista
    )

    return nothing
end

if !HAVE_DATA_SPARSE || !HAVE_DATA_SPARSE2
    load_all_datasets()
end
if !RAN_TRIALS
    # compile
    run_trial(type="dense"; m=100, n=200)

    @info "Finished compiling"
    if !TEST_MODE
        types = ["sparse", "dense"]
        #types = ["dense"]
        for type in types
            run_trial(type=type; verbose=true)
            @info "Finished with type=$type"
        end
    end
end
@info "Finished with all trials"


println("\\begin{tabular}{@{}lrrrrr@{}}")
println("\\toprule")
println("Dataset & GeNIOS & OSQP & COSMO (indirect) & COSMO (direct) & Mosek & FISTA \\\\")
println("\\midrule")
for type in ["dense", "sparse",]
    savefile = joinpath(SAVEPATH, SAVEFILE*"-$type.jld2")
    time_genios, time_osqp, time_cosmo_indirect, time_cosmo_direct, time_mosek, time_fista = 
        load(savefile, "time_genios", "time_osqp", "time_cosmo_indirect", "time_cosmo_direct", "time_mosek", "time_fista")
    
    println("$type & $(@sprintf("%.3f", time_genios)) & $(@sprintf("%.3f", time_osqp)) & $(@sprintf("%.3f", time_cosmo_indirect)) & $(@sprintf("%.3f", time_cosmo_direct)) & $(@sprintf("%.3f", time_mosek)) & $(@sprintf("%.3f", time_fista)) \\\\")
end
println("\\bottomrule")
println("\\end{tabular}")
