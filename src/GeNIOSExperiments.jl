module GeNIOSExperiments

# required packages for utilities
using Random, LinearAlgebra, SparseArrays, Printf
using CSV, DataFrames, JLD2, Statistics, OpenML, Tables     # for data
using Plots, LaTeXStrings                                   # for plotting
using JuMP, COSMO, OSQP, GeNIADMM, MosekTools               # other solvers
using IterativeSolvers, LinearMaps                          # for linear systems
using GeNIOS

include("utils.jl")

const DATAPATH = joinpath(@__DIR__, "..", "data")
const SAVEPATH = joinpath(@__DIR__, "..", "saved")
const FIGS_PATH = joinpath(@__DIR__, "..", "figures")

const DATAFILE_DENSE = joinpath(DATAPATH, "YearPredictionMSD.txt")
const DATAFILE_SPARSE = joinpath(DATAPATH, "real-sim.jld2")
const DATAFILE_SPARSE_2 = joinpath(DATAPATH, "news20.jld2")


end
