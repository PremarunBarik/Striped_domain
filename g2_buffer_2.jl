using CSV, Plots, DataFrames, DelimitedFiles, DataStructures

data = CSV.read("qr2_250K_timestamps.csv", DataFrame)
global mx_MCSteps = Matrix(data)

c = counter(mx_MCSteps)
global time_stamps = collect(keys(c))
global photon_count = collect(values(c))

global sorted_index = sortperm(time_stamps)
sort!(vec(time_stamps))
global sorted_photon_count = photon_count[sorted_index]

global exp_starting_time = minimum(time_stamps)
global exp_ending_time = maximum(time_stamps)
global reduced_times = Array{Int}(undef,0)
