using CSV, Plots, DataFrames, DelimitedFiles, DataStructures

data = CSV.read("qr2_250K_timestamps.csv", DataFrame)
global mx_MCSteps = Matrix(data)

c = counter(mx_MCSteps)
global time_stamps = collect(keys(c))
global photon_count = collect(values(c))

global sorted_index = sortperm(time_stamps)
sort!(vec(time_stamps))
global sorted_photon_count = photon_count[sorted_index]

#------------------------------------------------------------------------------------------------------------------------------#

#global zero_mx = vec(zeros(1,1))
#global time_stamps1 = vcat(zero_mx, time_stamps)
#global time_stamps2 = vcat(time_stamps, zero_mx)

#global del_time_stamps = time_stamps2 .- time_stamps1
#global del_time_stamps = del_time_stamps[2:length(del_time_stamps)-1]

#global sorted_del_time_stamps = sort(del_time_stamps)
#c_del = counter(sorted_del_time_stamps)
#global del_times = collect(keys(c_del))
#global del_times_count = collect(values(c))

#global sorted_index = sortperm(del_times)
#sort!(vec(del_times))
#global sorted_del_times_count = del_times_count[sorted_index]

#------------------------------------------------------------------------------------------------------------------------------#

global exp_starting_time = minimum(time_stamps)
global exp_ending_time = maximum(time_stamps)
global reduced_times = Array{Int}(undef,0)

global exp_duration = exp_ending_time-exp_starting_time
global bin_num = trunc(exp_duration/(5120)) |> Int64

global time_bin = zeros(bin_num, 1)
