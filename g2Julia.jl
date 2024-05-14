using CSV, Plots, DataFrames, DelimitedFiles, DataStructures, StatsBase

data = CSV.read("qr2_250K_timestamps.csv", DataFrame)

#global binwidth = 5120

global data = Matrix(data)
global timestamps = vec(data)
#global timestamps = round.(timestamps/binwidth)
#global timestamps = timestamps[1:10000]

global data = readdlm("ToyModel5_timestamps.txt")
global timestamps = vec(data)

#global bin_starting_time = ((trunc(minimum(timestamps)/binwidth) |>Int64) -1) * binwidth
#global bin_ending_time = ((trunc(maximum(timestamps)/binwidth) |>Int64) +1) * binwidth

global timebins = collect(0:1:maximum(timestamps))

hist = fit(Histogram, timestamps, timebins)
global bin_intensity = hist.weights

#delay_times_1 = collect(1:9)
#delay_times_2 = collect(10:1:2000)
#global delay_times = vcat(delay_times_1, delay_times_2)
global delay_times = collect(1:1000)

#------------------------------------------------------------------------------------------------------------------------------#
#MATRIX FOR STORING DATA
global auto_correlation = zeros(length(delay_times), 1) |> Array
#------------------------------------------------------------------------------------------------------------------------------#

#global denominator = (sum(bin_intensity)/length(bin_intensity)) .^2

for delay in eachindex(delay_times)
    global delay_time = delay_times[delay]

#    global bin_starting_time = ((trunc(minimum(timestamps)/(binwidth*delay_time)) |>Int64) -1) * binwidth
#    global bin_ending_time = ((trunc(maximum(timestamps)/(binwidth*delay_time)) |>Int64) +1) * binwidth

#    global timebins = collect(0:delay_time:maximum(timestamps))

#    global hist = fit(Histogram, timestamps, timebins)
#    global bin_intensity = hist.weights

#    global denominator = (sum(bin_intensity)/length(bin_intensity)) .^2

    global zero_mx = vec(zeros(delay_time, 1))

    global bin_intensity_i = vcat(zero_mx, bin_intensity)
    global bin_intensity_j = vcat(bin_intensity, zero_mx)

    global cross_intensity = bin_intensity_i .* bin_intensity_j
    
    global denominator = sum(bin_intensity[1:(length(bin_intensity)-delay_time)]) *sum(bin_intensity[delay_time:length(bin_intensity)])

    global g2 = (sum(cross_intensity)*(length(bin_intensity)-delay_time))/denominator
    global auto_correlation[delay] = g2

    println(delay_time)
end
plot((delay_times[1:(length(delay_times)-1)] .+1 ), (auto_correlation[1:(length(auto_correlation)-1)]), framestyle=:box, xscale=:log10, 
    linewidth=2, xlabel="Time delay", ylabel="g2", label="Julia script",
    xminorticks=9, yminorticks=10, minorgrid=true, dpi=300)
