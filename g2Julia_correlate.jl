using CSV, Plots, DataFrames, DelimitedFiles, DataStructures, StatsBase

data = CSV.read("259K_sorted.csv", DataFrame)

global binwidth = 5120

global data = Matrix(data)
global timestamps = vec(data)
global timestamps = round.(timestamps/binwidth)
#global timestamps = timestamps[1:10000]

#global data = readdlm("ToyModel5_timestamps200.txt")
#global timestamps = vec(data)

#global bin_starting_time = ((trunc(minimum(timestamps)/binwidth) |>Int64) -1) * binwidth
#global bin_ending_time = ((trunc(maximum(timestamps)/binwidth) |>Int64) +1) * binwidth

#global detector_window = 5e6
#global timebins = collect(trunc(minimum(timestamps)/detector_window):detector_window:maximum(timestamps))
global timebins = collect(minimum(timestamps):1:maximum(timestamps))


hist = fit(Histogram, timestamps, timebins)
global bin_intensity = hist.weights

#delay_times_1 = collect(1:9)
#delay_times_2 = collect(10:100:100000)
#global delay_times = vcat(delay_times_1, delay_times_2)
global delay_times = collect(1:2000)

#------------------------------------------------------------------------------------------------------------------------------#
#MATRIX FOR STORING DATA
global auto_correlation = zeros(length(delay_times), 1) |> Array
#------------------------------------------------------------------------------------------------------------------------------#

#global denominator = (sum(bin_intensity)/length(bin_intensity)) .^2

for delay in eachindex(delay_times)
    global delay_time = delay_times[delay]

    global zero_mx = vec(zeros(delay_time, 1))

    global bin_intensity_i = vcat(zero_mx, bin_intensity)
    global bin_intensity_j = vcat(bin_intensity, zero_mx)

    global cross_intensity = bin_intensity_i .* bin_intensity_j
    
    global denom = sum(bin_intensity[1:(length(bin_intensity)-delay_time)]) *sum(bin_intensity[delay_time:length(bin_intensity)])

    global g2 = sum(cross_intensity)#*(length(bin_intensity)-delay_time))/denom
    global auto_correlation[delay] = g2

    println(delay_time)
end
plot((delay_times[2:length(delay_times)] .+1)*binwidth * 1e-8 , (auto_correlation[2:length(auto_correlation)]), framestyle=:box, 
    linewidth=3, xlabel="Time delay (sec)", ylabel="g2", label="Julia correlate", xscale=:log10,
    xminorticks=10, yminorticks=10, minorgrid=true, dpi=300, tickfont=font(12), guidefont=font(12),
    legendfont=font(12), legend=:bottomleft)
title!("259K_sorted.csv")

#savefig("DetectorData_g2_linear.png")
#writedlm("DelayTimes.txt", delay_times[2:length(delay_times)]*binwidth * 1e-8)
#writedlm("Autocorrelation.txt", auto_correlation[2:length(auto_correlation)])
