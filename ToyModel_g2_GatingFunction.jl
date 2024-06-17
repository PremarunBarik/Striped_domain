using Plots, Random, DelimitedFiles, StatsBase

global MCSteps = 20000

#define switch function
global Ontime = 100
global Offtime = 0
global switch_function = vcat(ones(Ontime, 1), zeros(Offtime), 1)
global switch_function = repeat(switch_function, (trunc(MCSteps/(Ontime+Offtime)) |> Int64)+1, 1)
global switch_function = switch_function[1:MCSteps]

#define the detector size to collect the photon timestamps to replicate the experiment
global detector_length = 512
global detector_space = detector_length*detector_length
global detector_pixels = zeros(detector_space, 1)
global detectorWindow = 300


#collect the intensities throughout all the Monte Carlo steps

#introducing the noise
#global noise = rand(0:5, MCSteps)

#introduxing the signal
#global gating_period = 300
#x = collect(1:(gating_period))

#signal = 1 .- exp.((x .- (gating_period+20)) ./ 50)
#signal = repeat(signal, (trunc(MCSteps/(gating_period)) |> Int64)+1 ,1)
#global signal = 20*signal[1:MCSteps]

#the combined intensities (signal + noise)
#global intensities = noise .+ signal
#global intensities = intensities .* switch_function
#global intensities = Array{Int64}(trunc.(intensities))

#introducing the sinusoidal signal
mx = collect(1:MCSteps)
global signal_period = 200

global signal = 2 .+ 2*sin.(mx*2*pi/signal_period)
global noise = rand(1:100, MCSteps)
global intensities = signal .+ noise
global intensities = Array{Int64}(trunc.(intensities))
#global intensities = fill(10, MCSteps,1)


global timestamps = Float64[]

#function to generate timestamps on detector space
function generate_timestamps(step)

    if intensities[step] != 0
        global intensity = intensities[step]
        global rand_pos = rand(1:detector_space, intensity)

        for i in eachindex(rand_pos)
            if detector_pixels[rand_pos[i]]==0
                detector_pixels[rand_pos[i]] = step
            end
        end
    end

end

#function to download timestamps after one period of data collection on the detector
function download_timestamps()
    global timestamps_window = detector_pixels[ detector_pixels .!=0]
    global timestamps = vcat(timestamps, timestamps_window)

    global detector_pixels = zeros(detector_space, 1)
end

#changing the gated intensities to timestamps
for step in 1:MCSteps
    generate_timestamps(step)

    if (step % detectorWindow == 0)
        download_timestamps()
    end
end

#writedlm("ToyModel5_timestamps_SinSignal$(signal_period)PlusNoise.txt", timestamps)
#data = readdlm("/Users/premarunbarik/Documents/Research/Data_Codes/toy_model_autocorrelation/ToyModel5_timestamps_SinSignal.txt")
#data = readdlm("ToyModel5_timestamps_SinSignal200PlusNoise.txt")
#global timestamps = vec(data)
#global timestamps = sort!(timestamps)

#defininng time bins to calculate correlation
global timebins = collect(0:detectorWindow:maximum(timestamps))

#calculating incident photons in those timebins
hist = fit(Histogram, timestamps, timebins)
global bin_intensity = hist.weights

#define the lagbins to calculate correlation
global delay_times = collect(1:30)

#matrix to store correlation data
global auto_correlation = zeros(length(delay_times), 1) |> Array

#calculation of correlation for the mentioned lagbins
for delay in eachindex(delay_times)
    global delay_time = delay_times[delay]

    global zero_mx = vec(zeros(delay_time, 1))

    global bin_intensity_i = vcat(zero_mx, bin_intensity)
    global bin_intensity_j = vcat(bin_intensity, zero_mx)

    global cross_intensity = bin_intensity_i .* bin_intensity_j

    global denom = sum(bin_intensity[1:(length(bin_intensity)-delay_time)]) *sum(bin_intensity[delay_time:length(bin_intensity)])

    global g2 = (sum(cross_intensity)*(length(bin_intensity)-delay_time))/denom
    global auto_correlation[delay] = g2

    println(delay_time)
end

plot((delay_times[1:(length(delay_times))]*detectorWindow), (auto_correlation[1:(length(auto_correlation))] ), framestyle=:box, 
    linewidth=2, xlabel="Time delay (MC steps)", ylabel="g2", label="numpy.correlate",
    xminorticks=10, yminorticks=10, minorgrid=true, dpi=300, guidefont=font(12), tickfont=font(12), legendfont= font(12), legend=:bottomleft)

title!("g2 vs Delay time (Decorrelated Data)")

#plot(intensities[1:1000], framestyle=:box,
#    linewidth=2, xlabel="MC steps", ylabel="Intensity", label="Sinusoidal signal+noise",
#    xminorticks=9, yminorticks=10, minorgrid=true, dpi=300, guidefont=font(12), tickfont=font(12), legendfont= font(12))

#scatter(timestamps[1:5000], framestyle=:box,
#    ms=1, msw=0, ylabel="Timestamps", label="Sinusoidal signal",
#    xminorticks=9, yminorticks=10, minorgrid=true, dpi=300, guidefont=font(12), tickfont=font(12), legendfont= font(12))

#difference = auto_correlation[1:length(data3P)] .- data3P
#plot(data2P, difference, framestyle=:box, xscale=:log10,
#    linewidth=2, xlabel="Time delay (MC steps)", ylabel="Difference", label="pycorrelate - correlate",
#    xminorticks=9, yminorticks=10, minorgrid=true, dpi=300, guidefont=font(12), tickfont=font(12), legendfont= font(12))

#title!("Difference in g2 from correlate and pycorrelate")

#plot(data2P, data3P, framestyle=:box, xscale=:log10,
#    lw=2, ms=2, msw=0, xlabel="Time delay (MC steps)", ylabel="g2", label="pycorrelate.pcorrelate", legend=:topleft,
#    xminorticks=10, yminorticks=10, minorgrid=true, dpi=300, guidefont=font(12), tickfont=font(12), legendfont= font(12))

#title!("g2 vs Delay time (Decorrelated data)")
