using CUDA, Random, Plots, FFTW, StatsBase

rng = MersenneTwister(1234)
#total number of Monte Carlo steps
MC_steps = 20000

#define the detector size to collect the photon timestamps to replicate the experiment
global detector_length = 512
global detector_space = detector_length*detector_length
global detector_pixels = zeros(detector_space, 1)
global detectorWindow = 300

#list of delay times
#delay_times_1 = collect(1:9)
#delay_times_2 = Array{Int64}(reshape(collect(1:0.5:9.5) .* (10 .^ collect(1:2))', 18*2, 1))
#delay_times_3 = Array{Int64}(reshape(collect(1:0.1:9.9) .* (10 .^ collect(3:4))', 90*2, 1))
#global delay_times = vcat(delay_times_1, delay_times_2, delay_times_3)
global delay_times = collect(1:1000)

#define the grating function
#global gating_period = 200
#x = collect(1:(gating_period))

#introducing the sinusoidal signal
global signal_period = 200
global signal_amp = 2
global mx = collect(1:MC_steps)
global signal = signal_amp .+ signal_amp*sin.(mx*2*pi/signal_period)

#introducing noise to the signal
global noise_amp = 50
global noise = rand(0:noise_amp, MC_steps)

#add the signal and noise together
global intensities = signal + noise

#create a matrix with percentage of the whole monte carlo steps being 0ne and rest being zero.
global percent = 70
global ones_num = (percent/100)*MC_steps |> Int64
global random_filter = zeros(MC_steps)
global random_filter[randperm(MC_steps)[1:ones_num]] .= 1

#multiply the random filter to the intensities list
global intensities = intensities .* random_filter
global intensities = Array{Int64}(trunc.(intensities))

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
for step in 1:MC_steps
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
global bin_window = 20
global timebins = collect(0:10:maximum(timestamps))

#calculating incident photons in those timebins
hist = fit(Histogram, timestamps, timebins)
global bin_intensity = hist.weights

#define the lagbins to calculate correlation
global delay_times = collect(1:50)

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

plot((delay_times[1:(length(delay_times))]*bin_window), (auto_correlation[1:(length(auto_correlation))] ), framestyle=:box, 
    linewidth=2, xlabel="Time delay (MC steps)", ylabel="g2", legend=false,
    xminorticks=10, yminorticks=10, minorgrid=true, dpi=300, guidefont=font(12), tickfont=font(12))

title!("SinSignal amp:$(signal_amp), Noise amp:$(noise_amp), Percent:$(percent)")
#savefig("g2_SinSignal$(signal_amp)_Noise$(noise_amp)_Percent$(percent).png")

#scatter(signal[1:1000].*random_filter[1:1000], framestyle=:box, xlabel="MC steps", ylabel="Intensity",
#    guidefont=font(12), tickfont=font(12), legend=false,
#    ms=2, msw=0, dpi=300)
#title!("SinSignal amp:$(signal_amp), Noise amp:$(noise_amp), Percent:$(percent)")
#savefig("SinSignal$(signal_amp)_Noise$(noise_amp)_Percent$(percent).png")

