using Plots, Random, DelimitedFiles, FFTW, StatsBase

#define system size
n_x = 128
n_y = 128

N_sd = n_x * n_y

#define the detector size to collect the photon timestamps to replicate the experiment
global detector_length = 512
global detector_space = detector_length*detector_length
global detector_pixels = zeros(detector_space, 1)
global detectorWindow = 100

function make_dummy_stripes(stripe_width_up, stripe_width_down)
    global a = fill(1, n_y, stripe_width_up)
    global b = fill(-1, n_y, stripe_width_down)

    global mx = hcat(a, b)
    global mx = repeat(mx, 1, trunc(n_x/(stripe_width_up + stripe_width_down)) +1 |> Int64)
    global z_dir_sd = mx[:, 1:n_x]
    #global z_dir_sd = repeat(mx, 1, trunc(n_x/(stripe_width_up + stripe_width_down)) |> Int64)
end

#define total number of Monte Carlo Steps
global MCSteps = 5000

#define stripe width change interval
global snap_interval = 1000

#define box position on the fourier plot
global box_centre_x = (n_x/2 +1 ) |> Int64
global box_centre_y = (n_y/2 +1 + (n_x/(16))) |> Int64

global box_half_length = 1

global x_pos_box = zeros(2*box_half_length + 1, 1)
global y_pos_box = zeros(2*box_half_length + 1, 1)

for i in 1:(2*box_half_length + 1)
    x_pos_box[i] = (box_centre_x - (i - 2*box_half_length))
    y_pos_box[i] = (box_centre_y - (i - 2*box_half_length))
end

global x_pos_box = Array{Int64}(x_pos_box)
global y_pos_box = Array{Int64}(y_pos_box)

#calculate the intensity from a box in fourier space
function intensity_from_box()
    global photon_count = sum(z_dir_fft[x_pos_box, y_pos_box])
    return trunc(photon_count) |> Int64
end

#define the box on the fourier plot (for plotting)
global y_rect = [minimum(x_pos_box), maximum(x_pos_box), maximum(x_pos_box), minimum(x_pos_box), minimum(x_pos_box)]
global x_rect = [minimum(y_pos_box), minimum(y_pos_box), maximum(y_pos_box), maximum(y_pos_box), minimum(y_pos_box)]

#define a matrix to store intensities of Fourier plots throughout MC steps
global box_intensity = zeros(MCSteps, 1)

for i in 1:(MCSteps/snap_interval |> Int64)
    
    #make initial stripes 
    make_dummy_stripes(8,8)

    global anim = @animate for j in 1:snap_interval

        if (j == 100 || j == 900)
            make_dummy_stripes(9,7)
            println(j)
        end
        if (j == 200 || j == 800)
            make_dummy_stripes(10,6)
        end
        if (j == 300 || j == 700)
            make_dummy_stripes(11,5)
        end
        if (j == 400 || j == 600)
            make_dummy_stripes(10,6)
        end
        if (j == 500)
            make_dummy_stripes(11,5)
        end

        global step_count = ((i-1)*snap_interval) + j
        global z_dir_fft = abs.(fftshift(fft(z_dir_sd)))
        global box_intensity[step_count] = intensity_from_box()

#        display(heatmap(z_dir_sd, color=:grays, cbar=false, xticks=false, yticks=false, framestyle=:box, size=(400,400)))
#        println(step_count)

    end

#    heatmap(z_dir_sd, color=:grays, cbar=false, xticks=false, yticks=false, framestyle=:box, size=(400,400))
#    heatmap(z_dir_fft, color=:viridis, cbar=true, framestyle=:box, size=(400,400), alpha=0.5, minorgrid=true, minorticks=10)
#    plot!(x_rect, y_rect, lw=2, label="Defined box", color=:blue)
#    gif(anim, "SDdummy_config.gif", fps=1)
end

#gif(anim, "SDdummy_config.gif", fps=1)
#gif(anim, "SDdummy_config_fft.gif", fps=1)

global box_intensity = Array{Int64}(trunc.(box_intensity/100))

global timestamps = Float64[]

#function to generate timestamps on detector space
function generate_timestamps(step)

    if box_intensity[step] != 0
        global intensity = box_intensity[step]
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

#defininng time bins to calculate correlation
global timebins = collect(0:1:maximum(timestamps))

#calculating incident photons in those timebins
hist = fit(Histogram, timestamps, timebins)
global bin_intensity = hist.weights

#define the lagbins to calculate correlation
global delay_times = collect(1:2000)

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

plot((delay_times), (auto_correlation), framestyle=:box, 
    linewidth=2, xlabel="Time delay (MC steps)", ylabel="g2", legend=false,
    xminorticks=10, yminorticks=10, minorgrid=true, dpi=300, guidefont=font(12), tickfont=font(12), legendfont= font(12))

title!("g2 vs Delay time (Dummy Striped Domain)")

#scatter(timestamps[1:5000], framestyle=:box,
#    linewidth=2, ms=1, msw=0, xlabel="MC steps", ylabel="Timestamps", label="From defined box",
#    xminorticks=10, yminorticks=10, minorgrid=true, dpi=300, guidefont=font(12), tickfont=font(12), legendfont= font(12))

#title!("g2 vs Delay time (Dummy Striped domain)")


#plot(x_rect, y_rect)
