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
    a = fill(1, n_y, stripe_width_up)
    b = fill(-1, n_y, stripe_width_down)

    mx = hcat(a, b)
    mx = repeat(mx, 1, trunc(n_x/(stripe_width_up + stripe_width_down)) +1 |> Int64)
    global z_dir_sd = mx[:, 1:n_x]
    #global z_dir_sd = repeat(mx, 1, trunc(n_x/(stripe_width_up + stripe_width_down)) |> Int64)
end

make_dummy_stripes(11,5)
#heatmap(z_dir_sd, color=:grays, cbar=false, xticks=false, yticks=false, framestyle=:box, size=(400,400))

global z_dir_fft = abs.(fftshift(fft(z_dir_sd)))
heatmap(z_dir_fft, color=:viridis, cbar=true, framestyle=:box, size=(400,400), alpha=0.5, minorgrid=true, minorticks=10)

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

#define the box on the fourier plot (for plotting)
global y_rect = [minimum(x_pos_box), maximum(x_pos_box), maximum(x_pos_box), minimum(x_pos_box), minimum(x_pos_box)]
global x_rect = [minimum(y_pos_box), minimum(y_pos_box), maximum(y_pos_box), maximum(y_pos_box), minimum(y_pos_box)]

plot!(x_rect, y_rect)
