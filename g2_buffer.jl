using Plots, Random, FFTW, DelimitedFiles
ENV["GKSwstype"] = "100"

rng = MersenneTwister()

#define sampling rate
fs = 1

#define ranges of x and y axis.
x_min = 1
x_max = 40

y_min = 1
y_max = 40

x = collect(x_min:fs:x_max)
y = collect(y_min:fs:y_max)

#display(heatmap(fft_replica_t, color=:viridis, size =(400,400)))

#define the error scale for choosing the concentric rings for calculationg correlation
global error_scale = 0.5

#define system size
n_x = length(x)
n_y = length(y)
N_sd = n_x * n_y

#heatmap(replica_t, color=:viridis, cbar=false, size =(400,400))
#heatmap(replica_delt, color=:viridis, cbar=false, size =(400,400))

#reference position of fourier transform in 2D geometry
mx_sd = Array(collect(1:N_sd))

x_pos_sd = zeros(N_sd, 1)
y_pos_sd = zeros(N_sd, 1)

for i in 1:N_sd
    x_pos_sd[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sd[i] = ((i-1)%n_y)+1                         #1th position
end

x_pos_sd = reshape(x_pos_sd, n_x, n_y)
y_pos_sd = reshape(y_pos_sd, n_x, n_y)

function calculate_g2()
    #defining the center of the fourier transform Plots
    x_center = round(n_x/2) +1 |>Int64
    y_center = round(n_y/2) +1 |>Int64

    #distance of each point on fourier plot from the center point
    distance = sqrt.((x_pos_sd .- x_center).^2 .+ (y_pos_sd .- y_center).^2)
    cross_intensity_mx = Array{Complex{Float64}}(undef, n_x, n_y)
    self_intensity_mx = Array{Complex{Float64}}(undef, n_x, n_y)
    count_mx = zeros(n_x, n_y)

    #"for" loop to calculate coorelation between two replicas (numeretaor)
    for i in 1:n_x
        for j in 1:n_y
            distance_ij = distance[i,j]
            global intensity_mux =0
            global c = 0
            for k in 1:n_x
                for l in 1:n_y
                    if distance[k,l]<= (distance_ij+ error_scale) && distance[k,l]>= (distance_ij- error_scale)
                        global intensity_mux += fft_replica_delt[k,l]
                        global c = c+1
                    end
                end
            end
            cross_intensity_mx[i,j] = intensity_mux * fft_replica_t[i,j]
            count_mx[i,j] = c
        end
    end

    #"for" loop to calculate coorelation between two replicas (denominaor)
    for i in 1:n_x
        for j in 1:n_y
            distance_ij = distance[i,j]
            global intensity_mux =0
            global c = 0
            for k in 1:n_x
                for l in 1:n_y
                    if distance[k,l]<= (distance_ij+ error_scale) && distance[k,l]>= (distance_ij- error_scale)
                        global intensity_mux += fft_replica_t[k,l]
                        global c = c+1
                    end
                end
            end
            self_intensity_mx[i,j] = intensity_mux * fft_replica_t[i,j]
            count_mx[i,j] = c
        end
    end

    autocorrelation_fn = abs.(sum(cross_intensity_mx))/ abs.(sum(self_intensity_mx))
    return autocorrelation_fn
end
#------------------------------------------------------------------------------------------------------------------------------#

#matrix to save data
global g2_correlation = cross_intensity_mx = Array{Float64}(undef,0)

#------------------------------------------------------------------------------------------------------------------------------#
#main body
anim=@animate for snap in 1:10
    #define periodicity of sin function
    f_t(x,y) = sin(2*pi*y/10 + (snap -1)*2*pi/10)
    f_delt(x,y) = sin(2*pi*y/10+ (snap)*2*pi/10)

    global replica_t = f_t.(x',y)
    global replica_delt = f_delt.(x',y)

    #fourier transfor of replicas
    global fft_replica_t = fftshift(fft(replica_t))
    global fft_replica_delt = fftshift(fft(replica_delt))

    #calculate f2 value for two snapshots
    g2_value = calculate_g2()
    push!(g2_correlation, g2_value)

    heatmap(replica_delt, color=:viridis, size=(400,400))
end

gif(anim, "moving_grating_dummy.gif", fps=2)

plot(g2_correlation)
