using Plots, DelimitedFiles, Random, FFTW

rng = MersenneTwister()

#define the error scale for choosing the concentric rings for calculationg correlation
global error_scale = 0.5

#define the system size
n_x = 24
n_y = 24
N_sd = n_x*n_y

#reference position of fourier transform in 2D geometry
mx_sd = Array(collect(1:N_sd))

x_pos_sd = zeros(N_sd, 1)
y_pos_sd = zeros(N_sd, 1)

for i in 1:N_sd
    x_pos_sd[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sd[i] = ((i-1)%n_y)+1                         #1th position
end

global x_pos_sd = reshape(x_pos_sd, n_x, n_y)
global y_pos_sd = reshape(y_pos_sd, n_x, n_y)

#------------------------------------------------------------------------------------------------------------------------------#
#create the initial matrix
a = ones(24,4)
b = (-1)*ones(24,4)

global mx = hcat(a,b)
global mx = repeat(mx, 1,3)

#------------------------------------------------------------------------------------------------------------------------------#
function calculate_g2()
    #defining the center of the fourier transform Plots
    x_center = round(n_x/2) +1 |>Int64
    y_center = round(n_y/2) +1 |>Int64

    #distance of each point on fourier plot from the center point
    global distance = sqrt.((x_pos_sd .- x_center).^2 .+ (y_pos_sd .- y_center).^2)

    #fourier transform of the replicas a time step apart
    global fft_replica_t = fftshift(fft(replica_t))
    global fft_replica_delt = fftshift(fft(replica_delt))

    global cross_intensity_mx = Array{Complex{Float64}}(undef, n_x, n_y)
    global self_intensity_mx = Array{Complex{Float64}}(undef, n_x, n_y)
    global count_mx = zeros(n_x, n_y)

    #"for" loop to calculate coorelation between two replicas (numeretaor)
    for i in 1:n_x
        for j in 1:n_y
            global distance_ij = distance[i,j]
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
            global cross_intensity_mx[i,j] = intensity_mux * fft_replica_t[i,j]
            global count_mx[i,j] = c
        end
    end

    #"for" loop to calculate coorelation between two replicas (denominaor)
    for i in 1:n_x
        for j in 1:n_y
            global distance_ij = distance[i,j]
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
            global self_intensity_mx[i,j] = intensity_mux * fft_replica_t[i,j]
            global count_mx[i,j] = c
        end
    end

    autocorrelation_fn = abs.(sum(cross_intensity_mx))/ abs.(sum(self_intensity_mx))
    return autocorrelation_fn
end
#------------------------------------------------------------------------------------------------------------------------------#
#matrix to store g2 data
global g2_correlation = Array{Float64}(undef,0)

#------------------------------------------------------------------------------------------------------------------------------#
#main body
global replica_t = mx
global c = (-1)*ones(24,1)
anim=@animate for i in 1:24
    global replica_t = mx
    global mx = hcat(c,mx)
    global mx = mx[:, 1:end-1]

    global replica_delt = mx

    #------------------------------------------------------------------------------------------------------------------------------#
    #defining the center of the fourier transform Plots
    x_center = round(n_x/2) +1 |>Int64
    y_center = round(n_y/2) +1 |>Int64

    #distance of each point on fourier plot from the center point
    global distance = sqrt.((x_pos_sd .- x_center).^2 .+ (y_pos_sd .- y_center).^2)

    #fourier transform of the replicas a time step apart
    global fft_replica_t = fftshift(fft(replica_t))
    global fft_replica_delt = fftshift(fft(replica_delt))

    global cross_intensity_mx = Array{Complex{Float64}}(undef, n_x, n_y)
    global self_intensity_mx = Array{Complex{Float64}}(undef, n_x, n_y)
    global count_mx = zeros(n_x, n_y)

    #"for" loop to calculate coorelation between two replicas (numeretaor)
    for i in 1:n_x
        for j in 1:n_y
            global distance_ij = distance[i,j]
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
            global cross_intensity_mx[i,j] = intensity_mux * fft_replica_t[i,j]
            global count_mx[i,j] = c
        end
    end

    #"for" loop to calculate coorelation between two replicas (denominaor)
    for i in 1:n_x
        for j in 1:n_y
            global distance_ij = distance[i,j]
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
            global self_intensity_mx[i,j] = intensity_mux * fft_replica_t[i,j]
            global count_mx[i,j] = c
        end
    end

    autocorrelation_fn = abs.(sum(cross_intensity_mx))/ abs.(sum(self_intensity_mx))
    push!(g2_correlation, autocorrelation_fn)

    #------------------------------------------------------------------------------------------------------------------------------#
    if(i%4 == 0)
        global c .= (-1)*c
    end
    heatmap(mx, color=:grays, alpha=0.5, framestyle=:box, cbar=false, size=(400,400))
end

gif(anim, "Test.gif", fps=2)

plot(g2_correlation)
