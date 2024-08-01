using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions, DelimitedFiles, FFTW
ENV["GKSwstype"] = "100"
CUDA.set_runtime_version!(v"11.7")

global rng = MersenneTwister()
global B_global = 0.0   #globally applied field on the system
#global alpha_ratio = 0.5     #defined parameter for dipolar interaction energy
#global Temp = 4.0      #defined temperature of the system

#define temperature matrix with particular temperature values
#min_Temp = 
#max_Temp = 0.6
#Temp_step = 20
#Temp_interval = (max_Temp - min_Temp)/Temp_step
#Temp_values = collect(min_Temp:Temp_interval:max_Temp)
#Temp_values = reverse(Temp_values)

Temp_values_1 = collect(1.0:0.1:1.4)
Temp_values_2 = collect(1.5:0.05:3.5)
Temp_values_3 = collect(3.6:0.1:4.0)
Temp_values = vcat(Temp_values_1, Temp_values_2, Temp_values_3)
global Temp_values = reverse(Temp_values) |> Array{Float32}

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF MC MC STEPS 
global MC_steps = 200000
global MC_burns = 300000

#NUMBER OF LOCAL MOMENTS
n_x = 32
n_y = 32
n_z = 1
global N_sd = n_x*n_y

#NUMBER OF REPLICAS 
global replica_num = 20

#define interaction co-efficients of NN and NNN interactions
global J_NN = 6.0

#LENGTH OF DIPOLE 
global dipole_length = 1

#------------------------------------------------------------------------------------------------------------------------------#

#define the pinning sites 
pinning_site_num = 0

#define anisotropy energy of pinning sites
K_anisotropy = 50

#defining pinning site accross the replicas
pinning_site_pos = zeros(pinning_site_num, replica_num)

for k in 1:replica_num
    #selecting spins in random positionsin one single replica
    random_position = randperm(N_sd)
    pinning_site_pos[:,k] = random_position[1:pinning_site_num]
end

x_pos_pinning_site = zeros(pinning_site_num, replica_num) |> Array{Int32}
y_pos_pinning_site = zeros(pinning_site_num, replica_num) |> Array{Int32}

for k in 1:replica_num
    for i in 1:pinning_site_num
        x_pos_pinning_site[i,k] = trunc((pinning_site_pos[i,k]-1)/n_x)+1                    #10th position
        y_pos_pinning_site[i,k] = ((pinning_site_pos[i,k]-1)%n_y)+1                         #1th position
    end
end

pinning_site_pos .= pinning_site_pos .+ (N_sd*collect(0:replica_num-1))'
global pinning_site_pos = reshape(CuArray{Int32}(pinning_site_pos), pinning_site_num*replica_num, 1)

global pinning_energy = zeros(N_sd*replica_num, 1) |> CuArray{Float32}
global pinning_energy[pinning_site_pos] .= K_anisotropy

#------------------------------------------------------------------------------------------------------------------------------#

#SPIN ELEMENT DIRECTION IN REPLICAS
function initialize_spin_config()

    global z_dir_sd = dipole_length*[(-1)^rand(rng, Int32) for i in 1:N_sd]
    global z_dir_sd = repeat(z_dir_sd, replica_num, 1) |> CuArray{Float32}

    global z_dir_sd[pinning_site_pos] .= dipole_length
end

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sd = Array{Int32}(collect(1:N_sd*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY -- needed to define neighbors and to 
#plot the spin configuration. So, we don't need to create a Array of these matrices 
#also no need to repeat for replicas because spin positions are constant over replicas 

x_pos_sd = zeros(N_sd, replica_num) |> Array{Int32}
y_pos_sd = zeros(N_sd, replica_num) |> Array{Int32}

for k in 1:replica_num
    for i in 1:N_sd
        x_pos_sd[i,k] = trunc((i-1)/n_x)+1                    #10th position
        y_pos_sd[i,k] = ((i-1)%n_y)+1                         #1th position
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#NEAR NEIGHBOUR CALCULATION
NN_s = zeros(N_sd,replica_num)
NN_n = zeros(N_sd,replica_num)
NN_e = zeros(N_sd,replica_num)
NN_w = zeros(N_sd,replica_num)

for k in 1:replica_num
for i in 1:N_sd                             #loop over all the spin ELEMENTS
        if x_pos_sd[i,k]%n_x == 0
            r_e =  (x_pos_sd[i,k]-n_x)*n_x + y_pos_sd[i,k]
        else
            r_e =  x_pos_sd[i,k]*n_x + y_pos_sd[i,k]
        end
        NN_e[i,k] = r_e + (k-1)*N_sd
        
        #-----------------------------------------------------------#

        if x_pos_sd[i,k]%n_x == 1
            r_w = (x_pos_sd[i,k]+n_x-2)*n_x + y_pos_sd[i,k]
        else
            r_w = (x_pos_sd[i,k]-2)*n_x + y_pos_sd[i,k]
        end
        NN_w[i,k] = r_w + (k-1)*N_sd

        #-----------------------------------------------------------#

        if y_pos_sd[i,k]%n_y == 0
            r_n =  (x_pos_sd[i,k]-1)*n_x + (y_pos_sd[i,k]-n_y+1)
        else
            r_n = (x_pos_sd[i,k]-1)*n_x + y_pos_sd[i,k]+1
        end
        NN_n[i,k] = r_n + (k-1)*N_sd

        #-----------------------------------------------------------#

        if y_pos_sd[i,k]%n_y == 1
            r_s = (x_pos_sd[i,k]-1)*n_x + (y_pos_sd[i,k]+n_y-1)
        else
            r_s = (x_pos_sd[i,k]-1)*n_x + y_pos_sd[i,k]-1
        end
        NN_s[i,k] = r_s + (k-1)*N_sd

end
end

#------------------------------------------------------------------------------------------------------------------------------#

#REPLICA REFERENCE MATRIX OF RANDOMLY SELECTED SPIN IN MC_STEP
rand_rep_ref_sd = zeros(replica_num, 1)

for i in eachindex(rand_rep_ref_sd)
    rand_rep_ref_sd[i] = (i-1)*N_sd
end

#------------------------------------------------------------------------------------------------------------------------------#

global denom = readdlm("SD_LeknerSum_denom_L$(n_x).txt")
global denom = 2 .* denom |> CuArray{Float32}

#------------------------------------------------------------------------------------------------------------------------------#

#compute dipolar energy of the system
function compute_dipolar_energy()
    z_dir_sd_i = reshape(z_dir_sd, N_sd, 1, replica_num)
    z_dir_sd_j = reshape(z_dir_sd, 1, N_sd, replica_num)

    global dipolar_energy = (z_dir_sd_i .* z_dir_sd_j) .* denom
    global dipolar_energy = sum(dipolar_energy, dims = 2)


    global dipolar_energy = reshape(dipolar_energy, N_sd*replica_num, 1) |> CuArray{Float32}
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO EXCHANGE 
global energy_exchange = zeros(N_sd*replica_num, 1) |> CuArray{Float32}

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE EXCHANGE ENERGY OF THE SYSTEM
function compute_exchange_energy()
    energy_x = z_dir_sd.*(J_NN .* (z_dir_sd[NN_n] .+ z_dir_sd[NN_s] .+ z_dir_sd[NN_e] .+ z_dir_sd[NN_w]))

    global energy_exchange = energy_x
end

#------------------------------------------------------------------------------------------------------------------------------#
#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sd*replica_num, 1) |> CuArray{Float32}

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_tot_energy_spin_glass()
    compute_exchange_energy()
    compute_dipolar_energy()

    global energy_tot = (((-1).*energy_exchange) .+ (dipolar_energy) .- (B_global*z_dir_sd) .- pinning_energy)

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = zeros(replica_num, 1) |> CuArray{Float32}

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global rand_pos =  CuArray{Int32}(rand(rng, (1:N_sd), (replica_num, 1)))
    global r = rand_pos .+ rand_rep_ref_sd

    global del_energy = (-2)*energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#Matrix to keep track of which flipped how many times
#global flip_count = Array(zeros(N_sd*replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(rng, Temp)                                           #benchmark time: 438.17 microsecond(20x20), 15.9 ms(50x50)
    compute_del_energy_spin_glass(rng)

    trans_rate = exp.(-del_energy/Temp)
    global rand_num_flip = CuArray{Float32}(rand(rng, Float64, (replica_num, 1)))
    flipit = sign.(rand_num_flip .- trans_rate)
    global z_dir_sd[r] = flipit.*z_dir_sd[r]

#    flipit = (abs.(flipit .- 1))/2
#    global flip_count[r] = flip_count[r] .+ flipit
end

#------------------------------------------------------------------------------------------------------------------------------#

#function to calculate orientational disorder
function orientational_order_parameter()

    global AF_bonds_s = abs.((z_dir_sd .* z_dir_sd[NN_s]) .- 1)/2
    global AF_bonds_n = abs.((z_dir_sd .* z_dir_sd[NN_n]) .- 1)/2
    global AF_bonds_e = abs.((z_dir_sd .* z_dir_sd[NN_e]) .- 1)/2
    global AF_bonds_w = abs.((z_dir_sd .* z_dir_sd[NN_w]) .- 1)/2

    global AF_bonds_horizontal = AF_bonds_s .+ AF_bonds_n 
    global AF_bonds_vertical = AF_bonds_e .+ AF_bonds_w
    global AF_bonds_total = AF_bonds_e .+ AF_bonds_n .+ AF_bonds_s .+ AF_bonds_w

    global AF_bonds_horizontal = sum(reshape(AF_bonds_horizontal, N_sd, replica_num), dims=1)/N_sd |> Array{Float32}
    global AF_bonds_vertical = sum(reshape(AF_bonds_vertical, N_sd, replica_num), dims=1)/N_sd |> Array{Float32}
    global AF_bonds_total = sum(reshape(AF_bonds_total, N_sd, replica_num), dims=1)/N_sd |> Array{Float32}

    global O_hv = abs.(AF_bonds_horizontal - AF_bonds_vertical)/sum(AF_bonds_total)

end

#function to calculate uncertainties
function calculate_uncertainty()

    global AF_bonds_horizontal_av = sum(AF_bonds_horizontal_sum)/(MC_steps * replica_num) |> Float32
    global AF_bonds_horizontal_uncertainty = (AF_bonds_horizontal_sum .- AF_bonds_horizontal_av) .^ 2 |> Array{Float32}
    global AF_bonds_horizontal_uncertainty = sqrt(sum(AF_bonds_horizontal_uncertainty)/replica_num) |> Float32

    global AF_bonds_vertical_av = sum(AF_bonds_vertical_sum)/(MC_steps * replica_num) |> Float32
    global AF_bonds_vertical_uncertainty = (AF_bonds_vertical_sum .- AF_bonds_vertical_av) .^ 2 |> Array{Float32}
    global AF_bonds_vertical_uncertainty = sqrt(sum(AF_bonds_vertical_uncertainty)/replica_num) |> Float32

    global AF_bonds_total_av = sum(AF_bonds_total_sum)/(MC_steps * replica_num) |> Float32
    global AF_bonds_total_uncertainty = (AF_bonds_total_sum .- AF_bonds_total_av) .^ 2 |> Array{Float32}
    global AF_bonds_total_uncertainty = sqrt(sum(AF_bonds_total_uncertainty)/replica_num) |> Float32

    global OrientOrder_parameter_av = sum(OrientOrder_parameter_sum)/(MC_steps * replica_num) |> Float32
    global OrientOrder_parameter_uncertainty = (OrientOrder_parameter_sum .- OrientOrder_parameter_av) .^ 2 |> Array{Float32}
    global OrientOrder_parameter_uncertainty = sqrt(sum(OrientOrder_parameter_uncertainty)/replica_num) |> Float32
    global OrientOrder_parameter_av = OrientOrder_parameter_av*replica_num |> Float32
end

#------------------------------------------------------------------------------------------------------------------------------#

#define the circle to calculate intensity from
global fft_x_center = (trunc(n_x/2) + 1) |> Int32
global fft_y_center = (trunc(n_y/2) + 1) |> Int32

global fft_distance = sqrt.(((x_pos_sd[1:N_sd] .- fft_x_center) .^ 2 ) .+ ((y_pos_sd[1:N_sd] .- fft_y_center) .^ 2))

global stripe_width = 4
global circle_radius = n_x/(2*stripe_width)   #stripe width depend on the exchange to dipolar interaction coefficient ratio 
                                            #If J= 6.0, stripe width is 4, if J=8.9 stripe width is 8
global radius_error = 0.9                   #depending on the choice of this we can define how thick the concentric
                                            #circle would be on the fourier plot
global fft_cell_list = Int32[]              #list of matrix cells where we collect the intensities from the Fourier plots

for i in 1:N_sd
    if fft_distance[i]<= (circle_radius + radius_error) && fft_distance[i] >= (circle_radius - radius_error)
        append!(fft_cell_list, i)
    end
end
global fft_cell_list = fft_cell_list .+ (N_sd .* collect(0:replica_num-1))'

function fft_calculation()
    global z_dir_sd_fft = reshape(z_dir_sd, n_x, n_y, replica_num) |> Array{Float32}
    global z_dir_sd_fft = fftshift.(fft.(eachslice(z_dir_sd_fft, dims=3)))
    global z_dir_sd_fft = abs.(reduce((x,y) -> cat(x, y, dims=3), z_dir_sd_fft))
    global z_dir_sd_fft = z_dir_sd_fft ./ sum(z_dir_sd_fft, dims=(1,2))
    
    return z_dir_sd_fft
end

#------------------------------------------------------------------------------------------------------------------------------#

function fft_intensity_ring()

    global fft_intensity = StFct_sum[fft_cell_list] |> Array{Float32}
    global fft_intensity = sum(fft_intensity, dims=1) |> Array{Float32}
    global fft_intensity_ReplicaAv = sum(fft_intensity) / replica_num
    global fft_intensity_uncertaity = (fft_intensity .- fft_intensity_ReplicaAv) .^ 2
    global fft_intensity_uncertaity = sqrt(sum(fft_intensity_uncertaity)/replica_num)
    return fft_intensity_ReplicaAv, fft_intensity_uncertaity
end

#------------------------------------------------------------------------------------------------------------------------------#

#plotting fourier transform
function plot_fft_heatmap()

    global StFct_ReplicaAv = reshape(sum(StFct_sum, dims=3)/replica_num, n_x, n_y)

    # Define circle parameters
    circle_center = (fft_x_center, fft_y_center)
    radius1 = n_x/(2*stripe_width) + radius_error
    radius2 = n_x/(2*stripe_width) - radius_error
    theta = LinRange(0, 2π, 100)
    circle1_x = circle_center[1] .+ radius1 .* cos.(theta)
    circle1_y = circle_center[2] .+ radius1 .* sin.(theta)
    circle2_x = circle_center[1] .+ radius2 .* cos.(theta)
    circle2_y = circle_center[2] .+ radius2 .* sin.(theta)

    q = plot(size=(500, 400), dpi=300)
    heatmap!(q, StFct_ReplicaAv, color=:viridis, framestyle=:box, alpha=1.0)
    plot!(circle1_x, circle1_y, lw=2, lc=:red, legend=false)
    plot!(circle2_x, circle2_y, lw=2, lc=:red, legend=false)
    title!("32x32 system, Temp:$Temp, pinning sites:$pinning_site_num")

    return q
end

#------------------------------------------------------------------------------------------------------------------------------#

#PLOTTING SPINS HEATMAP 
function plot_config_heatmap()

    global z_dir_sd_plot = z_dir_sd |> Array
    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)

    plot_layout = @layout [a b c d; e f g h; i j k l; m n o p; q r s t]

    p = plot(layout = plot_layout, size=(800, 1000), dpi=300)

    for replica in 1:replica_num
        
        heatmap!(p, reshape(z_dir_sd_plot[:,replica], n_x, n_y), subplot=replica, color=:grays,
                cbar=false, framestyle=:none, aspect_ratio=:equal,
                ticks=false)
        scatter!(x_pos_pinning_site[:,replica], y_pos_pinning_site[:,replica], label=false, ms=3, msw=0, subplot=replica)
    end

    return p
end

#------------------------------------------------------------------------------------------------------------------------------#

#function to label a cluster
function define_cluster_label(N_sd, replica_num)

    global largest_label = 0 |> Int32
    global cluster_label_positive = zeros(N_sd*replica_num, 1) |> Array{Int32}
    global cluster_label_negative = zeros(N_sd*replica_num, 1) |> Array{Int32}
    global trial_num = n_x
    global z_dir_sd_cluster = z_dir_sd |> Array
    
    global cluster_NN_n = NN_n |> Array{Int32}
    global cluster_NN_s = NN_s |> Array{Int32}
    global cluster_NN_e = NN_e |> Array{Int32}
    global cluster_NN_w = NN_w |> Array{Int32}

    #-----------------------------------------------------------#

    for trials in 1:trial_num
    for spins in 1:N_sd*replica_num
        if z_dir_sd_cluster[spins] == 1
            neighbor_label = [cluster_label_positive[cluster_NN_e[spins]], cluster_label_positive[cluster_NN_s[spins]], cluster_label_positive[cluster_NN_w[spins]], cluster_label_positive[cluster_NN_n[spins]]]
            if (sum(neighbor_label) == 0) && (cluster_label_positive[spins] == 0)
                largest_label += 1
                cluster_label_positive[spins] = largest_label
            else
                sort!(neighbor_label)
                for neighbor in 1:4
                    if neighbor_label[neighbor] != 0
                        cluster_label_positive[spins] = neighbor_label[neighbor]
                        break
                    end
                end
            end
        end
    end 
    end
    #-----------------------------------------------------------#

    global largest_label = 0
    
    for trials in 1:trial_num
    for spins in 1:N_sd*replica_num
        if z_dir_sd_cluster[spins] == -1
            neighbor_label = [cluster_label_negative[cluster_NN_e[spins]], cluster_label_negative[cluster_NN_s[spins]], cluster_label_negative[cluster_NN_w[spins]], cluster_label_negative[cluster_NN_n[spins]]]
            if (sum(neighbor_label) == 0) && (cluster_label_negative[spins] == 0)
                largest_label += 1
                cluster_label_negative[spins] = largest_label
            else
                sort!(neighbor_label)
                for neighbor in 1:4
                    if neighbor_label[neighbor] != 0
                        cluster_label_negative[spins] = neighbor_label[neighbor]
                        break
                    end
                end
            end
        end
    end
    end
end
#------------------------------------------------------------------------------------------------------------------------------#

#function to calculate average cluster size
function calculate_cluster_size()
    #define_cluster_label(N_sd, replica_num)

    global cluster_size_positive = Array{Int32}(undef,0)
    global cluster_label_number_positive = Array{Int32}(undef,0)
    global cluster_size_negative = Array{Int32}(undef,0)
    global cluster_label_number_negative = Array{Int32}(undef,0)

    #-----------------------------------------------------------#

    for clusters in 1:N_sd*replica_num
        count = 0
        for population in 1:N_sd*replica_num
            if clusters == cluster_label_positive[population]
                count += 1
            end
        end
        if count!=0
            push!(cluster_size_positive, count)
            push!(cluster_label_number_positive, clusters)
        end
    end
    
    number_of_clusters = length(cluster_label_number_positive)
    global cluster_label_number_positive_redefined = collect(1:number_of_clusters)

    for clusters in 1:number_of_clusters
        replace!(cluster_label_positive, cluster_label_number_positive[clusters]=>cluster_label_number_positive_redefined[clusters])
    end

    #-----------------------------------------------------------#

    for clusters in 1:N_sd*replica_num
        count = 0
        for population in 1:N_sd*replica_num
            if clusters == cluster_label_negative[population]
                count += 1
            end
        end
        if count!=0
            push!(cluster_size_negative, count)
            push!(cluster_label_number_negative, clusters)
        end
    end
    
    number_of_clusters = length(cluster_label_number_negative)
    global cluster_label_number_negative_redefined = collect(1:number_of_clusters)

    for clusters in 1:number_of_clusters
        replace!(cluster_label_negative, cluster_label_number_negative[clusters]=>cluster_label_number_negative_redefined[clusters])
    end

    global cluster_dist_positive = append!(cluster_dist_positive, cluster_size_positive)
    global cluster_dist_negative = append!(cluster_dist_negative, cluster_size_negative)

end
#------------------------------------------------------------------------------------------------------------------------------#

#function to plot cluster size
function plot_cluster_size_distribution()

    lo = @layout [a b]
    plot1 = histogram(cluster_dist_positive, 
                label="Positive clusters, Bglob$(B_global)", markerstrokewidth=0,
                xlabel="Cluster size (number of spins)", ylabel="Cluster population",
                color=:red, normalize=true, framestyle=:box)
    plot2 = histogram(cluster_dist_negative, 
                label="Negative clusters, Bglob$(B_global)", markerstrokewidth=0,
                xlabel="Cluster size (number of spins)", ylabel="Cluster population",
                color=:blue, normalize=true, framestyle=:box)

    r = plot(plot1, plot2, layout= lo, dpi=300, title="($n_x)x$(n_x) system, Temp:$(Temp)")
    return r
end

#------------------------------------------------------------------------------------------------------------------------------#

#initialize the spin matrix before starting the Monte Carlo steps
initialize_spin_config()

#In this section we change all the 2D matrices to 1D matrices.
global mx_sd = reshape(CuArray{Int32}(mx_sd), N_sd*replica_num, 1)

global z_dir_sd = reshape(z_dir_sd, N_sd*replica_num, 1) |> CuArray{Float32}

global x_pos_sd = reshape(Array{Int32}(x_pos_sd), N_sd*replica_num, 1)
global y_pos_sd = reshape(Array{Int32}(y_pos_sd), N_sd*replica_num, 1)

global NN_e = reshape(CuArray{Int32}(NN_e), N_sd*replica_num, 1)
global NN_n = reshape(CuArray{Int32}(NN_n), N_sd*replica_num, 1)
global NN_s = reshape(CuArray{Int32}(NN_s), N_sd*replica_num, 1)
global NN_w = reshape(CuArray{Int32}(NN_w), N_sd*replica_num, 1)

global rand_rep_ref_sd = CuArray{Int32}(rand_rep_ref_sd)

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO SAVE DATA
global StFct_ring = zeros(length(Temp_values), 1) |> Array{Float32}
global StFct_ring_uncertainty = zeros(length(Temp_values), 1) |> Array{Float32}

global AF_bonds_vertical_temp = zeros(length(Temp_values), 1) |> Array{Float32}
global AF_bonds_vertical_uncertainty_temp = zeros(length(Temp_values), 1) |> Array{Float32}

global AF_bonds_horizontal_temp = zeros(length(Temp_values), 1) |> Array{Float32}
global AF_bonds_horizontal_uncertainty_temp = zeros(length(Temp_values), 1) |> Array{Float32}

global AF_bonds_total_temp = zeros(length(Temp_values), 1) |> Array{Float32}
global AF_bonds_total_uncertainty_temp = zeros(length(Temp_values), 1) |> Array{Float32}

global OrientOrder_parameter_temp = zeros(length(Temp_values), 1) |> Array{Float32}
global OrientOrder_parameter_uncertainty_temp = zeros(length(Temp_values), 1) |> Array{Float32}
#------------------------------------------------------------------------------------------------------------------------------#

#define the number of Monte carlo steps to take snap shot and calculate cluster numbers
global snap_interval = 1000
global cluster_snap_interval = 10000

#define the list to collect fft heatmaps
global frames_fft = []

#define the list to collect cluster distributions
global frames_cluster = []

#main body (Monte Carlo steps)
for i in eachindex(Temp_values)

    global Temp = Temp_values[i]

    for j in 1:MC_burns
        one_MC(rng, Temp)
    end

    #define the list to collect config heatmaps
    global frames_config = []

    global cluster_dist_positive = Array{Int32}(undef,0)
    global cluster_dist_negative = Array{Int32}(undef,0)

    global StFct_sum = zeros(n_x, n_y, replica_num) |> Array{Float32}
    global AF_bonds_horizontal_sum = zeros(1, replica_num) |> Array{Float32}
    global AF_bonds_vertical_sum = zeros(1, replica_num) |> Array{Float32}
    global AF_bonds_total_sum = zeros(1, replica_num) |> Array{Float32}
    global OrientOrder_parameter_sum = zeros(1, replica_num) |> Array{Float32}

    for j in 1:MC_steps
        one_MC(rng, Temp)
        fft_calculation()
        orientational_order_parameter()
        global StFct_sum += z_dir_sd_fft
        global AF_bonds_horizontal_sum += AF_bonds_horizontal
        global AF_bonds_vertical_sum += AF_bonds_vertical
        global AF_bonds_total_sum += AF_bonds_total
        global OrientOrder_parameter_sum += O_hv

        if j%snap_interval == 0
            push!(frames_config, plot_config_heatmap())
        end

        if j%cluster_snap_interval == 0
            define_cluster_label(N_sd, replica_num)
            calculate_cluster_size()
        end
    end
    anim1 = @animate for frame in 1:length(frames_config)
        plot(frames_config[frame], dpi=300)
    end
    gif(anim1, "SD_config_T$(Temp)L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.gif", fps=5)

    global StFct_sum = StFct_sum / MC_steps
    push!(frames_fft, plot_fft_heatmap())
    StFct_ring[i], StFct_ring_uncertainty[i] = fft_intensity_ring()

    push!(frames_cluster, plot_cluster_size_distribution())

    calculate_uncertainty()
    global AF_bonds_horizontal_temp[i] = AF_bonds_horizontal_av
    global AF_bonds_horizontal_uncertainty_temp[i] = AF_bonds_horizontal_uncertainty

    global AF_bonds_vertical_temp[i] = AF_bonds_vertical_av
    global AF_bonds_vertical_uncertainty_temp[i] = AF_bonds_vertical_uncertainty

    global AF_bonds_total_temp[i] = AF_bonds_total_av
    global AF_bonds_total_uncertainty_temp[i] = AF_bonds_total_uncertainty

    global OrientOrder_parameter_temp[i] = OrientOrder_parameter_av
    global OrientOrder_parameter_uncertainty_temp[i] = OrientOrder_parameter_uncertainty
end

anim2 = @animate for frame in 1:length(frames_fft)
    plot(frames_fft[frame], dpi=300)
end
gif(anim2, "SD_FFT_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.gif", fps=2)

anim3 = @animate for frame in 1:length(frames_cluster)
    plot(frames_cluster[frame], dpi=300)
end
gif(anim3, "SD_ClusterDist_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.gif", fps=2)

open("SD_StFct_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.txt", "w") do io 					#creating a file to save data
    for i in 1:length(Temp_values)
       println(io,i,"\t", Temp_values[i],"\t", StFct_ring[i], "\t", StFct_ring_uncertainty[i],"\t",
       AF_bonds_horizontal_temp[i], "\t", AF_bonds_horizontal_uncertainty_temp[i], "\t",
       AF_bonds_vertical_temp[i], "\t", AF_bonds_vertical_uncertainty_temp[i], "\t",
       AF_bonds_total_temp[i], "\t", AF_bonds_total_uncertainty_temp[i], "\t",
       OrientOrder_parameter_temp[i], "\t", OrientOrder_parameter_uncertainty_temp[i])
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

global z_dir_sd_plot = z_dir_sd |> Array
global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)

file = open("SD_StFct_FinalSpins_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.txt", "w")

    # Iterate through the rows of the matrix
    for row in eachrow(z_dir_sd_plot)
        # Join elements with a tab delimiter and write to the file
        println(file, join(row, "\t"))
    end
    # Close the file
close(file)

if pinning_site_num!=0
    
    file = open("SD_StFct_XposPinningSite_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.txt", "w")

        # Iterate through the rows of the matrix
        for row in eachrow(x_pos_pinning_site)
            # Join elements with a tab delimiter and write to the file
            println(file, join(row, "\t"))
        end
        # Close the file
    close(file)

    file = open("SD_StFct_YposPinningSite_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.txt", "w")

        # Iterate through the rows of the matrix
        for row in eachrow(y_pos_pinning_site)
            # Join elements with a tab delimiter and write to the file
            println(file, join(row, "\t"))
        end
        # Close the file
    close(file)
end

#------------------------------------------------------------------------------------------------------------------------------#
