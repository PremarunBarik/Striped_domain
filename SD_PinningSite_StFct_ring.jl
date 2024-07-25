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
pinning_site_num = 40

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
function plot_fft()

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

    heatmap(StFct_ReplicaAv, color=:viridis, framestyle=:box, size=(500,400),
            dpi=300, alpha=1.0)
    plot!(circle1_x, circle1_y, lw=2, lc=:red, legend=false)
    plot!(circle2_x, circle2_y, lw=2, lc=:red, legend=false)
    title!("32x32 system, Temp:$Temp, pinning sites:$pinning_site_num")
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

#------------------------------------------------------------------------------------------------------------------------------#

#main body (Monte Carlo steps)
anim = @animate for i in eachindex(Temp_values)

    global Temp = Temp_values[i]

    for j in 1:MC_burns
        one_MC(rng, Temp)
    end

    global StFct_sum = zeros(n_x, n_y, replica_num) |> Array{Float32}

    for j in 1:MC_steps
        one_MC(rng, Temp)
        global StFct_sum += fft_calculation()
    end

    global StFct_sum = StFct_sum / MC_steps
    plot_fft()

    StFct_ring[i], StFct_ring_uncertainty[i] = fft_intensity_ring()
end

gif(anim, "SD_FFT_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.gif", fps=2)

open("SD_StFct_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num)_ring.txt", "w") do io 					#creating a file to save data
    for i in 1:length(Temp_values)
       println(io,i,"\t", Temp_values[i],"\t", StFct_ring[i], "\t", StFct_ring_uncertainty[i])
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
