using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions, DelimitedFiles, FFTW
ENV["GKSwstype"] = "100"
CUDA.set_runtime_version!(v"11.7")

global rng = MersenneTwister()
global B_global = 0.0   #globally applied field on the system
global alpha_ratio = 0.6     #defined parameter for dipolar interaction energy
global Temp = 0.4      #defined temperature of the system

#define temperature matrix with particular temperature values
#min_Temp = 1.2
#max_Temp = 3.0
#Temp_step = 40
#global Temp = 2.2
#Temp_interval = (max_Temp - min_Temp)/Temp_step
#Temp_values = collect(min_Temp:Temp_interval:max_Temp)
#Temp_values = reverse(Temp_values)

#list of delay times
delay_times_1 = collect(1:9)
delay_times_2 = Array{Int64}(reshape(collect(1:0.5:9.5) .* (10 .^ collect(1:4))', 18*4, 1))
global delay_times = vcat(delay_times_1, delay_times_2)

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF MC MC STEPS 
global MC_steps = 100000
global MC_burns = 100000

#NUMBER OF LOCAL MOMENTS
n_x = 16
n_y = 16
n_z = 1

N_sd = n_x*n_y

#NUMBER OF REPLICAS 
replica_num = 20

#define interaction co-efficients of NN and NNN interactions
global J_NN = 1.0

#LENGTH OF DIPOLE 
dipole_length = 1

#SPIN ELEMENT DIRECTION IN REPLICAS
global z_dir_sd = dipole_length*[(-1)^rand(rng, Int64) for i in 1:N_sd]
global z_dir_sd = repeat(z_dir_sd, replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sd = Array(collect(1:N_sd*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY -- needed to define neighbors and to 
#plot the spin configuration. So, we don't need to create a Array of these matrices 
#also no need to repeat for replicas because spin positions are constant over replicas 

x_pos_sd = zeros(N_sd, replica_num)
y_pos_sd = zeros(N_sd, replica_num)

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
    global rand_rep_ref_sd[i] = (i-1)*N_sd
end

#------------------------------------------------------------------------------------------------------------------------------#
#In this section we change all the 2D matrices to 1D matrices.

global mx_sd = reshape(CuArray{Int64}(mx_sd), N_sd*replica_num, 1)

global z_dir_sd = reshape(z_dir_sd, N_sd*replica_num, 1) |> CuArray

global x_pos_sd = reshape(Array{Int64}(x_pos_sd), N_sd*replica_num, 1)
global y_pos_sd = reshape(Array{Int64}(y_pos_sd), N_sd*replica_num, 1)

global NN_e = reshape(CuArray{Int64}(NN_e), N_sd*replica_num, 1)
global NN_n = reshape(CuArray{Int64}(NN_n), N_sd*replica_num, 1)
global NN_s = reshape(CuArray{Int64}(NN_s), N_sd*replica_num, 1)
global NN_w = reshape(CuArray{Int64}(NN_w), N_sd*replica_num, 1)

global rand_rep_ref_sd = CuArray{Int64}(rand_rep_ref_sd)

#------------------------------------------------------------------------------------------------------------------------------#

global denom = readdlm("SD_LeknerSum_denom_L$(n_x).txt") |> CuArray
#global denom =  denom |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#compute dipolar energy of the system
function compute_dipolar_energy()
    z_dir_sd_i = reshape(z_dir_sd, N_sd, 1, replica_num)
    z_dir_sd_j = reshape(z_dir_sd, 1, N_sd, replica_num)

    global dipolar_energy = (z_dir_sd_i .* z_dir_sd_j) .* denom
    global dipolar_energy = sum(dipolar_energy, dims = 2)


    global dipolar_energy = reshape(dipolar_energy, N_sd*replica_num, 1)
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO EXCHANGE 
global energy_exchange = zeros(N_sd*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE EXCHANGE ENERGY OF THE SYSTEM
function compute_exchange_energy()
    global energy_x = z_dir_sd.*(J_NN .* (z_dir_sd[NN_n] .+ z_dir_sd[NN_s] .+ z_dir_sd[NN_e] .+ z_dir_sd[NN_w]))

    global energy_exchange = energy_x
end

#------------------------------------------------------------------------------------------------------------------------------#
#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sd*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_tot_energy_spin_glass()
    compute_exchange_energy()
    compute_dipolar_energy()

    global energy_tot = (((-1).*energy_exchange) .+ (alpha_ratio .* dipolar_energy) .- (B_global*z_dir_sd))

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = zeros(replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global rand_pos =  CuArray(rand(rng, (1:N_sd), (replica_num, 1)))
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
    global rand_num_flip = CuArray(rand(rng, Float64, (replica_num, 1)))
    flipit = sign.(rand_num_flip .- trans_rate)
    global z_dir_sd[r] = flipit.*z_dir_sd[r]

#    flipit = (abs.(flipit .- 1))/2
#    global flip_count[r] = flip_count[r] .+ flipit
end

#------------------------------------------------------------------------------------------------------------------------------#
#PLOTTING SPINS HEATMAP 
function plot_config_heatmap()

    global z_dir_sd_plot = z_dir_sd |> Array
    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)


    anim = @animate for replica in 1:replica_num
        heatmap(reshape(z_dir_sd_plot[:,replica], n_x, n_y), color=:grays, cbar=false, xticks=false, yticks=false, framestyle=:box, size=(400,400))
        title!("Temp:$Temp, J:$J_NN, replica:$replica")
    end
    gif(anim, "SD_ConfigAfterBurnSteps_L$(n_x)_alpha$(alpha_ratio)_Temp$(Temp)_Replica$(replica_num).gif", fps = 2)
end

#------------------------------------------------------------------------------------------------------------------------------#

#plotting fourier transform
function calculate_fft()

    global z_dir_sd_plot = z_dir_sd |> Array
    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)

    global fft_replica = zeros(N_sd, replica_num) |> Array

    for replica in 1:replica_num
        spin_fft = abs.(fftshift(fft(reshape(z_dir_sd_plot[:,replica], n_x, n_y))))
        global fft_replica[:, replica] = reshape(spin_fft, N_sd, 1)
    end

    return fft_replica
end

#------------------------------------------------------------------------------------------------------------------------------#
#auto-correlation function 
function correlation()
    global cross_intensity_mx += mx_t .* mx_delt
    global self_intensity_mx += mx_t
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STORING DATA
global FFT_MCSteps = zeros(N_sd, replica_num, MC_steps) |> Array

#------------------------------------------------------------------------------------------------------------------------------#
#MAIN BODY
#MC burn steps
for burn in 1:MC_burns
    one_MC(rng, Temp)
end

plot_config_heatmap()

#MC steps
for steps in 1:MC_steps
    one_MC(rng, Temp)                                                     #MONTE CARLO FUNCTION 
    global FFT_MCSteps[:,:,steps] = calculate_fft()
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STORING DATA
global auto_correlation = zeros(length(delay_times), 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#
#calculating Auto-correlation
for delay in eachindex(delay_times)

    global interval_count = 0

    global cross_intensity_mx = zeros(N_sd,replica_num)
    global self_intensity_mx = zeros(N_sd,replica_num)

    for i in 1:MC_steps
        global mx_t = FFT_MCSteps[:,:,i]

        if ((i+delay_times[delay]) <= MC_steps)
            global mx_delt = FFT_MCSteps[:,:, i+delay_times[delay]]
            correlation()

            global interval_count += 1
        end
    end
    #println(interval_count)
    global g2_value = (cross_intensity_mx/interval_count) ./ ((self_intensity_mx/interval_count) .^ 2)
    global g2_value = sum(g2_value, dims=1)/N_sd
    global auto_correlation[delay] = sum(g2_value)/replica_num
end

plot(delay_times, auto_correlation, framestyle=:box, xscale=:log10, linewidth=2, xlabel="Time delay", ylabel="g2", legend=false)
title!("Auto-correlation vs delay time")

savefig("AC_SD_L$(n_x)_alpha$(alpha_ratio)_Temp$(Temp)_Replica$(replica_num).png")

#------------------------------------------------------------------------------------------------------------------------------#

#SAVING THE GENERATED DATA
open("AC_SD_L$(n_x)_alpha$(alpha_ratio)_Temp$(Temp)_Replica$(replica_num).txt", "w") do io 					#creating a file to save data
    for i in 1:length(delay_times)
       println(io,i,"\t", delay_times[i],"\t", auto_correlation[i])
    end
 end
