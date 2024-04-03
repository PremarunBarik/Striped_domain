using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions, DelimitedFiles, FFTW
ENV["GKSwstype"] = "100"

rng = MersenneTwister()
global B_global = 0.0   #globally applied field on the system
global alpha_ratio = 0.6     #defined parameter for dipolar interaction energy
global Temp = 0.3      #defined temperature of the system

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF MC MC STEPS 
global MC_steps = 100000
global MC_burns = 100000

#list of delay times
global delay_times = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

#NUMBER OF LOCAL MOMENTS
n_x = 16
n_y = 16
n_z = 1

N_sd = n_x*n_y

#NUMBER OF REPLICAS 
replica_num = 10

#define interaction coefficients of NN and NNN interactions
global J_NN = 1.0

#LENGTH OF DIPOLE 
dipole_length = 1

#SPIN ELEMENT DIRECTION IN REPLICAS
global z_dir_sd = dipole_length*[1 for i in 1:N_sd]
global z_dir_sd = repeat(z_dir_sd, replica_num,1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sd = Array(collect(1:N_sd*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY -- needed to define neighbours and to 
#plot the spin configuration. So, we don't need to create an Array of these matrices 
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
    rand_rep_ref_sd[i] = (i-1)*N_sd
end

#------------------------------------------------------------------------------------------------------------------------------#
#In this section we change all the 2D matrices to 1D matrices.

mx_sd = reshape(CuArray{Int64}(mx_sd), N_sd*replica_num, 1)

#z_dir_sd = reshape(z_dir_sd, N_sd*replica_num, 1)

x_pos_sd = reshape(CuArray{Int64}(x_pos_sd), N_sd*replica_num, 1)
y_pos_sd = reshape(CuArray{Int64}(y_pos_sd), N_sd*replica_num, 1)

NN_e = reshape(CuArray{Int64}(NN_e), N_sd*replica_num, 1)
NN_n = reshape(CuArray{Int64}(NN_n), N_sd*replica_num, 1)
NN_s = reshape(CuArray{Int64}(NN_s), N_sd*replica_num, 1)
NN_w = reshape(CuArray{Int64}(NN_w), N_sd*replica_num, 1)

rand_rep_ref_sd = CuArray{Int64}(rand_rep_ref_sd)

#------------------------------------------------------------------------------------------------------------------------------#

global denom = readdlm("SD_LeknerSum_denom_L$(n_x).txt") |> CuArray

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
    energy_x = z_dir_sd.*(J_NN .* (z_dir_sd[NN_n] .+ z_dir_sd[NN_s] .+ z_dir_sd[NN_e] .+ z_dir_sd[NN_w]))

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

#MATRIX TO STORE DELTA ENERGY
global glauber = Array(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#function to flip a spin using KMC subroutine
function one_MC_kmc(rng, N_sd, replica_num, Temp)
    compute_tot_energy_spin_glass()

    trans_rate = exp.(-energy_tot/Temp)
    global glauber = trans_rate./(1 .+ trans_rate)
    loc = reshape(mx_sd, (N_sd,replica_num)) |> Array

    for k in 1:replica_num
        loc[:,k] = shuffle!(loc[:,k])
    end

    glauber_cpu = glauber |> Array
    trans_prob = glauber_cpu[loc] |> Array
    trans_prob_ps = cumsum(trans_prob, dims=1)

    for k in 1:replica_num
        chk = rand(rng, Float64)*trans_prob_ps[N_sd,k]
        for l in 1:N_sd
            if chk <= trans_prob_ps[l,k]
                z_dir_sd[loc[l,k]] = (-1)*z_dir_sd[loc[l,k]]
                global flip_count[loc[l,k]] = flip_count[loc[l,k]] + 1
            break
            end
        end
    end

end

#------------------------------------------------------------------------------------------------------------------------------#

#PLOTTING SPINS HEATMAP 
function plot_final_config_heatmap()

    global z_dir_sd_plot = z_dir_sd |> Array
    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)


    anim_2 = @animate for replica in 1:replica_num
        heatmap(reshape(z_dir_sd_plot[:,replica], n_x, n_y), color=:grays, cbar=false, xticks=false, yticks=false, framestyle=:box, size=(400,400))
        title!("Temp:$Temp, J:$J_NN, h:$B_global, replica:$replica")
    end

    gif(anim_2, "SD_final_config_L$(n_x)_J$(J_NN)_Temp$(Temp)_B$(B_global).gif", fps=2)

end

#------------------------------------------------------------------------------------------------------------------------------#

#plotting fourier transform
function plot_fft_heatmap()

    global z_dir_sd_plot = z_dir_sd |> Array
    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)

    global spin_fft = zeros(n_x, n_y, replica_num) |> Array

    for replica in 1:replica_num
        global spin_fft[:,:,replica_num] = abs.(fftshift(fft(reshape(z_dir_sd_plot[:,replica], n_x, n_y))))
    end
    global spin_fft = spin_fft/maximum(spin_fft)
    global spin_fft = sum(spin_fft, dims=3)/replica_num
    

    heatmap(spin_fft[:,:,1], color=:viridis, framestyle=:box, clim=(0,1), size=(400,400), xtickfont=font(12), ytickfont=font(12))
    title!("Temp:$Temp, J:$J_NN, h:$B_global")
end

#------------------------------------------------------------------------------------------------------------------------------#

#plotting fourier transform
function plot_final_fft_replica_heatmap()

    global z_dir_sd_plot = z_dir_sd |> Array
    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)

    global spin_fft = zeros(n_x, n_y, replica_num) |> Array
    
    for replica in 1:replica_num
        global spin_fft[:,:,replica] = abs.(fftshift(fft(reshape(z_dir_sd_plot[:,replica], n_x, n_y))))
    end
    
    global spin_fft = spin_fft/maximum(spin_fft)

    anim_3 = @animate for replica in 1:replica_num
        heatmap(spin_fft[:,:,replica], color=:viridis, framestyle=:box, clim=(0,1), size=(400,400), xtickfont=font(12), ytickfont=font(12))
    	title!("Temp:$Temp, J:$J_NN, h:$B_global, replica:$replica")
    end
    
    gif(anim_3, "SD_final_fft_replica_L$(n_x)_J$(J_NN)_B$(B_global).gif", fps=1)

end

#------------------------------------------------------------------------------------------------------------------------------#
#PLOTTING SPINS HEATMAP 
function plot_heatmap()

    heatmap(reshape(z_dir_sd, n_x, n_y), color=:grays, cbar=false, alpha=0.5, xticks=false, yticks=false, framestyle=:box, size=(400,400))
    title!("Temp:$Temp, alpha:$alpha_ratio, h:$B_global")

end

#------------------------------------------------------------------------------------------------------------------------------#
#PLOTTING Fourier transform HEATMAP 
function plot_fft_heatmap()

    heatmap(abs.(fftshift(fft(reshape(z_dir_sd, n_x, n_y)))), color=:viridis, alpha=0.5, xticks=false, yticks=false, framestyle=:box, size=(400,400))
    title!("Temp:$Temp, alpha:$alpha_ratio, h:$B_global")

end

#------------------------------------------------------------------------------------------------------------------------------#
function calculate_g2_point()

    global replica_t = reshape(replica_t, N_sd, replica_num)
    global replica_delt = reshape(replica_delt, N_sd, replica_num)

    global fft_replica_t = zeros(n_x, n_y, replica_num) |> Array
    global fft_replica_delt = zeros(n_x,n_y, replica_num) |> Array

    for replica in 1: replica_num

        #fourier transform of the replicas a time step apart
        global fft_replica_t[:,:,replica]= abs.(fftshift(fft(reshape(replica_t[:,replica], n_x, n_y))))
        global fft_replica_delt[:,:,replica] = abs.(fftshift(fft(reshape(replica_delt[:,replica], n_x, n_y))))

    end

    global cross_intensity_mx += (fft_replica_t .* fft_replica_delt) 
    global self_intensity_mx += fft_replica_t 

end

#------------------------------------------------------------------------------------------------------------------------------#
#matrix to store g2 data
global g2_correlation = zeros(length(delay_times), 1)

#------------------------------------------------------------------------------------------------------------------------------#
#MAIN BODY
#global replica_t = z_dir_sd
for delay in eachindex(delay_times)

    #SPIN ELEMENT DIRECTION IN REPLICAS
    global z_dir_sd = dipole_length*[(-1)^rand(rng, Int64) for i in 1:N_sd]
    global z_dir_sd = repeat(z_dir_sd, replica_num, 1) |> CuArray

    global cross_intensity_mx = zeros(n_x, n_y, replica_num) |> Array
    global self_intensity_mx = zeros(n_x, n_y, replica_num) |> Array

    for snaps in 1:(MC_steps/delay_times[delay] |>Int64)

        global replica_t = z_dir_sd[1:N_sd*replica_num] |> Array

        for j in 1:delay_times[delay]
            one_MC(rng, Temp)                                                     #MONTE CARLO FUNCTION 
        end
#       plot_heatmap()

        global replica_delt = z_dir_sd[1:N_sd*replica_num] |> Array
        
        calculate_g2_point()
    end

    global g2_value = (cross_intensity_mx/(MC_steps/delay_times[delay] |>Int64)) ./ ((self_intensity_mx/(MC_steps/delay_times[delay] |>Int64)) .^ 2)
    global g2_value = sum(g2_value, dims=2)
    global g2_value = sum(g2_value, dims=1)/(n_x*n_y)

    global g2_correlation[delay] = sum(g2_value)/replica_num
    
end

#gif(anim, "SD_config_L$(n_x)_alpha$(alpha_ratio)_T$(Temp).gif", fps=5)

#------------------------------------------------------------------------------------------------------------------------------#
#plotting the autocorrelation data 
plot(delay_times, g2_correlation, linewidth=2, framestyle=:box, size=(400,400), 
    xaxis="delay time", yaxis="g2", xtickfont=font(12), ytickfont=font(12), bg=:transparent,
    guidefont=font(12), legend=false, title="T:$(Temp), alpha:$(alpha_ratio)")

savefig("AutoCorrelation_L$(n_x)_alpha$(alpha_ratio)_T$(Temp)_replica$(replica_num).png")

#------------------------------------------------------------------------------------------------------------------------------#

#SAVING THE GENERATED DATA
open("AutoCorrelation_L$(n_x)_alpha$(alpha_ratio)_T$(Temp)_replica$(replica_num).txt", "w") do io 					#creating a file to save data
    for i in 1:length(g2_correlation)
       println(io,i,"\t",delay_times[i],"\t", g2_correlation[i])
    end
end
