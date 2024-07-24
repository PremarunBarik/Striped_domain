using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions, DelimitedFiles, FFTW
ENV["GKSwstype"] = "100"
#CUDA.set_runtime_version!(v"11.7")

global rng = MersenneTwister()
global B_global = 0.0   #globally applied field on the system
#global alpha_ratio = 0.5     #defined parameter for dipolar interaction energy
global Temp = 2.8      #defined temperature of the system

#define temperature matrix with particular temperature values
#min_Temp = 
#max_Temp = 0.6
#Temp_step = 20
#Temp_interval = (max_Temp - min_Temp)/Temp_step
#Temp_values = collect(min_Temp:Temp_interval:max_Temp)
#Temp_values = reverse(Temp_values)

#Temp_values_1 = collect(1.0:0.1:1.4)
#Temp_values_2 = collect(1.5:0.05:3.5)
#Temp_values_3 = collect(2.6:0.1:4.0)
#Temp_values = vcat(Temp_values_3, Temp_values_2, Temp_values_1)

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF MC MC STEPS 
global MC_steps = 10000000
global MC_burns = 400000

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
pinning_site_num = 20

#define anisotropy energy of pinning sites
K_anisotropy = 50

#defining pinning site accross the replicas
pinning_site_pos = zeros(pinning_site_num, replica_num)

for k in 1:replica_num
    #selecting spins in random positionsin one single replica
    random_position = randperm(N_sd)
    pinning_site_pos[:,k] = random_position[1:pinning_site_num]
end

x_pos_pinning_site = zeros(pinning_site_num, replica_num)
y_pos_pinning_site = zeros(pinning_site_num, replica_num)

for k in 1:replica_num
    for i in 1:pinning_site_num
        x_pos_pinning_site[i,k] = trunc((pinning_site_pos[i,k]-1)/n_x)+1                    #10th position
        y_pos_pinning_site[i,k] = ((pinning_site_pos[i,k]-1)%n_y)+1                         #1th position
    end
end

pinning_site_pos .= pinning_site_pos .+ (N_sd*collect(0:replica_num-1))'
global pinning_site_pos = reshape(CuArray{Int64}(pinning_site_pos), pinning_site_num*replica_num, 1)

global pinning_energy = zeros(N_sd*replica_num, 1) |> CuArray
global pinning_energy[pinning_site_pos] .= K_anisotropy

#------------------------------------------------------------------------------------------------------------------------------#

#SPIN ELEMENT DIRECTION IN REPLICAS
function initialize_spin_config()

    global z_dir_sd = dipole_length*[(-1)^rand(rng, Int64) for i in 1:N_sd]
    global z_dir_sd = repeat(z_dir_sd, replica_num, 1) |> CuArray

    global z_dir_sd[pinning_site_pos] .= dipole_length
end

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
    rand_rep_ref_sd[i] = (i-1)*N_sd
end

#------------------------------------------------------------------------------------------------------------------------------#

global denom = readdlm("SD_LeknerSum_denom_L$(n_x).txt")
global denom = 2 .* denom |> CuArray

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

    global energy_tot = (((-1).*energy_exchange) .+ (dipolar_energy) .- (B_global*z_dir_sd) .- pinning_energy)

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
global glauber = CuArray(zeros(replica_num, 1))

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

#function to calculate orientational disorder
function orientational_order_parameter()

    global AF_bonds_s = abs.((z_dir_sd .* z_dir_sd[NN_s]) .- 1)/2
    global AF_bonds_n = abs.((z_dir_sd .* z_dir_sd[NN_n]) .- 1)/2
    global AF_bonds_e = abs.((z_dir_sd .* z_dir_sd[NN_e]) .- 1)/2
    global AF_bonds_w = abs.((z_dir_sd .* z_dir_sd[NN_w]) .- 1)/2

    global AF_bonds_horizontal = AF_bonds_s .+ AF_bonds_n 
    global AF_bonds_vertical = AF_bonds_e .+ AF_bonds_w
    global AF_bonds_total = AF_bonds_e .+ AF_bonds_n .+ AF_bonds_s .+ AF_bonds_w

    global AF_bonds_horizontal_sum = sum(reshape(AF_bonds_horizontal, N_sd, replica_num), dims=1) |> Array
    global AF_bonds_vertical_sum = sum(reshape(AF_bonds_vertical, N_sd, replica_num), dims=1) |> Array
    global AF_bonds_total_sum = sum(reshape(AF_bonds_total, N_sd, replica_num), dims=1) |> Array

    global O_hv = abs.(AF_bonds_horizontal_sum - AF_bonds_vertical_sum)/sum(AF_bonds_total_sum)
    return O_hv
end

#------------------------------------------------------------------------------------------------------------------------------#

#function to calculate the fourier transform of the stripes
function plot_fft()

    global z_dir_sd_plot = z_dir_sd |> Array
    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)

    global fft_replica = zeros(N_sd, replica_num) |> Array

    for replica in 1:replica_num
        spin_fft = abs.(fftshift(fft(reshape(z_dir_sd_plot[:,replica], n_x, n_y))))
        global fft_replica[:, replica] = reshape(spin_fft, N_sd, 1)
    end

    global fft_replica = reshape(fft_replica, N_sd*replica_num,1)
end

#------------------------------------------------------------------------------------------------------------------------------#

#define the circle to calculate intensity from
global fft_x_center = (trunc(n_x/2) + 1) |> Int64
global fft_y_center = (trunc(n_y/2) + 1) |> Int64

global fft_distance = sqrt.(((x_pos_sd[1:N_sd] .- fft_x_center) .^ 2 ) .+ ((y_pos_sd[1:N_sd] .- fft_y_center) .^ 2))

global stripe_width = 4
global circle_radius = n_x/(2*stripe_width)   #stripe width depend on the exchange to dipolar interaction coefficient ratio 
                                            #If J= 6.0, stripe width is 4, if J=8.9 stripe width is 8
global radius_error = 1.0                   #depending on the choice of this we can define how thick the concentric
                                            #circle would be on the fourier plot
global fft_cell_list = Int64[]              #list of matrix cells where we collect the intensities from the Fourier plots

for i in 1:N_sd
    if fft_distance[i]<= (circle_radius + radius_error) && fft_distance[i] >= (circle_radius - radius_error)
        append!(fft_cell_list, i)
    end
end

global num_of_cells = length(fft_cell_list)
global fft_cell_list = fft_cell_list .+ (N_sd .* collect(0:replica_num-1))'
#global fft_cell_list = reshape(fft_cell_list, num_of_cells*replica_num, 1)

#function to calculate the intensity from Fourier transform
function fft_intensity_calculation()
    global z_dir_sd_plot = z_dir_sd |> CuArray
    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)

    global fft_replica = zeros(N_sd, replica_num) |> CuArray

    for replica in 1:replica_num
        spin_fft = abs.(fftshift(fft(reshape(z_dir_sd_plot[:,replica], n_x, n_y))))
        global fft_replica[:, replica] = reshape(spin_fft, N_sd, 1)
    end

    global fft_intensity = fft_replica[fft_cell_list]
    global fft_intensity = sum(fft_intensity, dims=1) |> Array

    return fft_intensity
end

function fft_intensity_alternative()
    global z_dir_sd_fft = reshape(z_dir_sd, n_x, n_y, replica_num) |> Array
    global z_dir_sd_fft = fftshift.(fft.(eachslice(z_dir_sd_fft, dims=3)))
    global z_dir_sd_fft = abs.(reduce((x,y) -> cat(x, y, dims=3), z_dir_sd_fft))
    global z_dir_sd_fft_av = sum(z_dir_sd_fft, dims=1)
    global z_dir_sd_fft_av = sum(z_dir_sd_fft_av, dims=2)/N_sd
    global z_dir_sd_fft = z_dir_sd_fft ./ z_dir_sd_fft_av

    global fft_intensity = z_dir_sd_fft[fft_cell_list] 
    global fft_intensity = sum(fft_intensity, dims=1)/num_of_cells |> Array

end
#------------------------------------------------------------------------------------------------------------------------------#

#initialize the spin matrix before starting the Monte Carlo steps
initialize_spin_config()

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

#MATRIX TO SAVE DATA
#global Order_parameter = zeros(length(Temp_values), 1) |> Array
global fft_intensity_MCsteps = zeros((MC_steps/1000 |> Int64), replica_num) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#define the number of Monte Carlo steps to average the fourier intensity over
global bin_window = 1000

#main body (Monte Carlo steps)

for j in 1:MC_burns
    one_MC(rng, Temp)
end

for i in 1:(MC_steps/bin_window |> Int64)

    global fft_intensity_sum = zeros(1, replica_num) |> Array

    for j in 1:bin_window
        one_MC(rng, Temp)    #32x32 - 10replicas: 1.8ms - 20replicas: 3.37ms
                             #64x64 - 10replicas: 16.3ms - 20replicas:21.1ms
                             #128x128 - 1replica: 18.91ms - 5 replicas: 67.25ms (maximum can be fitted into one GPU) 
        global fft_intensity_sum += fft_intensity_calculation()
    end

    global fft_intensity_MCsteps[i,:] = fft_intensity_sum/1000
end

#------------------------------------------------------------------------------------------------------------------------------#

writedlm("SD_FFTintensity_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num).txt", fft_intensity_MCsteps)

#scatter(Temp_values, Order_parameter, ms=2, msw =0, framestyle=:box, label="h: 0.0")
#plot!(Temp_values, Order_parameter, lw=1, label=false,
#    tickfont=font(12), legendfont=font(12), guidefont=font(12),
#    xlabel="Temperature (T)", ylabel="Oreder parameter (O_hv)")
#title!("Orientational order parameter Vs Temp")

#savefig("OrientationalOrderParamater_L$(n_x)_alpha$(alpha_ratio)_h$(B_global).png")

#------------------------------------------------------------------------------------------------------------------------------#
global bin_intensity = fft_intensity_MCsteps

#define the lagbins to calculate autocorrelation
global delay_times = collect(1:1000)

#matrix to store correlation data
global auto_correlation = zeros(length(delay_times), 1) |> Array
global auto_correlation_replica = zeros(length(delay_times), replica_num) |> Array

#calculation of correlation for the mentioned lagbins
for delay in eachindex(delay_times)
    global delay_time = delay_times[delay]

    global zero_mx = zeros(delay_time, replica_num) |> Array

    global bin_intensity_i = vcat(zero_mx, bin_intensity) |> Array
    global bin_intensity_j = vcat(bin_intensity, zero_mx) |> Array

    global cross_intensity = bin_intensity_i .* bin_intensity_j

    global denom = sum(bin_intensity[1:end-delay_time,:], dims=1) .* sum(bin_intensity[delay_time:end, :], dims=1)

    global g2 = (sum(cross_intensity, dims=1) ./ denom) * (length(bin_intensity[:,1])-delay_time)
    global auto_correlation[delay] = sum(g2)/replica_num
    global auto_correlation_replica[delay,:] = g2 

    println(delay_time)
end

open("SD_autocorrelation_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num).txt", "w") do io 					#creating a file to save data
    for i in 1:length(Temp_values)
       println(io,i,"\t", delay_times[i],"\t", auto_correlation[i])
    end
end

writedlm("SD_autocorrelationReplica_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num).txt", auto_correlation_replica)

#------------------------------------------------------------------------------------------------------------------------------#

#PLOTTING SPINS HEATMAP 
#function plot_final_config_heatmap()

#    global z_dir_sd_plot = z_dir_sd |> Array
#    global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)


#    anim_2 = @animate for replica in 1:replica_num
#        heatmap(reshape(z_dir_sd_plot[:,replica], n_x, n_y), color=:grays, cbar=false, xticks=false, yticks=false, framestyle=:box, size=(400,400))
#        scatter!(x_pos_pinning_site[:,replica], y_pos_pinning_site[:,replica], label=false, ms= 5, msw=0)
#        title!("Temp:$Temp, J:$J_NN, h:$B_global, replica:$replica")
#    end

#    gif(anim_2, "SD_FinalConfigL$(n_x)J$(J_NN)Temp$(Temp)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num).gif", fps=2)

#end

#plot_final_config_heatmap()

#------------------------------------------------------------------------------------------------------------------------------#

global z_dir_sd_plot = z_dir_sd |> Array
global z_dir_sd_plot = reshape(z_dir_sd_plot, N_sd, replica_num)


file = open("SD_final_spins_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num).txt", "w")

    # Iterate through the rows of the matrix
    for row in eachrow(z_dir_sd_plot)
        # Join elements with a tab delimiter and write to the file
        println(file, join(row, "\t"))
    end

    # Close the file
close(file)

file = open("SD_XposPinningSite_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num).txt", "w")

    # Iterate through the rows of the matrix
    for row in eachrow(x_pos_pinning_site)
        # Join elements with a tab delimiter and write to the file
        println(file, join(row, "\t"))
    end

    # Close the file
close(file)

file = open("SD_YposPinningSite_L$(n_x)J$(J_NN)B$(B_global)psNum$(pinning_site_num)repNum$(replica_num).txt", "w")

    # Iterate through the rows of the matrix
    for row in eachrow(y_pos_pinning_site)
        # Join elements with a tab delimiter and write to the file
        println(file, join(row, "\t"))
    end

    # Close the file
close(file)

#------------------------------------------------------------------------------------------------------------------------------#

