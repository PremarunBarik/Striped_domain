using Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions
ENV["GKSwstype"] = "100"

rng = MersenneTwister()
global B_global = 0.1   #globally applied field on the system
global alpha_ratio = 2.0     #defined parameter for dipolar interaction energy
global Temp = 0.35      #defined temperature of the system

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF MC MC STEPS 
MC_steps = 50000

#NUMBER OF LOCAL MOMENTS
n_x = 20
n_y = 20
n_z = 1

N_sd = n_x*n_y

#NUMBER OF REPLICAS 
replica_num = 1

#define interaction co-efficients of NN and NNN interactions
global J_NN = 0.5

#LENGTH OF DIPOLE 
dipole_length = 1

#SPIN ELEMENT DIRECTION IN REPLICAS
z_dir_sd = dipole_length*[(1)^rand(rng, Int64) for i in 1:N_sd]
z_dir_sd = repeat(z_dir_sd, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sd = Array(collect(1:N_sd*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY -- needed to define neighbors and to 
#plot the spin configuration. So, we don't need to create a Array of these matrices 
#also no need to repeat for replicas because spin positions are constant over replicas 

x_pos_sd = zeros(N_sd, 1)
y_pos_sd = zeros(N_sd, 1)


for i in 1:N_sd
    x_pos_sd[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sd[i] = ((i-1)%n_y)+1                         #1th position
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

mx_sd = reshape(Array{Int64}(mx_sd), N_sd*replica_num, 1)

x_pos_sd = reshape(Array{Int64}(x_pos_sd), N_sd*replica_num, 1)
y_pos_sd = reshape(Array{Int64}(y_pos_sd), N_sd*replica_num, 1)

NN_e = reshape(Array{Int64}(NN_e), N_sd*replica_num, 1)
NN_n = reshape(Array{Int64}(NN_n), N_sd*replica_num, 1)
NN_s = reshape(Array{Int64}(NN_s), N_sd*replica_num, 1)
NN_w = reshape(Array{Int64}(NN_w), N_sd*replica_num, 1)

rand_rep_ref_sd = Array{Int64}(rand_rep_ref_sd)

#------------------------------------------------------------------------------------------------------------------------------#

global kizzie = (x_pos_sd .- x_pos_sd')/n_x
global eta = (y_pos_sd .- y_pos_sd')/n_y

global l_list = collect(1:10)
global m_list = collect(-10:1:10)

global denom = zeros(N_sd, N_sd)

for i in 1:N_sd
    for j in 1:N_sd
        if (eta[i,j]==0)
            if (kizzie[i,j]==0)
                global denom[i,j] = 4*zeta(3/2)*(zeta(3/2, 1/4) - zeta(3/2, 3/4)) / (4^(3/2))/(n_x^3)
            else                
                global s = 0.0
                for m in 1:length(m_list)
                    for l in 1:length(l_list)
                        global s += l_list[l]*cos(2*pi*l_list[l]*eta[i,j])*abs(1/(kizzie[i,j] + m_list[m]))*besselk(1, (2*pi*l_list[l]*abs(kizzie[i,j] + m_list[m])))
                    end
                end
                global denom[i,j] = ((2*pi^2)/((sin(pi*kizzie[i,j]))^2) + (8*pi*s))/(n_x^3)
            end      
        else
            global s = 0.0
            for m in 1:length(m_list)
                for l in 1:length(l_list)
                    global s += l_list[l]*cos(2*pi*l_list[l]*kizzie[i,j])*abs(1/(eta[i,j] + m_list[m]))*besselk(1, (2*pi*l_list[l]*abs(eta[i,j] + m_list[m])))
                end
            end
            global denom[i,j] = ((2*pi^2)/((sin(pi*eta[i,j]))^2) + (8*pi*s))/(n_x^3)
        end
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#compute dipolar energy of the system
function compute_dipolar_energy()
    global dipolar_energy = (z_dir_sd .* z_dir_sd') .* denom

    global dipolar_energy = sum(dipolar_energy, dims=2)
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO EXCHANGE 
global energy_exchange = zeros(N_sd*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE EXCHANGE ENERGY OF THE SYSTEM
function compute_exchange_energy()
    energy_x = z_dir_sd.*(J_NN .* (z_dir_sd[NN_n] .+ z_dir_sd[NN_s] .+ z_dir_sd[NN_e] .+ z_dir_sd[NN_w]))

    global energy_exchange = energy_x
end

#------------------------------------------------------------------------------------------------------------------------------#
#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sd*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_tot_energy_spin_glass()
    compute_exchange_energy()
    compute_dipolar_energy()

    global energy_tot = (energy_exchange .- (alpha_ratio .* dipolar_energy) .+ (B_global*z_dir_sd))

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = zeros(replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global rand_pos =  Array(rand(rng, (1:N_sd), (replica_num, 1)))
    global r = rand_pos .+ rand_rep_ref_sd

    global del_energy = 2*energy_tot[r]

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
    global rand_num_flip = Array(rand(rng, Float64, (replica_num, 1)))
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
function plot_heatmap()

    heatmap(reshape(z_dir_sd, n_x, n_y), color=:grays, cbar=false, xticks=false, yticks=false, framestyle=:box, size=(400,400))
    title!("Temp:$Temp, alpha:$alpha_ratio, h:$B_global")

end

#------------------------------------------------------------------------------------------------------------------------------#
#MAIN BODY
#anim = @animate for snaps in 1:(MC_steps/1000 |>Int64)
#    for j in 1:1000
#        one_MC(rng, Temp)                                                     #MONTE CARLO FUNCTION 
#    end
#    plot_heatmap()
#end

#gif(anim, "SD_config_alpha$(alpha_ratio)_Temp$(Temp)_B$(B_global).gif", fps=5)


#------------------------------------------------------------------------------------------------------------------------------#
