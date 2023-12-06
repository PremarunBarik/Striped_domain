using Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions

rng = MersenneTwister()
global B_global = 0.0   #globally applied field on the system
global alpha = 0.35     #defined parameter for dipolar interaction energy
global Temp = 0.35      #defined temperature of the system

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF MC MC STEPS 
MC_steps = 50000
MC_burns = 50000

#NUMBER OF LOCAL MOMENTS
n_x = 10
n_y = 10
n_z = 1

N_sd = n_x*n_y

#NUMBER OF REPLICAS 
replica_num = 1

#SPIN ELEMENT DIRECTION IN REPLICAS
z_dir_sd = [(1)^rand(rng, Int64) for i in 1:N_sd]
z_dir_sg = repeat(z_dir_sd, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sd = Array(collect(1:N_sd*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY -- needed to define neighbors and to 
#plot the spin configuration. So, we don't need to create a Array of these matrices 
#also no need to repeat for replicas because spin positions are constant over replicas 

x_pos_sd = zeros(N_sd, 1)
y_pos_sd = zeros(N_sd, 1)
z_pos_sd = fill(n_z, N_sd)

for i in 1:N_sd
    x_pos_sd[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sd[i] = ((i-1)%n_y)+1                         #1th position
end

#------------------------------------------------------------------------------------------------------------------------------#

#ISING NEAR NEIGHBOUR CALCULATION
NN_s = zeros(N_sd,1)
NN_n = zeros(N_sd,1)
NN_e = zeros(N_sd,1)
NN_w = zeros(N_sd,1)

for i in 1:N_sd                             #loop over all the spin ELEMENTS
        if x_pos_sd[i]%n_x == 0
            r_e =  (x_pos_sd[i]-n_x)*n_x + y_pos_sd[i]
        else
            r_e =  x_pos_sd[i]*n_x + y_pos_sd[i]
        end
        NN_e[i] = r_e
        #-----------------------------------------------------------#
        if x_pos_sd[i]%n_x == 1
            r_w = (x_pos_sd[i]+n_x-2)*n_x + y_pos_sd[i]
        else
            r_w = (x_pos_sd[i]-2)*n_x + y_pos_sd[i]
        end
        NN_w[i] = r_w
        #-----------------------------------------------------------#
        if y_pos_sd[i]%n_y == 0
            r_n =  (x_pos_sd[i]-1)*n_x + (y_pos_sd[i]-n_y+1)
        else
            r_n = (x_pos_sd[i]-1)*n_x + y_pos_sd[i]+1
        end
        NN_n[i] = r_n
        #-----------------------------------------------------------#
        if y_pos_sd[i]%n_y == 1
            r_s = (x_pos_sd[i]-1)*n_x + (y_pos_sd[i]+n_y-1)
        else
            r_s = (x_pos_sd[i]-1)*n_x + y_pos_sd[i]-1
        end
        NN_s[i] = r_s
end

NN_s = repeat(NN_s, replica_num, 1)
NN_n = repeat(NN_n, replica_num, 1)
NN_e = repeat(NN_e, replica_num, 1)
NN_w = repeat(NN_w, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#INTERACTION COEFFICIENT MATRIX
J_NN = zeros(N_sd,N_sd,replica_num)

for i in 1:N_sd
    for j in i:N_sd
        for k in 1:replica_num
            if i==j
                continue
            else
                J_NN[i,j,k] = J_NN[j,i,k] = 1       #1 because in striped domains exchange interactions are ferromagnetic
            end
        end
    end
end


#------------------------------------------------------------------------------------------------------------------------------#

#REPLICA REFERENCE MATRIX OF SPIN ELEMENTS
spin_rep_ref = zeros(N_sd*replica_num,1)

for i in eachindex(spin_rep_ref)
    spin_rep_ref[i] = trunc((i-1)/N_sd)*N_sd
end

#REPLICA REFERENCE MATRIX OF RANDOMLY SELECTED SPIN IN MC_STEP
rand_rep_ref = zeros(replica_num, 1)

for i in eachindex(rand_rep_ref)
    rand_rep_ref[i] = (i-1)*N_sd
end

#------------------------------------------------------------------------------------------------------------------------------#

#CHANGING ALL THE MATRICES TO CU_ARRAY 
global z_dir_sd = Array(z_dir_sd)

#no need to change these matrices to Array -- changing the position from 1 to 0.5
global x_pos_sd = Array(x_pos_sd)                              
global y_pos_sd = Array(y_pos_sd)
global z_pos_sd = Array(z_pos_sd)

NN_s = Array{Int64}(NN_s)
NN_n = Array{Int64}(NN_n)
NN_e = Array{Int64}(NN_e)
NN_w = Array{Int64}(NN_w)

J_NN = Array(J_NN)

spin_rep_ref = Array{Int64}(spin_rep_ref)
rand_rep_ref = Array{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#DEFINING THE CHARGES UPPER AND LOWER CHARGE LAYER OF SPINS
#q_upper = z_dir_sd
#q_lower = (-1)*z_dir_sd

#DEFINING THE POSITION OF UPPER AND LOWER CHERGE LAYERS
#x_pos_upper = x_pos_sd
#y_pos_upper = y_pos_sd
#z_pos_upper = z_pos_sd .+ 0.5

#DEFINING THE POSITION OF UPPER AND LOWER CHERGE LAYERS
#x_pos_lower = x_pos_sd
#y_pos_lower = y_pos_sd
#z_pos_lower = z_pos_sd .- 0.5

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO EXCHANGE 
global energy_exchange = zeros(N_sd*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE EXCHANGE ENERGY OF THE SYSTEM
function compute_exchange_energy()
    r_s = (mx_sd.-1).*N_sd .+ NN_s
    r_n = (mx_sd.-1).*N_sd .+ NN_n 
    r_e = (mx_sd.-1).*N_sd .+ NN_e 
    r_w = (mx_sd.-1).*N_sd .+ NN_w 

    energy_x = z_dir_sd.*((J_NN[r_s].*z_dir_sd[NN_s .+ spin_rep_ref]) 
                        .+(J_NN[r_n].*z_dir_sd[NN_n .+ spin_rep_ref]) 
                        .+(J_NN[r_e].*z_dir_sd[NN_e .+ spin_rep_ref]) 
                        .+(J_NN[r_w].*z_dir_sd[NN_w .+ spin_rep_ref]))
   
    global energy_exchange = energy_x

    return energy_exchange
end

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATE EWALD SUM 
#-----------------------------------------------------------#
#REAL SPACE CALCULATIONS
#global alpha = 1/(3*sqrt(2)*n_x)
global n_cut_real = 50
global simulation_box_num = (2*n_cut_real + 1)^2

x_pos_ES = n_x*collect(-n_cut_real:1:n_cut_real)
x_pos_ES = repeat(x_pos_ES, inner=(2*n_cut_real + 1))
x_pos_ES = reshape(x_pos_ES, 1, 1, simulation_box_num)
    
y_pos_ES = n_y*collect(-n_cut_real:1:n_cut_real)
y_pos_ES = repeat(y_pos_ES, outer=(2*n_cut_real +1))
y_pos_ES = reshape(y_pos_ES, 1, 1, simulation_box_num)

global x_pos_sd_ES = x_pos_sd' .- x_pos_ES
global y_pos_sd_ES = y_pos_sd' .- y_pos_ES

distance_ij = sqrt.( ((x_pos_sd .- x_pos_sd_ES).^2) .+ ((y_pos_sd .- y_pos_sd_ES).^2))

term_1 = (1/2) .* (z_dir_sd .* z_dir_sd') ./ (distance_ij .^ 3)
replace!(term_1, Inf=>0)
term_1 = sum(term_1, dims=3)
term_1 = vec(sum(term_1, dims=2))

