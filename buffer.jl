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
n_x = 20
n_y = 20
n_z = 1

N_sd = n_x*n_y

#NUMBER OF REPLICAS 
replica_num = 1

#SPIN ELEMENT DIRECTION IN REPLICAS
z_dir_sd = [(-1)^rand(rng, Int64) for i in 1:N_sd]
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
global x_pos_sd = Array(x_pos_sd .- 0.5)                              
global y_pos_sd = Array(y_pos_sd .- 0.5)
global z_pos_sd = Array(z_pos_sd .- 0.5)

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

#creating a matrix with zero diagonal terms  to calculate dipolar interaction term
diag_zero = fill(1, (N_sd, N_sd)) |> Array 
global diag_zero[diagind(diag_zero)] .= 0
global  diag_zero_replica = repeat(diag_zero, replica_num, 1)

#creating a matrix with one diagonal terms  to calculate dipolar interaction term
global diag_one = fill(0, (N_sd, N_sd)) |> Array 
global diag_one[diagind(diag_one)] .= 1
#global  diag_zero = repeat(diag_zero, replica_num, 1)


#COMPUTE DISTANCE BETWEEN SPIN PAIRS
function calculate_distance_of_spinpairs()
    x_j = x_pos_sd' |>  Array
    x_j = repeat(x_j, N_sd, 1)                                  #Scalar indexing - less time consuming

    y_j = y_pos_sd' |>  Array
    y_j = repeat(y_j, N_sd, 1)                                  #Scalar indexing - less time consuming

    z_j = z_pos_sd' |>  Array
    z_j = repeat(z_j, N_sd, 1)                                  #Scalar indexing - less time consuming

    global distance_ij = sqrt.((x_pos_sd .- x_j).^2 .+ (y_pos_sd .- y_j).^2 .+ (z_pos_sd .- z_j).^2)
    global distance_ij = distance_ij .* diag_zero 
    global distance_ij = distance_ij .+ diag_one

    global distance_ij = repeat(distance_ij, replica_num, 1)

end 

#------------------------------------------------------------------------------------------------------------------------------#

calculate_distance_of_spinpairs()
#MATRIX TO STORE ENERGY DUE TO EXCHANGE 
global energy_dipolar = zeros(N_sd*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATE DIPOLAR INTERACTION ENERGY
function calculate_dipolar_energy()

    spin_mux = reshape(z_dir_sd, (N_sd, replica_num))' |>  Array
    spin_mux = repeat(spin_mux, inner = (N_sd, 1))                                  #Scalar indexing - less time consum$

    global dipolar_energy = z_dir_sd .* spin_mux 
    global dipolar_energy = alpha .* dipolar_energy ./ (distance_ij.^3)
    global dipolar_energy = dipolar_energy .* diag_zero_replica
    global energy_dipolar = sum(dipolar_energy, dims=2)

end
#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATE EWALD SUM 
#-----------------------------------------------------------#
global alpha = 1
global n_cut_real = 1
global simulation_box_num = (2*n_cut_real + 1)^2

x_pos_ES = n_x*collect(-n_cut_real:1:n_cut_real)
x_pos_ES = repeat(x_pos_ES, inner=(2*n_cut_real + 1))
x_pos_ES = repeat(x_pos_ES, outer=N_sd)
    
y_pos_ES = n_y*collect(-n_cut_real:1:n_cut_real)
y_pos_ES = repeat(y_pos_ES, outer=(2*n_cut_real +1))
y_pos_ES = repeat(y_pos_ES, outer=N_sd)

x_pos_sd_ES = repeat(x_pos_sd, outer = simulation_box_num)
y_pos_sd_ES = repeat(y_pos_sd, outer = simulation_box_num)
z_pos_sd = repeat(z_pos_sd, outer = simulation_box_num)

global x_pos_sd_ES = x_pos_sd_ES - x_pos_ES
global y_pos_sd_ES = y_pos_sd_ES - y_pos_ES

distance_ij = sqrt.( ((x_pos_sd .- x_pos_sd_ES').^2) .+ ((y_pos_sd .- y_pos_sd_ES').^2))
#-----------------------------------------------------------#
global n_cut_reciprocal = 

function Ewald_sum()

    B_term = (erfc.(alpha .* distance_ij) .+ ((2*alpha/pi) .* (distance_ij) .* (exp.(-(alpha^2) .* (distance_ij.^2))))) ./ (distance_ij.^3)
    C_term = ((3 .* erfc.(alpha .* distance_ij)) .+ ((2*alpha/pi) .* (3 .+ (2 .* (alpha^2) .* (distance_ij.^2))) .* exp.((-alpha^2) .* (distance_ij .^ 2)))) ./ (distance_ij .^ 5) 

    z_dir_sd_ES = repeat(z_dir_sd, outer = simulation_box_num)
    energy_real = (z_dir_sd.*z_dir_sd_ES') ./ (distance_ij) .* erfc(alpha .* distance_ij)
    energy_real = sum(energy_real, dims=2)
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sd*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_tot_energy_spin_glass()
    compute_exchange_energy()
    calculate_dipolar_energy()

    global energy_tot = (energy_exchange .- energy_dipolar .+ (B_global*z_dir_sd))

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
    global r = rand_pos .+ rand_rep_ref

    global del_energy = 2*energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#Matrix to keep track of which flipped how many times
#global flip_count = Array(zeros(N_sg*replica_num, 1))

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

#MAIN BODY
#MC BURN STEPS
for j in 1:MC_burns

    one_MC(rng, Temp)

end
#-----------------------------------------------------------#
for j in 1:MC_steps

    one_MC(rng, Temp)                                                     #MONTE CARLO FUNCTION 

end

#------------------------------------------------------------------------------------------------------------------------------#

#PLOTTING SPINS WITH CROSS AND DOT
global marker_pallete = [:xcross, :square, :circle]
global z_dir_sd_plot = z_dir_sd[1:N_sd] .+ 2

scatter(x_pos_sd, y_pos_sd, markershape=marker_pallete[z_dir_sd_plot], 
        markersize=4, legend=false, framestyle=:box, size=(400,400))
savefig("striped_domins_alpha$(alpha)_temp$(Temp).png")
#------------------------------------------------------------------------------------------------------------------------------#
