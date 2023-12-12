using Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions

rng = MersenneTwister()
global B_global = 0.0   #globally applied field on the system
global alpha_ratio = 0.35     #defined parameter for dipolar interaction energy
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

#LENGTH OF DIPOLE 
dipole_length = 1

#SPIN ELEMENT DIRECTION IN REPLICAS
z_dir_sd = [(-1)^rand(rng, Int64) for i in 1:N_sd]
z_dir_sd = repeat(z_dir_sd, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sd = Array(collect(1:N_sd*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY -- needed to define neighbors and to 
#plot the spin configuration. So, we don't need to create a Array of these matrices 
#also no need to repeat for replicas because spin positions are constant over replicas 

x_pos_sd = zeros(N_sd, 1)
y_pos_sd = zeros(N_sd, 1)
z_pos_sd = fill(0.5, N_sd, 1)

for i in 1:N_sd
    x_pos_sd[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sd[i] = ((i-1)%n_y)+1                         #1th position
end

z_pos_upper = z_pos_sd .+ (dipole_length/2)
z_pos_lower = z_pos_sd .- (dipole_length/2)

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

NN_s = Array{Int64}(NN_s)
NN_n = Array{Int64}(NN_n)
NN_e = Array{Int64}(NN_e)
NN_w = Array{Int64}(NN_w)

J_NN = Array(J_NN)

spin_rep_ref = Array{Int64}(spin_rep_ref)
rand_rep_ref = Array{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATE EWALD SUM 
#-----------------------------------------------------------#
#REAL SPACE CALCULATIONS
global alpha = 0.2
global n_cut_real = 10
global simulation_box_num = (2*n_cut_real + 1)^2

x_pos_real = n_x*collect(-n_cut_real:1:n_cut_real)
x_pos_real = repeat(x_pos_real, inner=(2*n_cut_real + 1))
global x_pos_real = reshape(x_pos_real, 1 , 1, simulation_box_num)
    
y_pos_real = n_y*collect(-n_cut_real:1:n_cut_real)
y_pos_real = repeat(y_pos_real, outer=(2*n_cut_real +1))
global y_pos_real = reshape(y_pos_real, 1, 1, simulation_box_num)

global z_pos_real = zeros(1, 1, simulation_box_num)

global x_pos_sd_real = x_pos_sd' .+ x_pos_real
global y_pos_sd_real = y_pos_sd' .+ y_pos_real
global z_pos_upper_real = z_pos_upper' .+ z_pos_real
global z_pos_lower_real = z_pos_lower' .+ z_pos_real

global distance_upper_i_upper_j = sqrt.( ((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2) .+ ((z_pos_upper .- z_pos_upper_real).^2))
global distance_upper_i_lower_j = sqrt.( ((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2) .+ ((z_pos_upper .- z_pos_lower_real).^2))
global distance_lower_i_upper_j = sqrt.( ((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2) .+ ((z_pos_lower .- z_pos_upper_real).^2))
global distance_lower_i_lower_j = sqrt.( ((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2) .+ ((z_pos_lower .- z_pos_lower_real).^2))

#-----------------------------------------------------------#
#RECIPROCAL SPACE CALCULATIONS
global n_cut_reciprocal = 10

x_pos_reciprocal = (2*pi/n_x)*collect(1:n_cut_reciprocal)
x_pos_reciprocal = repeat(x_pos_reciprocal, inner=n_cut_reciprocal)
global x_pos_reciprocal = reshape(x_pos_reciprocal, 1, 1, n_cut_reciprocal^2)

y_pos_reciprocal = (2*pi/n_y)*collect(1:n_cut_reciprocal)
y_pos_reciprocal = repeat(y_pos_reciprocal, outer=n_cut_reciprocal)
global y_pos_reciprocal = reshape(y_pos_reciprocal, 1, 1, n_cut_reciprocal^2)

global z_pos_reciprocal = (2*pi/ dipole_length) .* ones(1, 1, n_cut_reciprocal^2)

global k_mod_2 = (x_pos_reciprocal .^ 2) .+ (y_pos_reciprocal .^ 2) .+ (z_pos_reciprocal .^ 2)
global exponential_term = exp.( - k_mod_2 ./ (4*alpha^2)) 
global cosine_term_upper_upper = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal) .+ ((z_pos_upper .- z_pos_upper') .* z_pos_reciprocal))
global cosine_term_upper_lower = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal) .+ ((z_pos_upper .- z_pos_lower') .* z_pos_reciprocal))
global cosine_term_lower_upper = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal) .+ ((z_pos_lower .- z_pos_upper') .* z_pos_reciprocal))
global cosine_term_lower_lower = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal) .+ ((z_pos_lower .- z_pos_lower') .* z_pos_reciprocal))

#-----------------------------------------------------------#
function calculate_Ewald_sum()
    q_upper = z_dir_sd
    q_lower = (-1) .* z_dir_sd

    q_upper_upper = (q_upper .* q_upper')
    q_upper_lower = (q_upper .* q_lower')
    q_lower_upper = (q_lower .* q_upper')
    q_lower_lower = (q_lower .* q_lower')

    term_1 = 1/2*(replace!((q_upper_upper .* erfc.(alpha .* distance_upper_i_upper_j) ./ distance_upper_i_upper_j), Inf=>0)
                .+ replace!((q_upper_lower .* erfc.(alpha .* distance_upper_i_lower_j) ./ distance_upper_i_lower_j), Inf=>0)
                .+ replace!((q_lower_upper .* erfc.(alpha .* distance_lower_i_upper_j) ./ distance_lower_i_upper_j), Inf=>0)
                .+ replace!((q_lower_lower .* erfc.(alpha .* distance_lower_i_lower_j) ./ distance_lower_i_lower_j), Inf=>0))

    term_1 = sum(term_1, dims=3) 
    term_1= vec(sum(term_1, dims=2))

    term_2 = (2*pi/(n_x*n_y*dipole_length)) .* ((q_upper_upper .* cosine_term_upper_upper) 
                .+ (q_upper_lower .* cosine_term_upper_lower) 
                .+ (q_lower_upper .* cosine_term_lower_upper)
                .+ (q_lower_lower .* cosine_term_lower_lower)) .* exponential_term ./ k_mod_2
    
    term_2 = sum(term_2, dims=3) 
    term_2= vec(sum(term_2, dims=2))
                
    term_3 = (alpha/sqrt(2*pi)).* ((q_upper .^2) .+ (q_lower .^2))

    global energy_ewald_sum_CC = term_1 .+ term_2 .- term_3

end


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
#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sd*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_tot_energy_spin_glass()
    compute_exchange_energy()
    calculate_Ewald_sum()

    global energy_tot = (energy_exchange .- energy_ewald_sum_CC .+ (B_global*z_dir_sd))

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
