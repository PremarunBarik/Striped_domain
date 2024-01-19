using Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions

rng = MersenneTwister()
global B_global = 0.0   #globally applied field on the system
#global alpha = 0.35     #defined parameter for dipolar interaction energy
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
z_dir_sd = [-(1)^rand(rng, Int64) for i in 1:N_sd]
#z_dir_sd = [1, 1, -1, 1]
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

#creating a matrix with zero diagonal terms  to calculate dipolar interaction term
diag_zero = fill(1, (N_sd, N_sd)) |> Array 
global diag_zero[diagind(diag_zero)] .= 0

#creating a matrix with one diagonal terms  to calculate dipolar interaction term
global diag_one = fill(0, (N_sd, N_sd)) |> Array 
global diag_one[diagind(diag_one)] .= 1

#CALCULATE EWALD SUM 
#-----------------------------------------------------------#

function Ewald_sum()
    
    #REAL SPACE CALCULATIONS
    global alpha = 0.05
    global n_cut_real_es = 1
    global simulation_box_num_es = (2*n_cut_real_es + 1)^2

    x_pos_real_es = n_x*collect(-n_cut_real_es:1:n_cut_real_es)
    x_pos_real_es = repeat(x_pos_real_es, inner=(2*n_cut_real_es + 1))
    global x_pos_real_es = reshape(x_pos_real_es, 1 , 1, simulation_box_num_es)
    
    y_pos_real_es = n_y*collect(-n_cut_real_es:1:n_cut_real_es)
    y_pos_real_es = repeat(y_pos_real_es, outer=(2*n_cut_real_es +1))
    global y_pos_real_es = reshape(y_pos_real_es, 1, 1, simulation_box_num_es)

    global z_pos_real_es = zeros(1, 1, simulation_box_num_es)

    global x_pos_sd_real_es = x_pos_sd' .+ x_pos_real_es
    global y_pos_sd_real_es = y_pos_sd' .+ y_pos_real_es
    global z_pos_upper_real_es = z_pos_upper' .+ z_pos_real_es
    global z_pos_lower_real_es = z_pos_lower' .+ z_pos_real_es

    global distance_upper_i_upper_j_es = sqrt.( ((x_pos_sd .- x_pos_sd_real_es).^2) .+ ((y_pos_sd .- y_pos_sd_real_es).^2) .+ ((z_pos_upper .- z_pos_upper_real_es).^2))
    global distance_upper_i_lower_j_es = sqrt.( ((x_pos_sd .- x_pos_sd_real_es).^2) .+ ((y_pos_sd .- y_pos_sd_real_es).^2) .+ ((z_pos_upper .- z_pos_lower_real_es).^2))
    global distance_lower_i_upper_j_es = sqrt.( ((x_pos_sd .- x_pos_sd_real_es).^2) .+ ((y_pos_sd .- y_pos_sd_real_es).^2) .+ ((z_pos_lower .- z_pos_upper_real_es).^2))
    global distance_lower_i_lower_j_es = sqrt.( ((x_pos_sd .- x_pos_sd_real_es).^2) .+ ((y_pos_sd .- y_pos_sd_real_es).^2) .+ ((z_pos_lower .- z_pos_lower_real_es).^2))

    distance_upper_i_upper_j_es[:,:, (simulation_box_num_es+1)/2 |> Int64] += diag_one
    distance_lower_i_lower_j_es[:,:, (simulation_box_num_es+1)/2 |> Int64] += diag_one
    #-----------------------------------------------------------#
    #RECIPROCAL SPACE CALCULATIONS
    global n_cut_reciprocal_es = 1

    x_pos_reciprocal_es = (2*pi/n_x)*collect(1:n_cut_reciprocal_es)
    x_pos_reciprocal_es = repeat(x_pos_reciprocal_es, inner=n_cut_reciprocal_es)
    global x_pos_reciprocal_es = reshape(x_pos_reciprocal_es, 1, 1, n_cut_reciprocal_es^2)

    y_pos_reciprocal_es = (2*pi/n_y)*collect(1:n_cut_reciprocal_es)
    y_pos_reciprocal_es = repeat(y_pos_reciprocal_es, outer=n_cut_reciprocal_es)
    global y_pos_reciprocal_es = reshape(y_pos_reciprocal_es, 1, 1, n_cut_reciprocal_es^2)

    global z_pos_reciprocal_es = (2*pi/ dipole_length) .* ones(1, 1, n_cut_reciprocal_es^2)

    global k_mod_2_es = (x_pos_reciprocal_es .^ 2) .+ (y_pos_reciprocal_es .^ 2) .+ (z_pos_reciprocal_es .^ 2)
    global exponential_term_es = exp.( - k_mod_2_es ./ (4*alpha^2)) 
    global cosine_term_upper_upper_es = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal_es) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal_es) .+ ((z_pos_upper .- z_pos_upper') .* z_pos_reciprocal_es))
    global cosine_term_upper_lower_es = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal_es) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal_es) .+ ((z_pos_upper .- z_pos_lower') .* z_pos_reciprocal_es))
    global cosine_term_lower_upper_es = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal_es) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal_es) .+ ((z_pos_lower .- z_pos_upper') .* z_pos_reciprocal_es))
    global cosine_term_lower_lower_es = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal_es) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal_es) .+ ((z_pos_lower .- z_pos_lower') .* z_pos_reciprocal_es))

    #-----------------------------------------------------------#

    q_upper_es = z_dir_sd
    q_lower_es = (-1) .* z_dir_sd

    q_upper_upper_es = (q_upper_es .* q_upper_es')
    q_upper_lower_es = (q_upper_es .* q_lower_es')
    q_lower_upper_es = (q_lower_es .* q_upper_es')
    q_lower_lower_es = (q_lower_es .* q_lower_es')

    term_1_1_es = (q_upper_upper_es .* erfc.(alpha .* distance_upper_i_upper_j_es) ./ distance_upper_i_upper_j_es)
    term_1_1_es[:,:, (simulation_box_num_es+1)/2 |> Int64] = term_1_1_es[:,:, (simulation_box_num_es+1)/2 |> Int64] .* diag_zero

    term_1_2_es = (q_upper_lower_es .* erfc.(alpha .* distance_upper_i_lower_j_es) ./ distance_upper_i_lower_j_es)
    term_1_2_es[:,:, (simulation_box_num_es+1)/2 |> Int64] = term_1_2_es[:,:, (simulation_box_num_es+1)/2 |> Int64] .* diag_zero

    term_1_3_es = (q_lower_upper_es .* erfc.(alpha .* distance_lower_i_upper_j_es) ./ distance_lower_i_upper_j_es)
    term_1_3_es[:,:, (simulation_box_num_es+1)/2 |> Int64] = term_1_3_es[:,:, (simulation_box_num_es+1)/2 |> Int64] .* diag_zero

    term_1_4_es = (q_lower_lower_es .* erfc.(alpha .* distance_lower_i_lower_j_es) ./ distance_lower_i_lower_j_es)
    term_1_4_es[:,:, (simulation_box_num_es+1)/2 |> Int64] = term_1_4_es[:,:, (simulation_box_num_es+1)/2 |> Int64] .* diag_zero

    term_1_es = (1/2) .* (term_1_1_es .+ term_1_2_es .+ term_1_3_es .+ term_1_4_es)
    term_1_es = sum(term_1_es, dims=3) 
    term_1_es= vec(sum(term_1_es, dims=2))

    term_2_es = (2*pi/(n_x*n_y*dipole_length)) .* ((q_upper_upper_es .* cosine_term_upper_upper_es) 
                .+ (q_upper_lower_es .* cosine_term_upper_lower_es) 
                .+ (q_lower_upper_es .* cosine_term_lower_upper_es)
                .+ (q_lower_lower_es .* cosine_term_lower_lower_es)) .* exponential_term_es ./ k_mod_2_es
    
    term_2_es = sum(term_2_es, dims=3) 
    term_2_es= vec(sum(term_2_es, dims=2))
                
#    term_3_es = (alpha/sqrt(pi)).* ((q_upper_es .^2) .+ (q_lower_es .^2))

    global energy_ewald_sum_CC = term_1_es .+ term_2_es 

end

function Asymptotic_soln()
    #REAL SPACE CALCULATIONS
    #global alpha = 1.5
    global n_cut_real = 20
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

    distance_upper_i_upper_j[:,:, (simulation_box_num+1)/2 |> Int64] += diag_one
    distance_lower_i_lower_j[:,:, (simulation_box_num+1)/2 |> Int64] += diag_one


    q_upper = z_dir_sd
    q_lower = (-1) .* z_dir_sd

    q_upper_upper = (q_upper .* q_upper')
    q_upper_lower = (q_upper .* q_lower')
    q_lower_upper = (q_lower .* q_upper')
    q_lower_lower = (q_lower .* q_lower')

    term_1_1 = (q_upper_upper ./ distance_upper_i_upper_j)
    term_1_1[:,:, (simulation_box_num+1)/2 |> Int64] = term_1_1[:,:, (simulation_box_num+1)/2 |> Int64] .* diag_zero

    term_1_2 = (q_upper_lower ./ distance_upper_i_lower_j)
    term_1_2[:,:, (simulation_box_num+1)/2 |> Int64] = term_1_2[:,:, (simulation_box_num+1)/2 |> Int64] .* diag_zero

    term_1_3 = (q_lower_upper ./ distance_lower_i_upper_j)
    term_1_3[:,:, (simulation_box_num+1)/2 |> Int64] = term_1_3[:,:, (simulation_box_num+1)/2 |> Int64] .* diag_zero

    term_1_4 = (q_lower_lower ./ distance_lower_i_lower_j)
    term_1_4[:,:, (simulation_box_num+1)/2 |> Int64] = term_1_4[:,:, (simulation_box_num+1)/2 |> Int64] .* diag_zero

    term_1 = (1/2) .* (term_1_1 .+ term_1_2 .+ term_1_3 .+ term_1_4)
    term_1 = sum(term_1, dims=3) 
    term_1= vec(sum(term_1, dims=2))

end
