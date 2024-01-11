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

#LENGTH OF DIPOLE 
dipole_length = 0.5

#SPIN ELEMENT DIRECTION IN REPLICAS
x_dir_sd = [(1)^rand(rng, Int64) for i in 1:N_sd]
x_dir_sd = repeat(x_dir_sd, replica_num, 1)

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

x_pos_positive = x_pos_sd .+ (dipole_length/2)
x_pos_negative = x_pos_sd .- (dipole_length/2)

#------------------------------------------------------------------------------------------------------------------------------#

function Ewald_sum_CC()
    #REAL SPACE CALCULATIONS
    global alpha = 0.1
    global n_cut_real_es = 1
    global simulation_box_num_es = (2*n_cut_real_es + 1)^2

    x_pos_real_es = n_x*collect(-n_cut_real_es:1:n_cut_real_es)
    x_pos_real_es = repeat(x_pos_real_es, inner=(2*n_cut_real_es + 1))
    x_pos_real_es = reshape(x_pos_real_es, 1 , 1, simulation_box_num_es)
    
    y_pos_real_es = n_y*collect(-n_cut_real_es:1:n_cut_real_es)
    y_pos_real_es = repeat(y_pos_real_es, outer=(2*n_cut_real_es +1))
    y_pos_real_es = reshape(y_pos_real_es, 1, 1, simulation_box_num_es)

    global x_pos_positive_real_es = x_pos_positive' .+ x_pos_real_es
    global x_pos_negative_real_es = x_pos_negative' .+ x_pos_real_es
    global y_pos_sd_real_es = y_pos_sd' .+ y_pos_real_es

    distance_positive_i_positive_j_es = sqrt.( ((x_pos_positive .- x_pos_positive_real_es).^2) .+ ((y_pos_sd .- y_pos_sd_real_es).^2))
    distance_positive_i_negative_j_es = sqrt.( ((x_pos_positive .- x_pos_negative_real_es).^2) .+ ((y_pos_sd .- y_pos_sd_real_es).^2))
    distance_negative_i_positive_j_es = sqrt.( ((x_pos_negative .- x_pos_positive_real_es).^2) .+ ((y_pos_sd .- y_pos_sd_real_es).^2))
    distance_negative_i_negative_j_es = sqrt.( ((x_pos_negative .- x_pos_negative_real_es).^2) .+ ((y_pos_sd .- y_pos_sd_real_es).^2))

    #-----------------------------------------------------------#
    #RECIPROCAL SPACE CALCULATIONS
    global n_cut_reciprocal_es = 1

    x_pos_reciprocal_es = (2*pi/n_x)*collect(1:n_cut_reciprocal_es)
    x_pos_reciprocal_es = repeat(x_pos_reciprocal_es, inner=n_cut_reciprocal_es)
    x_pos_reciprocal_es = reshape(x_pos_reciprocal_es, 1, 1, n_cut_reciprocal_es^2)

    y_pos_reciprocal_es = (2*pi/n_y)*collect(1:n_cut_reciprocal_es)
    y_pos_reciprocal_es = repeat(y_pos_reciprocal_es, outer=n_cut_reciprocal_es)
    y_pos_reciprocal_es = reshape(y_pos_reciprocal_es, 1, 1, n_cut_reciprocal_es^2)

    global k_mod_2_es = (x_pos_reciprocal_es .^ 2) .+ (y_pos_reciprocal_es .^ 2)
    global exponential_term_es = exp.( - k_mod_2_es ./ (4*alpha^2)) 
    global cosine_term_positive_positive_es = cos.(((x_pos_positive .- x_pos_positive') .* x_pos_reciprocal_es) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal_es))
    global cosine_term_positive_negative_es = cos.(((x_pos_positive .- x_pos_negative') .* x_pos_reciprocal_es) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal_es))
    global cosine_term_negative_positive_es = cos.(((x_pos_negative .- x_pos_positive') .* x_pos_reciprocal_es) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal_es))
    global cosine_term_negative_negative_es = cos.(((x_pos_negative .- x_pos_negative') .* x_pos_reciprocal_es) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal_es))

    #-----------------------------------------------------------#

    q_positive_es = x_dir_sd
    q_negative_es = (-1) .* x_dir_sd

    q_positive_positive_es = (q_positive_es .* q_positive_es')
    q_positive_negative_es = (q_positive_es .* q_negative_es')
    q_negative_positive_es = (q_negative_es .* q_positive_es')
    q_negative_negative_es = (q_negative_es .* q_negative_es')

    term_1_es = 1/2*(replace!((q_positive_positive_es .* erfc.(alpha .* distance_positive_i_positive_j_es) ./ distance_positive_i_positive_j_es), Inf=>0)
                .+ replace!((q_positive_negative_es .* erfc.(alpha .* distance_positive_i_negative_j_es) ./ distance_positive_i_negative_j_es), Inf=>0)
                .+ replace!((q_negative_positive_es .* erfc.(alpha .* distance_negative_i_positive_j_es) ./ distance_negative_i_positive_j_es), Inf=>0)
                .+ replace!((q_negative_negative_es .* erfc.(alpha .* distance_negative_i_negative_j_es) ./ distance_negative_i_negative_j_es), Inf=>0))

    replace!(term_1_es, Inf=>0)
    term_1_es = sum(term_1_es, dims=3) 
    term_1_es = vec(sum(term_1_es, dims=2))
    
    term_2_es = (2*pi/(n_x*n_y)) .* ((q_positive_positive_es .* cosine_term_positive_positive_es) 
                .+ (q_positive_negative_es .* cosine_term_positive_negative_es) 
                .+ (q_negative_positive_es .* cosine_term_negative_positive_es)
                .+ (q_negative_negative_es .* cosine_term_negative_negative_es)) .* exponential_term_es ./ k_mod_2_es
    
    term_2_es = sum(term_2_es, dims=3) 
    term_2_es = vec(sum(term_2_es, dims=2))
                

    term_3_es = (alpha/sqrt(pi)).* ((q_positive_es.^2) .+ (q_negative_es .^ 2))

    energy_ewald_sum_CC = term_1_es .+ term_2_es .- term_3_es
end

function Asymptotic_soln()
    
    #REAL SPACE CALCULATIONS
    #global alpha = 1.5
    global n_cut_real = 50
    global simulation_box_num = (2*n_cut_real + 1)^2

    x_pos_real = n_x*collect(-n_cut_real:1:n_cut_real)
    x_pos_real = repeat(x_pos_real, inner=(2*n_cut_real + 1))
    x_pos_real = reshape(x_pos_real, 1 , 1, simulation_box_num)
    
    y_pos_real = n_y*collect(-n_cut_real:1:n_cut_real)
    y_pos_real = repeat(y_pos_real, outer=(2*n_cut_real +1))
    y_pos_real = reshape(y_pos_real, 1, 1, simulation_box_num)

    global x_pos_positive_real = x_pos_positive' .+ x_pos_real
    global x_pos_negative_real = x_pos_negative' .+ x_pos_real
    global y_pos_sd_real = y_pos_sd' .+ y_pos_real

    distance_positive_i_positive_j = sqrt.( ((x_pos_positive .- x_pos_positive_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2))
    distance_positive_i_negative_j = sqrt.( ((x_pos_positive .- x_pos_negative_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2))
    distance_negative_i_positive_j = sqrt.( ((x_pos_negative .- x_pos_positive_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2))
    distance_negative_i_negative_j = sqrt.( ((x_pos_negative .- x_pos_negative_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2))

    q_positive = x_dir_sd
    q_negative = (-1) .* x_dir_sd

    q_positive_positive = (q_positive .* q_positive')
    q_positive_negative = (q_positive .* q_negative')
    q_negative_positive = (q_negative .* q_positive')
    q_negative_negative = (q_negative .* q_negative')

    term_1 = 1/2*(replace!((q_positive_positive ./ distance_positive_i_positive_j), Inf=>0)
                .+ replace!((q_positive_negative ./ distance_positive_i_negative_j), Inf=>0)
                .+ replace!((q_negative_positive ./ distance_negative_i_positive_j), Inf=>0)
                .+ replace!((q_negative_negative ./ distance_negative_i_negative_j), Inf=>0))

    replace!(term_1, Inf=>0)
    term_1 = sum(term_1, dims=3) 
    term_1= vec(sum(term_1, dims=2))
                
end
