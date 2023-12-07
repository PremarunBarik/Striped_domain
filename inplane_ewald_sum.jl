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

x_pos_positive = x_pos_sd .+ ((dipole_length/2) * x_dir_sd)
x_pos_negative = x_pos_sd .- ((dipole_length/2) * x_dir_sd)

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATE EWALD SUM 
#-----------------------------------------------------------#
#REAL SPACE CALCULATIONS
global alpha = 0.5
global n_cut_real = 5
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

#-----------------------------------------------------------#
#RECIPROCAL SPACE CALCULATIONS
global n_cut_reciprocal = 12

x_pos_reciprocal = (2*pi/n_x)*collect(1:n_cut_reciprocal)
x_pos_reciprocal = repeat(x_pos_reciprocal, inner=n_cut_reciprocal)
x_pos_reciprocal = reshape(x_pos_reciprocal, 1, 1, n_cut_reciprocal^2)

y_pos_reciprocal = (2*pi/n_y)*collect(1:n_cut_reciprocal)
y_pos_reciprocal = repeat(y_pos_reciprocal, outer=n_cut_reciprocal)
y_pos_reciprocal = reshape(y_pos_reciprocal, 1, 1, n_cut_reciprocal^2)

global k_mod_2 = (x_pos_reciprocal .^ 2) .+ (y_pos_reciprocal .^ 2)
global exponential_term = exp.( - k_mod_2 ./ (4*alpha^2)) 
global cosine_term_positive_positive = cos.(((x_pos_positive .- x_pos_positive') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal))
global cosine_term_positive_negative = cos.(((x_pos_positive .- x_pos_negative') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal))
global cosine_term_negative_positive = cos.(((x_pos_negative .- x_pos_positive') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal))
global cosine_term_negative_negative = cos.(((x_pos_negative .- x_pos_negative') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal))

#-----------------------------------------------------------#

function Ewald_sum_CC()
    q_positive = x_dir_sd
    q_negative = (-1) .* x_dir_sd

    q_positive_positive = (q_positive .* q_positive')
    q_positive_negative = (q_positive .* q_negative')
    q_negative_positive = (q_negative .* q_positive')
    q_negative_negative = (q_negative .* q_negative')

    term_1 = 1/2*(replace!((q_positive_positive .* erfc.(alpha .* distance_positive_i_positive_j) ./ distance_positive_i_positive_j), Inf=>0)
                .+ replace!((q_positive_negative .* erfc.(alpha .* distance_positive_i_negative_j) ./ distance_positive_i_negative_j), Inf=>0)
                .+ replace!((q_negative_positive .* erfc.(alpha .* distance_negative_i_positive_j) ./ distance_negative_i_positive_j), Inf=>0)
                .+ replace!((q_negative_negative .* erfc.(alpha .* distance_negative_i_negative_j) ./ distance_negative_i_negative_j), Inf=>0))

    replace!(term_1, Inf=>0)
    term_1 = sum(term_1, dims=3) 
    term_1= vec(sum(term_1, dims=2))
    
    term_2 = (2*pi/(n_x*n_y)) .* ((q_positive_positive .* cosine_term_positive_positive) 
                .+ (q_positive_negative .* cosine_term_positive_negative) 
                .+ (q_negative_positive .* cosine_term_negative_positive)
                .+ (q_negative_negative .* cosine_term_negative_negative)) .* exponential_term ./ k_mod_2
    
    term_2 = sum(term_2, dims=3) 
    term_2= vec(sum(term_2, dims=2))
                

    term_3 = (alpha/sqrt(pi)).* ((q_positive.^2) .+ (q_negative .^ 2))

    energy_ewald_sum_CC = term_1 .+ term_2 .- term_3
end
