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
dipole_length = 1

#SPIN ELEMENT DIRECTION IN REPLICAS
z_dir_sd = [(1)^rand(rng, Int64) for i in 1:N_sd]
z_dir_sd = repeat(z_dir_sd, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sd = Array(collect(1:N_sd*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY -- needed to define neighbors and to 
#plot the spin configuration. So, we don't need to create a Array of these matrices 
#also no need to repeat for replicas because spin positions are constant over replicas 

x_pos_sd = zeros(N_sd, 1)
y_pos_sd = zeros(N_sd, 1)
z_pos_sd = fill(1.5, N_sd, 1)

for i in 1:N_sd
    x_pos_sd[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sd[i] = ((i-1)%n_y)+1                         #1th position
end

z_pos_upper = z_pos_sd .+ (dipole_length/2)
z_pos_lower = z_pos_sd .- (dipole_length/2)

#------------------------------------------------------------------------------------------------------------------------------#

radius_cutoff = collect(1:30)
energy_radius = zeros(30,1)


#CALCULATE EWALD SUM 
#-----------------------------------------------------------#
#REAL SPACE CALCULATIONS
global alpha = 0.05
global n_cut_real = 30
global simulation_box_num = (2*n_cut_real + 1)^2

x_pos_real = n_x*collect(-n_cut_real:1:n_cut_real)
x_pos_real = repeat(x_pos_real, inner=(2*n_cut_real + 1))
x_pos_real = reshape(x_pos_real, 1 , 1, simulation_box_num)
    
y_pos_real = n_y*collect(-n_cut_real:1:n_cut_real)
y_pos_real = repeat(y_pos_real, outer=(2*n_cut_real +1))
y_pos_real = reshape(y_pos_real, 1, 1, simulation_box_num)

z_pos_real = zeros(1, 1, simulation_box_num)

global x_pos_sd_real = x_pos_sd' .+ x_pos_real
global y_pos_sd_real = y_pos_sd' .+ y_pos_real
global z_pos_upper_real = z_pos_upper' .+ z_pos_real
global z_pos_lower_real = z_pos_lower' .+ z_pos_real

distance_upper_i_upper_j = sqrt.( ((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2) .+ ((z_pos_upper .- z_pos_upper_real).^2))
distance_upper_i_lower_j = sqrt.( ((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2) .+ ((z_pos_upper .- z_pos_lower_real).^2))
distance_lower_i_upper_j = sqrt.( ((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2) .+ ((z_pos_lower .- z_pos_upper_real).^2))
distance_lower_i_lower_j = sqrt.( ((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2) .+ ((z_pos_lower .- z_pos_lower_real).^2))

#-----------------------------------------------------------#
#RECIPROCAL SPACE CALCULATIONS
global n_cut_reciprocal = 10

x_pos_reciprocal = (2*pi/n_x)*collect(1:n_cut_reciprocal)
x_pos_reciprocal = repeat(x_pos_reciprocal, inner=n_cut_reciprocal)
x_pos_reciprocal = reshape(x_pos_reciprocal, 1, 1, n_cut_reciprocal^2)

y_pos_reciprocal = (2*pi/n_y)*collect(1:n_cut_reciprocal)
y_pos_reciprocal = repeat(y_pos_reciprocal, outer=n_cut_reciprocal)
y_pos_reciprocal = reshape(y_pos_reciprocal, 1, 1, n_cut_reciprocal^2)

z_pos_reciprocal = (2*pi/ dipole_length) .* ones(1, 1, n_cut_reciprocal^2)

global k_mod_2 = (x_pos_reciprocal .^ 2) .+ (y_pos_reciprocal .^ 2) .+ (z_pos_reciprocal .^ 2)
global exponential_term = exp.( - k_mod_2 ./ (4*alpha^2)) 
global cosine_term_upper_upper = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal) .+ ((z_pos_upper .- z_pos_upper') .* z_pos_reciprocal))
global cosine_term_upper_lower = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal) .+ ((z_pos_upper .- z_pos_lower') .* z_pos_reciprocal))
global cosine_term_lower_upper = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal) .+ ((z_pos_lower .- z_pos_upper') .* z_pos_reciprocal))
global cosine_term_lower_lower = cos.(((x_pos_sd .- x_pos_sd') .* x_pos_reciprocal) .+ ((y_pos_sd .- y_pos_sd') .* y_pos_reciprocal) .+ ((z_pos_lower .- z_pos_lower') .* z_pos_reciprocal))

#-----------------------------------------------------------#

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
                

    term_3 = (alpha/sqrt(pi)).* ((q_upper .^2) .+ (q_lower .^2))

    energy_ewald_sum_CC = term_1 .+ term_2 .- term_3

#    energy_radius[radius] = sum(energy_ewald_sum_CC)/N_sd


#scatter!(radius_cutoff, energy_radius, label="Ewald_sum, alpha:$alpha", framestyle=:box)
#xlabel!("Cutoff number of replica layer", guidefont=font(14), xtickfont=font(12))
#ylabel!("Ewald_sum_energy", guidefont=font(14), xtickfont=font(12))
#savefig("2D_EA_SpatialCorrelation_g.png")
