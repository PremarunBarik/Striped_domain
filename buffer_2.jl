#asymptotic - dipole-dipole - in-plane
using Random, Plots, LinearAlgebra, BenchmarkTools, SpecialFunctions

#This code does not implement ewald sum method. Here we calculate the dontribution of dipolar energy of a lot
#of image spins (millions) on the spins inside the main system or simulation box. the final matrix--term_1--contains
#the energy terms spin wise. Meaning the matrix contains energy terms spinwise.

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

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATE EWALD SUM 
#-----------------------------------------------------------#
#REAL SPACE CALCULATIONS
#global alpha = 1/(3*sqrt(2)*n_x)
global n_cut_real = 50
global simulation_box_num = (2*n_cut_real + 1)^2

x_pos_real = n_x*collect(-n_cut_real:1:n_cut_real)
x_pos_real = repeat(x_pos_real, inner=(2*n_cut_real + 1))
x_pos_real = reshape(x_pos_real, 1 , 1, simulation_box_num)
    
y_pos_real = n_y*collect(-n_cut_real:1:n_cut_real)
y_pos_real = repeat(y_pos_real, outer=(2*n_cut_real +1))
y_pos_real = reshape(y_pos_real, 1, 1, simulation_box_num)

global x_pos_sd_real = x_pos_sd' .- x_pos_real
global y_pos_sd_real = y_pos_sd' .- y_pos_real

distance_ij = sqrt.(((x_pos_sd .- x_pos_sd_real).^2) .+ ((y_pos_sd .- y_pos_sd_real).^2))

term_1 = (1/2) .* (x_dir_sd .* x_dir_sd') ./ (distance_ij .^ 3)
replace!(term_1, Inf=>0)

term_2 = (3/2) .* (x_dir_sd .*(x_pos_sd .- x_pos_sd_real)) .* (x_dir_sd' .*(x_pos_sd .- x_pos_sd_real)) ./ (distance_ij .^ 5)
replace!(term_2, NaN=>0)

asymptotic_energy = term_1 .- term_2
asymptotic_energy = sum(asymptotic_energy, dims=3)
asymptotic_energy = vec(sum(asymptotic_energy, dims=2))
