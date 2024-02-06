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
z_dir_sd = [(1)^rand(rng, Int64) for i in 1:N_sd]
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

#------------------------------------------------------------------------------------------------------------------------------#

#creating a matrix with zero diagonal terms  to calculate dipolar interaction term
diag_zero = fill(1, (N_sd, N_sd)) |> Array 
global diag_zero[diagind(diag_zero)] .= 0

#creating a matrix with one diagonal terms  to calculate dipolar interaction term
global diag_one = fill(0, (N_sd, N_sd)) |> Array 
global diag_one[diagind(diag_one)] .= 1

#------------------------------------------------------------------------------------------------------------------------------#


global eta = 0.14233
global n_cut_es = 3
global simulation_box_num_es = (2*n_cut_es + 1)^2

x_pos_es = n_x*collect(-n_cut_es:1:n_cut_es)
x_pos_es = repeat(x_pos_es, inner=(2*n_cut_es + 1))
global x_pos_es = reshape(x_pos_es, 1 , 1, simulation_box_num_es)
    
y_pos_es = n_y*collect(-n_cut_es:1:n_cut_es)
y_pos_es = repeat(y_pos_es, outer=(2*n_cut_es +1))
global y_pos_es = reshape(y_pos_es, 1, 1, simulation_box_num_es)

global x_pos_sd_es = x_pos_sd' .+ x_pos_es
global y_pos_sd_es = y_pos_sd' .+ y_pos_es

global R_ij = sqrt.((x_pos_sd .- x_pos_sd_es).^2 .+ (y_pos_sd .- y_pos_sd_es).^2)
global R_ij[:,:, (simulation_box_num_es+1)/2 |> Int64] += diag_one

global A_term = ((erfc.(eta .* R_ij) ./ R_ij) .+ ((2*eta) .* exp.( - (eta^2) .* (R_ij .^ 2))) ./ sqrt(pi)) ./ (R_ij .^ 2)

function Ewald_sum()
    
    delta_H1 = (-2)*(z_dir_sd .* z_dir_sd') .* A_term
    delta_H1[:,:, (simulation_box_num_es+1)/2 |> Int64] =  delta_H1[:,:, (simulation_box_num_es+1)/2 |> Int64] .* diag_zero
    delta_H1 = sum(delta_H1, dims=3)
    delta_H1 = vec(sum(delta_H1, dims=2))

    delta_H2 = (-2/N_sd) .* (z_dir_sd) .* sum(z_dir_sd)

    delta_H_dipolar = delta_H1 .+ delta_H2 
end

#------------------------------------------------------------------------------------------------------------------------------#

n_cut_as = 10
global simulation_box_num_as = (2*n_cut_as + 1)^2

x_pos_as = n_x*collect(-n_cut_as:1:n_cut_as)
x_pos_as = repeat(x_pos_as, inner=(2*n_cut_as + 1))
global x_pos_as = reshape(x_pos_as, 1 , 1, simulation_box_num_as)
    
y_pos_as = n_y*collect(-n_cut_as:1:n_cut_as)
y_pos_as = repeat(y_pos_as, outer=(2*n_cut_as +1))
global y_pos_as = reshape(y_pos_as, 1, 1, simulation_box_num_as)

global x_pos_sd_as = x_pos_sd' .+ x_pos_as
global y_pos_sd_as = y_pos_sd' .+ y_pos_as

global R_ij_as = sqrt.((x_pos_sd .- x_pos_sd_as).^2 .+ (y_pos_sd .- y_pos_sd_as).^2)
global R_ij_as[:,:, (simulation_box_num_as+1)/2 |> Int64] += diag_one

function Sum_as()
    
    delta_E = (-2)*(z_dir_sd .* z_dir_sd') ./ (R_ij_as .^ 3)
    delta_E[:,:, (simulation_box_num_as+1)/2 |> Int64] =  delta_E[:,:, (simulation_box_num_as+1)/2 |> Int64] .* diag_zero
    delta_E = sum(delta_E, dims=3)
    delta_E = vec(sum(delta_E, dims=2))

end
