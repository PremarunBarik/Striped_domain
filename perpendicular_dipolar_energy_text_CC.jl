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
n_x = 2
n_y = 2
n_z = 1

N_sd = n_x*n_y

#NUMBER OF REPLICAS 
replica_num = 1

#LENGTH OF DIPOLE 
dipole_length = 1

#SPIN ELEMENT DIRECTION IN REPLICAS
#z_dir_sd = [(1)^rand(rng, Int64) for i in 1:N_sd]
z_dir_sd = [1, 1, -1, 1]
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
#REAL SPACE CALCULATIONS
#global alpha = 1.5
global n_cut_real = 100
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

#energy_radius[radius] = sum(term_1)/N_sd
                


#scatter!(radius_cutoff, energy_radius, label="Not Ewald sum", framestyle=:box)
#xlabel!("Cutoff number of replica layer", guidefont=font(14), xtickfont=font(12))
#ylabel!("Ewald_sum_energy", guidefont=font(14), xtickfont=font(12))
#savefig("Ewald_sum_comparison_inplane_CC.png")
