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

global kizzie = abs.(x_pos_sd .- x_pos_sd')/n_x
global eta = abs.(y_pos_sd .- y_pos_sd')/n_y

global l_list = collect(1:10)
global m_list = collect(-10:1:10)

global denom = zeros(N_sd, N_sd)
global denom_counter = zeros(N_sd, N_sd)

for i in 1:N_sd
    for j in 1:N_sd

        if (eta[i,j]==0)

            if (kizzie[i,j]==0)

                denom[i,j] = 4*zeta(3/2)*(zeta(3/2, 1/4) - zeta(3/2, 3/4)) / (4^(3/2))/(n_x^3)
                denom_counter[i,j] = 1
            
            else
                
                sum = 0.0

                for l in 1:length(l_list)
                    for m in 1:length(m_list)
                        sum += cos(2*pi*l_list[l]*eta[i,j])*abs(l_list[l]/(kizzie[i,j]+ m_list[m]))*besselk(1, (2*pi*l_list[l]*abs(kizzie[i,j]+m_list[m])))
                    end
                end

            denom[i,j] = (8*pi/(n_x^3))*(sum + 2*(zeta(2, kizzie[i,j]) - zeta(2, -kizzie[i,j]) - kizzie[i,j]^(-2)))
            denom_counter[i,j] = 2

            end
        
        else

            sum = 0.0

            for l in 1:length(l_list)
                for m in 1:length(m_list)
                    sum += cos(2*pi*l_list[l]*kizzie[i,j])*abs(l_list[l]/(eta[i,j]+ m_list[m]))*besselk(1, (2*pi*l_list[l]*abs(eta[i,j]+m_list[m])))
                end
            end

            denom[i,j] = (8*pi/(n_x^3))*(sum + 2*(zeta(2, eta[i,j]) - zeta(2, -eta[i,j]) - eta[i,j]^(-2)))
            denom_counter[i,j] = 3
        end
    end
end

