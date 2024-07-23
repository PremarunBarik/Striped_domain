using Plots, DelimitedFiles, FFTW

global FinalSpins = readdlm("SD_final_spins_T3.1L32J6.0B0.0psNum20repNum20_point.txt")

global Temp = 3.1
global pinning_site_num = 20
global replica_num = 20
global n_x = 32
global n_y = 32
global N_sd = n_x * n_y

if pinning_site_num != 0
    global x_pos_ps = readdlm("SD_XposPinningSite_T$(Temp)L32J6.0B0.0psNum$(pinning_site_num)repNum20_point.txt")
    global y_pos_ps = readdlm("SD_YposPinningSite_T$(Temp)L32J6.0B0.0psNum$(pinning_site_num)repNum20_point.txt")
end

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sd = Array(collect(1:N_sd*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY -- needed to define neighbors and to 
#plot the spin configuration. So, we don't need to create a Array of these matrices 
#also no need to repeat for replicas because spin positions are constant over replicas 

x_pos_sd = zeros(N_sd, replica_num)
y_pos_sd = zeros(N_sd, replica_num)

for k in 1:replica_num
    for i in 1:N_sd
        x_pos_sd[i,k] = trunc((i-1)/n_x)+1                    #10th position
        y_pos_sd[i,k] = ((i-1)%n_y)+1                         #1th position
    end
end

#------------------------------------------------------------------------------------------------------------------------------#
#PLOTTING SPINS HEATMAP 
function plot_config_heatmap()

    global z_dir_sd_plot = FinalSpins

    anim = @animate for replica in 1:replica_num
        heatmap(reshape(z_dir_sd_plot[:,replica], n_x, n_y), color=:grays, cbar=false, 
                xticks=false, yticks=false, framestyle=:box, size=(500,400),
                dpi=300)
        title!("Temp:$Temp, replica:$replica, pinning sites:$pinning_site_num")
        if pinning_site_num != 0
            scatter!(x_pos_ps[:,replica], y_pos_ps[:,replica], legend=false, 
            ms=3, msw = 0)
        end
    end
    gif(anim, "SD_Config_L$(n_x)Temp$(Temp)psNum$(pinning_site_num)repNum$(replica_num).gif", fps = 2)
end

#------------------------------------------------------------------------------------------------------------------------------#

#define the circle to calculate intensity from
global fft_x_center = (trunc(n_x/2) + 1) |> Int64
global fft_y_center = (trunc(n_y/2) + 1) |> Int64

global fft_distance = sqrt.(((x_pos_sd[1:N_sd] .- fft_x_center) .^ 2 ) .+ ((y_pos_sd[1:N_sd] .- fft_y_center) .^ 2))

global stripe_width = 4
global circle_radius = n_x/(2*stripe_width)   #stripe width depend on the exchange to dipolar interaction coefficient ratio 
                                            #If J= 6.0, stripe width is 4, if J=8.9 stripe width is 8
global radius_error = 0.9                   #depending on the choice of this we can define how thick the concentric
                                            #circle would be on the fourier plot
global fft_cell_list = Int64[]              #list of matrix cells where we collect the intensities from the Fourier plots

for i in 1:N_sd
    if fft_distance[i]<= (circle_radius + radius_error) && fft_distance[i] >= (circle_radius - radius_error)
        append!(fft_cell_list, i)
    end
end

global num_of_cells = length(fft_cell_list)

global z_dir_sd_plot = FinalSpins
global spin_fft = zeros(n_x, n_y) 

for replica in 1:replica_num
    global spin_fft += abs.(fftshift(fft(reshape(z_dir_sd_plot[:,replica], n_x, n_y))))/N_sd
end
global spin_fft = spin_fft/replica_num

#------------------------------------------------------------------------------------------------------------------------------#

#plotting fourier transform
function plot_fft()

    # Define circle parameters
    circle_center = (fft_x_center, fft_y_center)
    radius1 = n_x/(2*stripe_width) + radius_error
    radius2 = n_x/(2*stripe_width) - radius_error
    theta = LinRange(0, 2π, 100)
    circle1_x = circle_center[1] .+ radius1 .* cos.(theta)
    circle1_y = circle_center[2] .+ radius1 .* sin.(theta)
    circle2_x = circle_center[1] .+ radius2 .* cos.(theta)
    circle2_y = circle_center[2] .+ radius2 .* sin.(theta)

    heatmap(spin_fft, color=:viridis, framestyle=:box, size=(500,400),
            dpi=300, alpha=1.0)
    plot!(circle1_x, circle1_y, lw=2, lc=:red, legend=false)
    plot!(circle2_x, circle2_y, lw=2, lc=:red, legend=false)
    title!("32x32 system, Temp:$Temp, pinning sites:$pinning_site_num")
    savefig("SD_FFT_L$(n_x)Temp$(Temp)psNum$(pinning_site_num)repNum$(replica_num).png")
end

#------------------------------------------------------------------------------------------------------------------------------#

function fft_intensity_alternative()
    global fft_intensity = spin_fft[fft_cell_list] 
    global fft_intensity = sum(fft_intensity)

end

#------------------------------------------------------------------------------------------------------------------------------#

#plot_config_heatmap()
#plot_fft()

fft_intensity_alternative()
