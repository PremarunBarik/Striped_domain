using CUDA, Random, Plots, FFTW

rng = MersenneTwister(1234)
#total number of Monte Carlo steps
MC_steps = 200000

#define the size of a 1 system
global N = 100
#define the number of replicas
#global replica_num = 10

#define a 1D vector with beads either pointing Up or down.
global mx = rand(1:25)

#list of delay times
#delay_times_1 = collect(1:9)
#delay_times_2 = Array{Int64}(reshape(collect(1:0.5:9.5) .* (10 .^ collect(1:2))', 18*2, 1))
#delay_times_3 = Array{Int64}(reshape(collect(1:0.1:9.9) .* (10 .^ collect(3:4))', 90*2, 1))
#global delay_times = vcat(delay_times_1, delay_times_2, delay_times_3)
global delay_times = collect(1:1000)

#define the grating function
global gating_period = 200
x = collect(1:(gating_period))

gating_function = 1 .- exp.((x .- gating_period) ./ 100)
gating_function = repeat(gating_function, (MC_steps/(gating_period) |> Int64)+1 ,1)
global gating_function = gating_function[1:MC_steps]'

#define the switch function
ontime = 200
offtime = 100
switch_function = vcat(ones(ontime,1), zeros(offtime),1)
global switch_function = repeat(switch_function, (trunc(MC_steps/(ontime+offtime)) |> Int64)+1 ,1)
global switch_function = switch_function[1:MC_steps]

#-------------------------------------------------------------------#
#monte carlo step
function One_MC()
    #global r = rand(1:N)

    #rand_num = rand()

    #if rand_num < 0.5
    #    global mx[r] = rand(1:50)
    #end
    
    global mx = rand(1:25)
end

#-------------------------------------------------------------------#
#MATRIX FOR STORING DATA
global mx_MCSteps = zeros(MC_steps, 1) |> Array

#-------------------------------------------------------------------#
#main body
for steps in 1:MC_steps
    One_MC()
    global mx_MCSteps[steps, 1] = mx
end

#incorporate the grating function to Monte Carlo steps
#global mx_MCSteps = mx_MCSteps .* gating_function
#global mx_MCSteps = mx_MCSteps .* switch_function

data = readdlm("ToyModel_DecorrelatedIntensity.txt")
global mx_MCSteps = data
global mx_MCSteps = mx_MCSteps .* switch_function
global bin_intensity = mx_MCSteps

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STORING DATA
global auto_correlation = zeros(length(delay_times), 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#
global denominator = (sum(bin_intensity)/length(bin_intensity)) .^2

for delay in eachindex(delay_times)
    global delay_time = delay_times[delay]

    global zero_mx = vec(zeros(delay_time, 1))

    global bin_intensity_i = vcat(zero_mx, bin_intensity)
    global bin_intensity_j = vcat(bin_intensity, zero_mx)

    global cross_intensity = bin_intensity_i .* bin_intensity_j
    
    global g2 = (sum(cross_intensity)/(length(bin_intensity)-delay_time))/denominator
    global auto_correlation[delay] = g2

    println(delay_time)
end
plot(((delay_times[1:length(delay_times)]) .+1), (auto_correlation[1:length(auto_correlation)]), framestyle=:box, xscale=:log10, 
    linewidth=2, xlabel="Time delay", ylabel="g2", label="Julia script(normalization=True)",
    xminorticks=9, yminorticks=10, minorgrid=true, dpi=300)
title!("ToyModel-Decorrelated data(no switch function)")
