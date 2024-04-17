using CUDA, Random, Plots, FFTW

rng = MersenneTwister(1234)
#total number of Monte Carlo steps
MC_steps = 100000

#define the size of a 1 system
global N = 100
#define the number of replicas
#global replica_num = 10

#define a 1D vector with beads either pointing Up or down.
global mx = [rand(1:50) for i in 1:N]

#list of delay times
delay_times_1 = collect(1:9)
delay_times_2 = Array{Int64}(reshape(collect(1:0.5:9.5) .* (10 .^ collect(1:4))', 18*4, 1))
#delay_times_3 = Array{Int64}(collect(1:0.5:5) .* 10^5)
global delay_times = vcat(delay_times_1, delay_times_2)

#define the grating function
global grating_period = 10000
x = collect(1:(grating_period))

gating_function = 1 .- exp.((x .- grating_period) ./ 1000)
gating_function = repeat(gating_function, (MC_steps/(gating_period) |> Int64)+1 ,1)
global gating_function = gating_function[1:MC_steps]'

#-------------------------------------------------------------------#
#monte carlo step
function One_MC()
    global r = rand(1:N)

    rand_num = rand()

    if rand_num < 0.5
        global mx[r] = rand(1:50)
    end
end

#-------------------------------------------------------------------#
#auto-correlation functio 
function correlation()
    global cross_intensity_mx += mx_t .* mx_delt
    global self_intensity_mx += mx_t
end

#-------------------------------------------------------------------#
#MATRIX FOR STORING DATA
global mx_MCSteps = zeros(N, MC_steps) |> Array

#-------------------------------------------------------------------#
#main body
for steps in 1:MC_steps
    One_MC()
    global mx_MCSteps[:,steps] = mx
end

#incorporate the grating function to Monte Carlo steps
global mx_MCSteps = mx_MCSteps .* gating_function

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STORING DATA
global auto_correlation = zeros(length(delay_times), 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#
#calculating Auto-correlation
for delay in eachindex(delay_times)

    global interval_count = 0

    global cross_intensity_mx = zeros(N,1)
    global self_intensity_mx = zeros(N,1)

    for i in 1:MC_steps
        global mx_t = mx_MCSteps[:,i]

        if ((i+delay_times[delay]) <= MC_steps)
            global mx_delt = mx_MCSteps[:, i+delay_times[delay]]
            correlation()

            global interval_count += 1
        end
    end
    println(interval_count)
    global g2_value = (cross_intensity_mx/interval_count) ./ ((self_intensity_mx/interval_count) .^ 2)
    global auto_correlation[delay] = sum(g2_value)/N
end

plot(delay_times, auto_correlation, framestyle=:box, xscale=:log10, linewidth=2, xlabel="Time delay", ylabel="Auto-correlation", legend=false)
title!("Model#3")
