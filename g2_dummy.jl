using CUDA, Random, Plots, FFTW

rng = MersenneTwister(1234)
#total number of Monte Carlo steps
MC_steps = 100000

#define the size of a 1 system
N = 1000

#define a 1D vector with beads either pointing Up or down.
global mx = [rand(1:100) for i in 1:N]

#list of delay times
delay_times = collect(1:10:5001)

#-------------------------------------------------------------------#
#monte carlo step
function One_MC()
    global r = rand(1:N)
    global mx[r] += (rand() < 0.5 ? 0 : 1)*((-1)^rand(rng, Int64))*rand(1:10)
end

#-------------------------------------------------------------------#
#auto-correlation functio 
function correlation()
    global cross_intensity_mx += mx_t .* mx_delt
    global self_intensity_mx += mx_t
end

#-------------------------------------------------------------------#
#matrix to save data
global auto_correlation = zeros(length(delay_times), 1)

#-------------------------------------------------------------------#
#main body
for delay in eachindex(delay_times)

    global mx = [rand(1:100) for i in 1:N]

    global c = 0
    global interval_count = 0

    global cross_intensity_mx = zeros(N,1)
    global self_intensity_mx = zeros(N,1)

    global mx_t = mx[1:N]
    for i in 1:MC_steps
        
        One_MC()

        c += 1

        if (c == delay_times[delay])
            global mx_delt = mx[1:N]
            correlation()

            global c = 0
            global interval_count += 1
            global mx_t = mx[1:N]

        end
    end
    println(interval_count)
    global g2_value = (cross_intensity_mx/interval_count) ./ ((self_intensity_mx/interval_count) .^ 2)
    global auto_correlation[delay] = sum(g2_value)/N
end

plot(delay_times, auto_correlation, framestyle=:box, xscale=:log10, xlabel="Time delay", ylabel="Auto-correlation", legend=false)
title!("Model#2 (delta_I: +10 : -10)")
