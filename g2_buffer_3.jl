using CUDA, Random, Plots, FFTW

rng = MersenneTwister(1234)
#total number of Monte Carlo steps
MC_steps = 100000

#define the size of a 1 system
N = 1000

#define a 1D vector with beads either pointing Up or down.
global mx = [(-1)^rand(rng, Int64) for i in 1:N]

#list of delay times
delay_times = 10*collect(1:500)

#-------------------------------------------------------------------#
#monte carlo step
function One_MC()
    global r = rand(1:N)
    global mx[r] = (-1)*mx[r]
end

#-------------------------------------------------------------------#
#auto-correlation functio 
function correlation()
    global mx_mux += mx_t .* mx_delt
end

#-------------------------------------------------------------------#
#matrix to save data
global auto_correlation = zeros(length(delay_times), 1)

#-------------------------------------------------------------------#
#main body
for delay in eachindex(delay_times)
    global c = 0
    global step_count = 0

    global mx_mux = zeros(N,1)
    global mx_t = mx[1:N]
    for i in 1:MC_steps
        
        One_MC()

        c += 1

        if (c == delay_times[delay])
            global mx_delt = mx[1:N]
            correlation()

            global c = 0
            global step_count += 1
            global mx_t = mx[1:N]

        end
    end
    println(step_count)
    global auto_correlation[delay] = sum(mx_mux)/(step_count*N)
end

plot(delay_times, auto_correlation, framestyle=:box, )
