using Plots, Random, DelimitedFiles

global MCSteps = 100000
global MCWindow = 100


global detector_length = 256
global detector_space = detector_length*detector_length
global detector_pixels = zeros(detector_space, 1)

global timestamps = Float64[]

function One_MC(step)
    
    global intensity = rand(1:25)
    global rand_pos = rand(1:detector_space, intensity)

    for i in eachindex(rand_pos)
        if detector_pixels[rand_pos[i]]==0
            detector_pixels[rand_pos[i]] = step
        end
    end

end

function download_timestamps()
    global timestamps_window = detector_pixels[ detector_pixels .!=0]
    global timestamps = vcat(timestamps, timestamps_window)

    global detector_pixels = zeros(detector_space, 1)
end

for step in 1:MCSteps
    One_MC(step)

    if (step % MCWindow == 0)
        download_timestamps()
    end
end

writedlm("ToyModel5_timestamps.txt", timestamps)
