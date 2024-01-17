using Plots, FFTW

#define sampling rate
fs = 1

#define ranges of x and y axis.
x_min = 1
x_max = 100

y_min = 1
y_max = 100

x = collect(x_min:fs:x_max)
y = collect(y_min:fs:y_max)

#define periodicity of sin function
f(x,y) = sin(2*pi*y/5)

z = f.(x',y)

heatmap(z)

#contour(x,y,z)

z_fft = fftshift(fft(z))

heatmap(real(z_fft), color=:viridis)
