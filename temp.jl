using Plots
include(joinpath(@__DIR__,"forward_model.jl"))
t = 0:20
x = [6.0, 5.0, 0.6000000000000001, 0.7, 0.4]
h = hrf_fun(x...,t)
plot(t,h)
