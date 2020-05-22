include(joinpath(@__DIR__,"optimizers.jl"))
include(joinpath(@__DIR__,"forward_model.jl"))

## Generate data
x_gt = [1,3,2.5, 6, 10, 0.8, 0.9, 0.4]
x_t = reverse_input(x_gt)
x_back = transform_input(x_t)
# println(x_t)
stim_sequence = load_stimulus()
input = forward_model(stim_sequence, x_t, isComplex=false)

## Regression
# f(x) = loss(forward_2D(stim_sequence,x),input)
f(x) = loss(forward_model(stim_sequence,x),input,x,λ=0)
f_real(x) = real(f(x))
g(x) = diff_complex(f, x)

## Begin
max_n = 10;
u0 = [0, 1, 1, 5, 9, 0.5, 0.6, 0.3]
u0_t = reverse_input(u0)
x_hist = barrier_cg(f_real,g,constraints,u0_t,max_n,γ=10,debug=true)
println(transform_input(last(x_hist)))
plot(log.(f_real.(x_hist)))   # Convergence

## Resulted signal
outfit = forward_model(stim_sequence, last(x_hist), isComplex=false)
plot(outfit)
plot!(input)

## hrf
t=0:20
plot(hrf_fun(transform_input(last(x_hist))[4:end]...,t))
plot!(hrf_fun(6, 10, 0.8, 0.9, 0.4, t))

## 2D Visualization
y = 0:0.5:15
x = y
plot(x,y,(x,y)->f_real([x,y]),st=:contour)
# plot!(Tuple.(x_hist),markershape=:circle,line=:arrow,linewidth=1)
