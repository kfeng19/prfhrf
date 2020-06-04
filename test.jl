include(joinpath(@__DIR__,"optimizers.jl"))
include(joinpath(@__DIR__,"forward_model.jl"))
# plotly()
gr()
## Generate data
# x_gt = [1,3,2.5, 7, 9, 0.8, 0.9, 0.4]
while true
    global x_gt = [5*randn(),5*randn(),rand(0:0.1:9), rand(1:0.1:9), rand(1:0.1:9), rand(), rand(), rand(0.1:0.1:0.9)]
    if x_gt[5]*x_gt[7] > x_gt[4] * x_gt[6]
        break
    end
end
# input = generate_data(x_gt)
x_t = reverse_input(x_gt)
stim_sequence = load_stimulus()
time_series = forward_model(stim_sequence, x_t, isComplex=false)

## Regression
# f(x) = loss(forward_2D(stim_sequence,x),time_series,x,λ=0)
f(x) = loss(forward_model(stim_sequence,x),time_series,x,λ=0)
f_real(x) = real(f(x))
g(x) = diff_complex(f, x)

## Begin
config_dict = Dict("ϵ"=>5e-3,"cg_max_iter"=>50, "g_th"=>10)
max_n = 10;
u0 = [0, 1, 1, 3, 5, 0.5, 0.6, 0.3]
u0_t = reverse_input(u0)
# x_hist = Ip_gradient(f_real, g, constraints, u0_t, max_n, config_dict, γ=10,method=cg)
x_hist = Ip_gradient(f_real, g, constraints, u0_t, max_n, config_dict, ρ=1, γ=10,method=bfgs)
println(last(x_hist))
println(transform_input(last(x_hist)))
plot(log.(f_real.(x_hist)),xlabel="Iterations",ylabel="log loss")   # Convergence

## Resulted signal
outfit = forward_model(stim_sequence, last(x_hist), isComplex=false)
plot(outfit)
plot!(time_series)

## hrf
t=0:20
plot(hrf_fun(transform_input(last(x_hist))[4:end]...,t))
plot!(hrf_fun(x_gt[4:end]..., t))

## 2D Visualization
y = 1:0.1:4
x = y
plot(x,y,(x,y)->f_real([x,y]),st=:contour,levels=exp.(3:0.5:20))
plot!(Tuple.(x_hist),markershape=:circle,line=:arrow,linewidth=1)

## visualize final point
x_temp = [-6.5401287100106975, 3.81610296419304, 4.599999624208395, 2.4999925745194616, 6.099965347853528, 0.4332453988970216, 0.7323600871279149, 0.2000007510018203];
x_temp_t = reverse_input(x_temp)
# f_real(x_temp_t)
plot(ylim=(0,5e6))
vf = []
for k=1:8
    x_temp_t2 = copy(x_temp_t)
    w0 = x_temp_t[k]
    vw = range(w0/2,1.5w0,length=100)
    vf = []
    for w in vw
        x_temp_t2[k] = w
        push!(vf, f_real(x_temp_t2))
    end
    # println(vf)
    plot!(vf)
end
xlabel!("x[i]")
ylabel!("Loss")
current()
