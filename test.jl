include(joinpath(@__DIR__,"optimizers.jl"))
include(joinpath(@__DIR__,"forward_model.jl"))
using MAT
# plotly()
# gr()

## Load "do-work" hrf parameters
work_file = matopen(joinpath(@__DIR__,"../do_work_new.mat"))
M_work = read(work_file, "doWork")
close(work_file)

## Prepare input
hrf_param_ind = rand(1:size(M_work,1))
hrf_params = M_work[hrf_param_ind,:]
prf_params = [5*randn(),5*randn(),rand(0.1:0.1:6)]
x_gt = vcat(prf_params,hrf_params)

# while true
#     global x_gt = [5*randn(),5*randn(),rand(0.1:0.1:6), rand(1:0.1:9), rand(1:0.1:9), rand(), rand(), rand(0.1:0.1:0.9)]
#     if x_gt[5]*x_gt[7] > x_gt[4] * x_gt[6] && x_gt[4] * x_gt[6] > 3
#         break
#     end
# end

## generate_data
# input = generate_data(x_gt)
x_t = reverse_input(x_gt)
stim_sequence = load_stimulus()
time_series = forward_model(stim_sequence, x_t, isComplex=false)
# Look at hrf
t = 0:0.1:20
plot(t,hrf_fun(x_gt[4:end]..., t))
xlabel!("time (s)")
ylabel!("level (a.u.)")
## Regression
# f(x) = loss(forward_2D(stim_sequence,x),time_series,x,λ=0)
f(x) = loss(forward_model(stim_sequence,x),time_series,x,λ=1e3,regularizer=soft_regularizer)
f_real(x) = real(f(x))
g(x) = diff_complex(f, x)

## Begin
# backtracking scale too large may lead to spikes at beginning of each IP iteration
config_dict = Dict("ϵ"=>1e-3,"cg_max_iter"=>50, "g_th"=>10, "cg_scale0"=>5)
max_n = 10;
u0 = [0, 0, 1, 6, 8, 0.6, 0.8, 0.3]
# while true
#     global u0 = [5*randn(),5*randn(),rand(0.1:0.1:6), rand(1:0.1:9), rand(1:0.1:9), rand(), rand(), rand(0.1:0.1:0.9)]
#     if u0[5]*u0[7] > u0[4] * u0[6] && u0[4] * u0[6] > 3
#         break
#     end
# end
u0_t = reverse_input(u0)
# u0_t = [-0.61037161069572, -0.42834464211034795, -0.2903652423384784, 0.6332112087524869, 6.667964960178128, 0.9674976194821926, 0.09885535890646496, -0.859434633201733];
x_hist = Ip_gradient(f_real, g, constraints, u0_t, max_n, config_dict, ρ=1, γ=10,method=bfgs)
# x_hist = Ip_gradient(f_real, g, constraints, u0_t, max_n, config_dict, ρ=1, γ=10,method=cg)
println(last(x_hist))
println(transform_input(last(x_hist)))
plot(log.(f_real.(x_hist)),xlabel="Iterations",ylabel="log loss")   # Convergence

## Resultant signal
outfit = forward_model(stim_sequence, last(x_hist), isComplex=false)
plot(outfit)
plot!(time_series)
xlabel!("t (s)")
ylabel!("signal (a.u.)")

## hrf
t=0:20
plot(hrf_fun(transform_input(last(x_hist))[4:end]...,t))
plot!(hrf_fun(x_gt[4:end]..., t))
xlabel!("t (s)")
ylabel!("signal (a.u.)")

## 2D Visualization
y = 1:0.1:4
x = y
plot(x,y,(x,y)->f_real([x,y]),st=:contour,levels=exp.(3:0.5:20))
plot!(Tuple.(x_hist),markershape=:circle,line=:arrow,linewidth=1)

## visualize final point
# x_temp = [-6.5401287100106975, 3.81610296419304, 4.599999624208395, 2.4999925745194616, 6.099965347853528, 0.4332453988970216, 0.7323600871279149, 0.2000007510018203];
# x_temp_t = reverse_input(x_temp)
x_temp_t = last(x_hist)
plot(ylim=(0,1e5))
# plot()
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
