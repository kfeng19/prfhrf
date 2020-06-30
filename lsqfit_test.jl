include(joinpath(@__DIR__,"forward_model.jl"))
using MAT
using LsqFit

## model
stim_sequence = load_stimulus()
model(x,p) = forward_model(stim_sequence[:,:,1:length(x)], p, isComplex=false, in_type="unbounded")

## Load "do-work" hrf parameters
work_file = matopen(joinpath(@__DIR__,"../do_work_new.mat"))
M_work = read(work_file, "doWork")
close(work_file)

## Prepare ground truth data
hrf_param_ind = rand(1:size(M_work,1))
hrf_params = M_work[hrf_param_ind,:]
prf_params = [5*randn(),5*randn(),rand(0.1:0.1:6)]
p_gt = vcat(prf_params,hrf_params)

## Look at hrf
t = 0:0.1:20
plot(t,hrf_fun(p_gt[4:end]..., t))
xlabel!("time (s)")
ylabel!("level (a.u.)")
## data
xdata = 1:size(stim_sequence,3)
ydata = forward_model(stim_sequence, reverse_input(p_gt), isComplex=false, in_type="unbounded")
p0 = [0, 0, 1, 6, 8, 0.8, 0.8, 0.5]
# take a Look
plot(ydata)
## fit
@time fit_bounds = curve_fit(model, xdata, ydata, reverse_input(p0))

## Check output
out_t = fit_bounds.param
out = transform_input(out_t)
