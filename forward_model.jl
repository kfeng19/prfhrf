using MAT
using LinearAlgebra
using Plots
using DSP

## Functions
# Load stimulus sequence
function load_stimulus(filename="stimulus_ordered.mat")
    stim_file = matopen("stimulus_ordered.mat")
    stim_sequence = read(stim_file, "stimToWrite")
    close(stim_file)
    return stim_sequence
end

function get_gaussian(σ, x0, y0)
    y = range(-10,10,length=101)
    x = y'
    return exp.(-((x.-x0).^2 .+ (y.-y0).^2)/(2*σ^2));
end

function transform_input(θ_t)
    x0_t, y0_t, σ_t, a1_t, a2_t, b1_t, b2_t, c_t = θ_t
    x0 = 20 * atan(x0_t) / pi
    y0 = 20 * atan(y0_t) / pi
    σ = 10 * atan(σ_t) / pi + 5.
    a1 = a1_t ^ 2
    a2 = a2_t ^ 2
    b1 = atan(b1_t) / pi + 0.5
    b2 = atan(b2_t) / pi + 0.5
    c = atan(c_t) / pi + 0.5
    return [x0, y0, σ, a1, a2, b1, b2, c]
end

function reverse_input(θ)
    x0, y0, σ, a1, a2, b1, b2, c = θ
    x0_t = tan(pi*x0/20)
    y0_t = tan(pi*y0/20)
    σ_t = tan(pi*(σ-5)/10)
    a1_t = sqrt(a1)
    a2_t = sqrt(a2)
    b1_t = tan(pi*(b1-0.5))
    b2_t = tan(pi*(b2-0.5))
    c_t = tan(pi*(c-0.5))
    return [x0_t, y0_t, σ_t, a1_t, a2_t, b1_t, b2_t, c_t]
end

function forward_2D(stim_sequence, x)
    θ_t = [0.158, 0.51, -1., x[1], x[2], 0.894, 0.949, -0.325]
    return forward_model(stim_sequence, θ_t)
end

function forward_model(stim_sequence, θ_t; isComplex=true)
    x0, y0, σ, a1, a2, b1, b2, c = transform_input(θ_t)
    # a1,a2,b1,b2,c>0, a2b2>a1b1 c<1
    # σ, a1, a2, b1, b2,c = [2.5, 6, 10, 0.8, 0.9, 0.4]
    pRF = get_gaussian(σ,x0,y0)
    N = size(stim_sequence,3)
    time_series = isComplex ? ComplexF64[] : Float64[]  # To accomodate complex differentiation
    for k = 1:N
        # Order of dot is important since first element is conjugated
        push!(time_series,stim_sequence[:,:,k]⋅pRF)
    end
    # println(typeof(time_series))
    t = 0:20
    hrf = hrf_fun(a1, a2, b1, b2, c, t)
    # println(typeof(hrf))
    res = conv(time_series,hrf)
    return res[1:N]
end

function hrf_fun(a1, a2, b1, b2, c, t)
    d1, d2 = a1*b1, a2*b2
    # hrf(t=1) < 0.1, |hrf(t=30)|<0.01
    hrf = (t/d1).^a1.* exp.(-(t .- d1)/b1)- c*(t/d2).^a2 .* exp.(-(t.-d2)/b2);
    return hrf
end

function constraints_2D(x::Array)
    return return(x[1]^2*0.8-x[2]^2*0.9)
end

function constraints(x::Array)
    a1, a2, b1, b2 = x[4]^2, x[5]^2, x[6]^2, x[7]^2
    return [a1*b1-a2*b2]
end

## Utility
function generate_data(x::Array)
    x_t = reverse_input(x)
    stim_sequence = load_stimulus()
    input = forward_model(stim_sequence, x_t, isComplex=false)
    return input
end

## Run model
function test()
    θ = [4, 6, 1+0.1im, 6, 12, 0.9, 0.9, 0.35]
    stim_sequence = load_stimulus()
    res = forward_model(stim_sequence,θ)
    return res
end

# res = test()
