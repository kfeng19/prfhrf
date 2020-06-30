include(joinpath(@__DIR__,"Utilities.jl"))

function Ip_gradient(f_obj::Function, g_obj::Function, c::Function, x0::Array, max_n, config_dict; ρ=1, γ=5, debug=false, method::Function)
    # println("x0=$x0")
    barrier(x) = -sum([real(ci)≥-1 ? log(-ci) : 0 for ci in c(x)])
    Δ = Inf
    # ϵ = 1e-3
    ϵ = config_dict["ϵ"]
    out = []
    iter = 1
    cg_scale = config_dict["cg_scale0"]
    while iter ≤ max_n && Δ > ϵ
        println("\niterration $iter, ρ=$ρ")
        f_complete(x) = f_obj(x) + 1/ρ * barrier(x)
        g_complete(x) = g_obj(x) + 1/ρ * diff_complex(barrier, x)
        # x_hist = cg(f_complete, g_complete, c, x0, max_iter=config_dict["cg_max_iter"], β_backtracking=0.1, g_th=config_dict["g_th"], debug=debug,scale=cg_scale)
        x_hist, mean_δ = method(f_complete, g_complete, c, x0, β_backtracking=0.1, scale=cg_scale, config_dict=config_dict)
        append!(out, x_hist)
        x1 = last(x_hist)
        Δ = norm(x1-x0)
        println("Δ=$Δ")
        if Δ < ϵ    # Terminate with small cg improvement
            println("Terminating with Δ=$Δ")
            break
        end
        x0 = x1
        ρ *= γ
        iter += 1
        # backtracking scale too large may lead to spikes at beginning of each IP iteration
        # cg_scale *= 0.5;
        # cg_scale = Δ < config_dict["cg_scale0"] ? 0.5 * cg_scale + 0.5 * Δ : config_dict["cg_scale0"]
        println("mean_δ=$mean_δ")
        cg_scale = min(mean_δ, config_dict["cg_scale0"])
    end
    return out
end

function bfgs(f::Function, g::Function, c::Function, x0; β_backtracking, scale=1., config_dict)
    max_iter = config_dict["cg_max_iter"]
    g_th = config_dict["g_th"]
    max_scale = config_dict["cg_scale0"]
    x_hist = [x0]
    m = length(x0)
    Q = Matrix(1.0I, m, m)
    g0 = g(x0);
    mean_δ = 0
    for k=1:max_iter
        println("bfgs iter $k")
        g_norm = norm(g0)
        println("|g|: $g_norm")
        # println("g: $g0")
        log_loss = log(f(x0))
        println("loss: $log_loss" )
        # println("x: $x0")
        if g_norm < g_th
            println("Terminating cg with |g|=$g_norm, iter=$k")
            println("Terminate x: ", x0)
            return x_hist, mean_δ
        end
        d0 = -Q*g0
        if isnan(norm(d0))
            println("Q: $Q")
            println("g0: $g0")
            throw(ErrorException)
        end
        # println("d: $d0")
        x1, counter = backtracking_line_search(f, g, c, x0, d=d0, β=β_backtracking, scale=scale, p=0.5);
        g1 = g(x1);
        δx = x1 - x0;
        norm_dx = norm(δx)
        if norm_dx < eps()
            # dx too small can result in NaN Q
            println("Too small progress in BFGS: $norm_dx")
            break
        end
        mean_δ = (k == 1 ? norm_dx : (mean_δ * 0.9 + norm_dx * 0.1))
        δg = g1 - g0;
        Q[:] = Q - (δx*δg'*Q + Q*δg*δx') / (δx'*δg) + (1 + (δg'*Q*δg)/(δx'*δg))[1]*(δx*δx')/(δx'*δg)
        if isnan(norm(Q))
            println("x0: $x0")
            println("x1: $x1")
            println("δx: $δx")
            println("g0: $g0")
            println("g1: $g1")
            println("δg: $δg")
            throw(ErrorException)
        end
        x0 = x1;
        g0 = g1;
        push!(x_hist,x0);
        # extreme multipliers (both up & down scaling) leads to unstable behavior
        if counter ≥ 10
            scale *= 0.9
        elseif counter < 2
            scale *= 1.1
        end
        scale = min(scale, max_scale)
    end
    return x_hist, mean_δ;
end

function cg(f::Function, g::Function, c::Function, x0; β_backtracking, debug=true, scale=1., config_dict)
    max_iter = config_dict["cg_max_iter"]
    g_th = config_dict["g_th"]
    x_hist = [x0];
    g0 = g(x0);
    g_norm = norm(g0)
    println("|g|: $g_norm")
    println("g: $g0")
    if g_norm < g_th
        println("Terminating cg with |g|=$g_norm, iter=0")
        return x_hist
    end
    # println("cg g0=$g0")
    d0 = -g0;
    # println(x0)
    x1, counter = backtracking_line_search(f, g, c, x0, d=-g0, β=β_backtracking, scale=scale);
    dx_norm = norm(x1-x0)
    push!(x_hist, x1);
    mean_δ = dx_norm
    x0=x1;
    # feasible = false
    for k = 1:max_iter-1
        println("cg iter $k")
        g1 = g(x0);
        g_norm = norm(g1)
        println("|g|: $g_norm")
        println("g: $g0")
        if g_norm < g_th
            if debug
                println("Terminating cg with |g|=$g_norm, iter=$k")
            end
            break;
        end
        β = max(0, g1⋅(g1-g0)/(g0⋅g0)); # ?
        d1 = -g1 + β*d0;
        x1, counter = backtracking_line_search(f, g, c, x0, d=d1, β=β_backtracking, scale=scale);
        push!(x_hist,x1)
        mean_δ = mean_δ * 0.9 + norm(x1-x0) * 0.1
        x0 = x1;
        d0 = d1;
        g0 = g1;
        # scale *= 0.95
        if counter ≥ 10
            scale *= 0.9
        elseif counter < 2
            scale *= 1.1
        end
    end
    return x_hist, mean_δ;
end

function backtracking_line_search(f::Function, g::Function, c::Function, x::Array; d::Array, p::Real, β::Real, n::Int=10, scale=1.)
    # @param n: iterations of backtracking. Must be large enough to enable small steps appear and trigger cg to stop
    # @return x+αd: accepted design point
    println("scale=$scale")
    d_norm = norm(d)
    println("dnorm=$d_norm")
    d /= d_norm+eps()
    # α = d_norm > scale ? scale/d_norm : scale
    α = scale # α defines the step size
    counter = 0;
    f0 = f(x);
    g0 = g(x);  # Gradient
    penalty = quadratic_penalty(c, x+α*d)
    # Consider both cases of in&out feasible region
    while (penalty > 0) || (counter < n && f(x+α.*d) > f0 + β*α*(g0 ⋅ d))
        counter += 1;
        # p too small mey lead to minimal progress
        α *= p;
        penalty = quadratic_penalty(c, x+α*d)
        if counter > 100
            println("Warning, too many iterations in backtracking")
            break;
        end
    end
    # if debug
    println("α=$α")
    x1 = x+α*d
    for elem in x1
        if isnan(elem)
            println("x: $x");
            println("α: $α")
            println("d: $d")
            throw(ErrorException)
        end
    end
    println("$counter iterations in backtracking line search.\n")
    # println("object = ", f(x1))
    # println("x+αd=$x1")
    return x1, counter;
end
