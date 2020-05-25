include(joinpath(@__DIR__,"Utilities.jl"))

function barrier_cg(f_obj::Function, g_obj::Function, c::Function, x0::Array, max_n; ρ=1, γ=5, debug=false)
    # println("x0=$x0")
    barrier(x) = -sum([real(ci)≥-1 ? log(-ci) : 0 for ci in c(x)])
    Δ = Inf
    ϵ = 1e-2
    out = []
    iter = 0
    while iter < max_n && Δ > ϵ
        println("\niterration $iter, ρ=$ρ")
        f_complete(x) = f_obj(x) + 1/ρ * barrier(x)
        g_complete(x) = g_obj(x) + 1/ρ * diff_complex(barrier, x)
        x_hist = cg(f_complete, g_complete, c, x0, max_iter=50, β_backtracking=0.1, debug=debug)
        append!(out, x_hist)
        x1 = last(x_hist)
        # println("x0=",x0)
        # println("x1=",x1)
        Δ = norm(x1-x0)
        println("Δ=$Δ")
        if Δ < ϵ    # Terminate with small cg improvement
            println("Terminating with Δ=$Δ")
            break
        end
        x0 = x1
        ρ *= γ
        iter += 1
    end
    # println("ρ=10^",log10(ρ))
    return out
end

function cg(f::Function, g::Function, c::Function, x0; max_iter::Int, β_backtracking, debug=false)
    # println("cg x0=$x0")
    g_th = 1e5;
    ϵ=1e-3; # Termination condition
    x_hist = [x0];
    g0 = g(x0);
    g_norm = norm(g0)
    println("g: $g_norm")
    if g_norm < g_th
        println("Terminating cg with |g|=$g_norm, iter=0")
        return x_hist
    end
    # println("cg g0=$g0")
    d0 = -g0;
    # println(x0)
    x1 = backtracking_line_search(f, g, c, x0, d=-g0, n=10, β=β_backtracking);
    dx_norm = norm(x1-x0)
    # if dx_norm<ϵ  # Better not terminate on step size
    #     println("Breaking cg with small step size $dx_norm, iter=0")
    #     return x_hist
    # end
    push!(x_hist, x1);
    x0=x1;
    # feasible = false
    for k = 1:max_iter-1
        # println("x0=$x0")
        g1 = g(x0);
        g_norm = norm(g1)
        if g_norm < g_th
            if debug
                println("Terminating cg with |g|=$g_norm, iter=$k")
            end
            break;
        end
        β = max(0, g1⋅(g1-g0)/(g0⋅g0)); # ?
        d1 = -g1 + β*d0;
        x1 = backtracking_line_search(f, g, c, x0, d=d1, β=β_backtracking);
        push!(x_hist,x1)
        # The termination condition must be consistent with backtracking iterations, otherwise will never be reached. e.g. ϵ>1/2^n
        # dx_norm = norm(x1-x0)
        # if dx_norm<ϵ # Better not terminate on step size
        #     println("Breaking cg with small step size $dx_norm, iter=$k")
        #     break;
        # end
        x0 = x1;
        d0 = d1;
        g0 = g1;
    end
    return x_hist;
end

function backtracking_line_search(f::Function, g::Function, c::Function, x::Array; d::Array, p::Real=0.5, β::Real, n::Int=10, scale=1.)
    # @param n: iterations of backtracking. Must be large enough to enable small steps appear and trigger cg to stop
    # @return x+αd: accepted design point
    d_norm = norm(d)
    α = d_norm > 1 ? scale/d_norm : scale
    # α=1/(d_norm+1e-6)
    counter = 0;
    f0 = f(x);
    g0 = g(x);  # Gradient
    penalty = quadratic_penalty(c, x+α*d)
    # Consider both cases of in&out feasible region
    while (penalty > 0) || (counter < n && f(x+α.*d) > f0 + β*α*(g0 ⋅ d))
        counter += 1;
        α *= p;
        penalty = quadratic_penalty(c, x+α*d)
        if counter > 100
            println("Warning, too many iterations in backtracking")
            break;
        end
    end
    # if debug
    # println("α=$α")
    # println("$counter iterations in backtracking line search.")
    return x+α*d;
end
