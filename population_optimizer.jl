using Distributions

mutable struct Particle
    x::Array
    v::Array
    x_best::Array
    y_best::Number
end

function rand_population_cauchy(m, μ, σ)
    n = length(μ)
    return [[rand(Cauchy(μ[j],σ[j])) for j in 1:n] for i in 1:m]
end

function init_particles(f, locations)
    n = length(locations[1])
    return [Particle(locations[k],zeros(n),locations[k],f(locations[k])) for k=1:length(locations)]
end

function particle_swarm_optimization(f, population, k_max; w=1, c1=1, c2=1)
    n = length(population[1].x)
    x_best, y_best = copy(population[1].x_best), Inf
    for P in population
        if P.y_best < y_best
            x_best[:], y_best = P.x, P.y_best
        end
    end
    for k in 1:k_max
        for P in population
            r1, r2 = rand(n), rand(n)
            P.x += P.v
            P.v = w*P.v + c1*r1.*(P.x_best - P.x) + c2*r2.*(x_best-P.x)
            y = f(P.x)
            if y < y_best
                x_best[:], y_best = P.x, y
            end
            if y < P.y_best
                P.x_best[:] = P.x
                P.y_best = y
            end
        end
        println("Iter $k, y_best=$y_best")
    end
    return population, x_best, y_best
end
