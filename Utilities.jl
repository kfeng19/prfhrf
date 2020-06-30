function loss(test, ref, x; λ=0, regularizer::Function)
    # @param x: direct input
    diff = test-ref
    # return log(sum(diff.^2))
    # println(x)
    return sum(diff.^2) + λ*regularizer(real.(x))
end

function quadratic_penalty(c::Function ,x)
    c_vals = c(x)
    res = 0
    for elem in c_vals
        res += real(elem) > 0 ? elem.^2 : 0
    end
    return res
end

function diff_complex(f, x; h=1e-8)
    out = Real[]
    N = length(x)
    for ind in 1:N
        vh = [ind==k ? im*h : 0 for k=1:N]
        push!(out, imag(f(x+vh))/h);
    end
    return out
end
