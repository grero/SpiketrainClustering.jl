# This should probably just use MultivariateStats.KernelPCA

"""
```
function find_max_divergence(x,y)
```
Find a Schoenberg kernel that maximises the divergence between spike trains
`x` and `y`.
"""
function find_max_divergence(x,y;maxiter=1000, rel_tol=sqrt(eps(Float64)),opt=Flux.Optimise.AdaGrad())
    params,kernelc = get_kernel_function(x)
    #here we just maximize divergence directly

    function loss(ps)
        kernel=kernelc(ps)
        -divergence(kernel, x, y)
    end

    L = fill(0.0,maxiter+1)
    L[1] = loss(params)
    prog = Progress(maxiter, dt=1.0)
    stop_idx = 0
    for i in 1:maxiter
        grads = only(Zygote.gradient(loss, params))
        Optimise.update!(opt, params, grads)
        L[i+1] = loss(params)
        if L[i+1] < L[i]
            if abs((L[i+1] - L[i])/L[i]) <= rel_tol
                stop_idx = i
                finish!(prog)
                break
            end
        end
        next!(prog, showvalues=[(:loss, L[i+1])])
    end
    kernelc(params),L[1:stop_idx+1]
end

function center_kernel!(K::Matrix{T}) where T <: Real
    n,c= size(K)
    means1 = vec(mean(K, dims=2))
    means2 = vec(mean(K, dims=1))
    tot = sum(means1)/n
    for i in 1:n
        for j in 1:c
            K[i,j] -= means1[i] + means2[j] - tot
        end
    end
    return K
end

# use the pre-computed kernel for KernelPCA.