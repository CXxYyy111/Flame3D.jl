using AMDGPU, BenchmarkTools

function m1(a, x::Float64)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    b = 1.0
    sum = 0.0
    for n = 1:100
        b = (Float64(n)+0.1)
        b = b*b
        b = b*b*b
        sum += b
    end
    a[i] = sum
    return
end

function m2(a, x::Float64)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    b = 1.0
    sum = 0.0
    for n = 1:100
        b = @fastmath (Float64(n)+0.1)^6
        sum += b
    end
    a[i] = sum
    return
end


a = AMDGPU.zeros(Float64, 800)

@benchmark AMDGPU.@sync @roc groupsize=800 m1(a, 0.12)