using AMDGPU
using StaticArrays

function test(a)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    b = MVector{20, Float64}(undef)
    for n = 1:20
        @inbounds b[n] = n
    end

    @inbounds a[i] = sum(b)
    return
end

function test2(a, b)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x

    for n = 1:20
        @inbounds b[n] = n
    end

    @inbounds a[i] = sum(b)
    return
end

a1 = zeros(Float64, 128)
a2 = zeros(Float64, 128)
a = ROCArray(zeros(Float64, 128))
b = ROCArray(zeros(Float64, 128, 20))
@roc groupsize=128 test(a)
copyto!(a1, a)
@show a1
@roc groupsize=128 test2(a, b)
copyto!(a2, a)