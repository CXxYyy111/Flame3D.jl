using AMDGPU, StaticArrays

function kernel(C, A)
    Ctmp = @ROCStaticLocalArray(Float64, (10, 80)) # if we want 8 elements

    idx = workitemIdx().x
    Ctmp[1, idx] = A[idx] + C[1]
    AMDGPU.Device.sync_workgroup()

    C[idx] = Ctmp[1, idx]
    return
end

function kernel1(C, A)
    Ctmp = @MMatrix zeros(Float64, 10, 80)

    idx = workitemIdx().x
    Ctmp[1, idx] = A[idx] + C[1]
    AMDGPU.Device.sync_workgroup()

    C[idx] = Ctmp[1, idx]
    return
end

RC = AMDGPU.rand(Float64, 80)
RA = AMDGPU.rand(Float64, 80)
@roc groupsize=80 kernel(RC, RA)
@roc groupsize=80 kernel1(RC, RA)