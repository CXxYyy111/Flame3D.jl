# CUDA-aware MPI not available
function exchange_ghost(Q, NV, rank, comm, sbuf_h, sbuf_d, rbuf_h, rbuf_d)
    nthreads = (NG, 16, 16)
    nblocks = (1, cld((Ny+2*NG), 16), cld((Nz+2*NG), 16))

    # x+
    src = (rank - 1 == -1 ? MPI.PROC_NULL : (rank - 1)) 
    dst = (rank + 1 == Nprocs ? MPI.PROC_NULL : (rank + 1))

    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @roc groupsize=nthreads gridsize=ngroupss pack_R(sbuf_d, Q, NV)
            copyto!(sbuf_h, sbuf_d)
        end
        MPI.Sendrecv!(sbuf_h, rbuf_h, comm; dest=dst, source=src)
        if src != MPI.PROC_NULL
            copyto!(rbuf_d, rbuf_h)
            @roc groupsize=nthreads gridsize=ngroupss unpack_L(rbuf_d, Q, NV)
        end
    end

    # x-
    src = (rank + 1 == Nprocs ? MPI.PROC_NULL : (rank + 1)) 
    dst = (rank - 1 == -1 ? MPI.PROC_NULL : (rank - 1))

    if src != MPI.PROC_NULL || dst != MPI.PROC_NULL
        if dst != MPI.PROC_NULL
            @roc groupsize=nthreads gridsize=ngroupss pack_L(sbuf_d, Q, NV)
            copyto!(sbuf_h, sbuf_d)
        end
        MPI.Sendrecv!(sbuf_h, rbuf_h, comm; dest=dst, source=src)
        if src != MPI.PROC_NULL
            copyto!(rbuf_d, rbuf_h)
            @roc groupsize=nthreads gridsize=ngroupss unpack_R(rbuf_d, Q, NV)
        end
    end
end

function pack_R(buf, Q, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[Nxp+i, j, k, n]
    end
    return
end

function pack_L(buf, Q, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds buf[i, j, k, n] = Q[NG+i, j, k, n]
    end
    return
end

function unpack_L(buf, Q, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds Q[i, j, k, n] = buf[i, j, k, n]
    end
    return
end

function unpack_R(buf, Q, NV)
    i = workitemIdx().x + (workgroupIdx().x - 0x1) * workgroupDim().x
    j = workitemIdx().y + (workgroupIdx().y - 0x1) * workgroupDim().y
    k = workitemIdx().z + (workgroupIdx().z - 0x1) * workgroupDim().z

    if i > NG || j > Ny+2*NG || k > Nz+2*NG
        return
    end

    for n = 1:NV
        @inbounds Q[i+Nxp+NG, j, k, n] = buf[i, j, k, n]
    end
    return
end