using MPI
using WriteVTK
using LinearAlgebra, StaticArrays, AMDGPU
using JLD2, JSON, HDF5

include("split.jl")
include("schemes.jl")
include("viscous.jl")
include("boundary.jl")
include("utils.jl")
include("div.jl")
include("reactions.jl")
include("thermo.jl")
include("mpi.jl")

function flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, λ, μ, Fhx, Fhy, Fhz, consts)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, dξdx, dξdy, dξdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_x(Fx, ϕ, Fp, Fm, Ncons, consts)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, dηdx, dηdy, dηdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_y(Fy, ϕ, Fp, Fm, Ncons, consts)

    @roc groupsize=nthreads gridsize=ngroups fluxSplit(Q, Fp, Fm, dζdx, dζdy, dζdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_z(Fz, ϕ, Fp, Fm, Ncons, consts)

    @roc groupsize=nthreads gridsize=ngroups viscousFlux_x(Fv_x, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fhx, consts)
    @roc groupsize=nthreads gridsize=ngroups viscousFlux_y(Fv_y, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fhy, consts)
    @roc groupsize=nthreads gridsize=ngroups viscousFlux_z(Fv_z, Q, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, λ, μ, Fhz, consts)

    @roc groupsize=nthreads gridsize=ngroups div(U, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dt, J)
end

function specAdvance(ρi, Q, Yi, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, D, Fhx, Fhy, Fhz, thermo, consts)

    @roc groupsize=nthreads gridsize=ngroups split(ρi, Q, Fp_i, Fm_i, dξdx, dξdy, dξdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_x(Fx_i, ϕ, Fp_i, Fm_i, Nspecs, consts)

    @roc groupsize=nthreads gridsize=ngroups split(ρi, Q, Fp_i, Fm_i, dηdx, dηdy, dηdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_y(Fy_i, ϕ, Fp_i, Fm_i, Nspecs, consts)

    @roc groupsize=nthreads gridsize=ngroups split(ρi, Q, Fp_i, Fm_i, dζdx, dζdy, dζdz)
    @roc groupsize=nthreads gridsize=ngroups WENO_z(Fz_i, ϕ, Fp_i, Fm_i, Nspecs, consts)

    @roc groupsize=nthreads gridsize=ngroups specViscousFlux_x(Fd_x, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fhx, thermo, consts)
    @roc groupsize=nthreads gridsize=ngroups specViscousFlux_y(Fd_y, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fhy, thermo, consts)
    @roc groupsize=nthreads gridsize=ngroups specViscousFlux_z(Fd_z, Q, Yi, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, D, Fhz, thermo, consts)

    @roc groupsize=nthreads gridsize=ngroups divSpecs(ρi, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dt, J)
end

function time_step(rank, comm, thermo, consts, react)
    Nx_tot = Nxp+2*NG
    Ny_tot = Ny+2*NG
    Nz_tot = Nz+2*NG

    # global indices
    lo = rank*Nxp+1
    hi = (rank+1)*Nxp+2*NG
    # lo_ng = rank*Nxp+NG+1
    # hi_ng = (rank+1)*Nxp+NG

    if restart[end-2:end] == ".h5"
        if rank == 0
            printstyled("Restart\n", color=:yellow)
        end
        fid = h5open(restart, "r", comm)
        Q_h = fid["Q_h"][:, :, :, :, rank+1]
        ρi_h = fid["ρi_h"][:, :, :, :, rank+1]
        close(fid)

        Q  =   ROCArray(Q_h)
        ρi =   ROCArray(ρi_h)
    else
        Q_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nprim)
        ρi_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
        Q = AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nprim)
        ρi =AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
        initialize(Q, ρi, thermo)

        copyto!(Q_h, Q)
        copyto!(ρi_h, ρi)
    end
    
    ϕ_h = zeros(Float64, Nx_tot, Ny_tot, Nz_tot) # shock sensor

    # load mesh metrics
    fid = h5open("metrics.h5", "r", comm)
    dξdx_h = fid["dξdx"][lo:hi, :, :]
    dξdy_h = fid["dξdy"][lo:hi, :, :]
    dξdz_h = fid["dξdz"][lo:hi, :, :]
    dηdx_h = fid["dηdx"][lo:hi, :, :]
    dηdy_h = fid["dηdy"][lo:hi, :, :]
    dηdz_h = fid["dηdz"][lo:hi, :, :]
    dζdx_h = fid["dζdx"][lo:hi, :, :]
    dζdy_h = fid["dζdy"][lo:hi, :, :]
    dζdz_h = fid["dζdz"][lo:hi, :, :]

    J_h = fid["J"][lo:hi, :, :] 
    x_h = fid["x"][lo:hi, :, :] 
    y_h = fid["y"][lo:hi, :, :] 
    z_h = fid["z"][lo:hi, :, :]
    close(fid)

    # move to device memory
    dξdx = ROCArray(dξdx_h)
    dξdy = ROCArray(dξdy_h)
    dξdz = ROCArray(dξdz_h)
    dηdx = ROCArray(dηdx_h)
    dηdy = ROCArray(dηdy_h)
    dηdz = ROCArray(dηdz_h)
    dζdx = ROCArray(dζdx_h)
    dζdy = ROCArray(dζdy_h)
    dζdz = ROCArray(dζdz_h)
    J = ROCArray(J_h)

    # allocate on device
    Yi =   AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    ϕ  =   AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot) # Shock sensor
    U  =   AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fp =   AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fm =   AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Ncons)
    Fx =   AMDGPU.zeros(Float64, Nxp+1, Ny, Nz, Ncons)
    Fy =   AMDGPU.zeros(Float64, Nxp, Ny+1, Nz, Ncons)
    Fz =   AMDGPU.zeros(Float64, Nxp, Ny, Nz+1, Ncons)
    Fv_x = AMDGPU.zeros(Float64, Nxp+1, Ny, Nz, 4)
    Fv_y = AMDGPU.zeros(Float64, Nxp, Ny+1, Nz, 4)
    Fv_z = AMDGPU.zeros(Float64, Nxp, Ny, Nz+1, 4)

    Fp_i = AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    Fm_i = AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    Fx_i = AMDGPU.zeros(Float64, Nxp+1, Ny, Nz, Nspecs) # species advection
    Fy_i = AMDGPU.zeros(Float64, Nxp, Ny+1, Nz, Nspecs) # species advection
    Fz_i = AMDGPU.zeros(Float64, Nxp, Ny, Nz+1, Nspecs) # species advection
    Fd_x = AMDGPU.zeros(Float64, Nxp+1, Ny, Nz, Nspecs) # species diffusion
    Fd_y = AMDGPU.zeros(Float64, Nxp, Ny+1, Nz, Nspecs) # species diffusion
    Fd_z = AMDGPU.zeros(Float64, Nxp, Ny, Nz+1, Nspecs) # species diffusion
    Fhx = AMDGPU.zeros(Float64, Nxp+1, Ny, Nz, 3) # enthalpy diffusion
    Fhy = AMDGPU.zeros(Float64, Nxp, Ny+1, Nz, 3) # enthalpy diffusion
    Fhz = AMDGPU.zeros(Float64, Nxp, Ny, Nz+1, 3) # enthalpy diffusion

    μ = AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    λ = AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot)
    D = AMDGPU.zeros(Float64, Nx_tot, Ny_tot, Nz_tot, Nspecs)
    
    Un = similar(U)
    ρn = similar(ρi)

    # MPI buffer 
    Qsbuf_h = zeros(Float64, NG, Ny_tot, Nz_tot, Nprim)
    Qrbuf_h = similar(Qsbuf_h)
    dsbuf_h = zeros(Float64, NG, Ny_tot, Nz_tot, Nspecs)
    drbuf_h = similar(dsbuf_h)
    Mem.pin(Qsbuf_h)
    Mem.pin(Qrbuf_h)
    Mem.pin(dsbuf_h)
    Mem.pin(dsbuf_h)

    Qsbuf_d = ROCArray(Qsbuf_h)
    Qrbuf_d = ROCArray(Qrbuf_h)
    dsbuf_d = ROCArray(dsbuf_h)
    drbuf_d = ROCArray(drbuf_h)

    # initial
    @roc groupsize=nthreads gridsize=ngroups prim2c(U, Q)
    exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
    exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
    MPI.Barrier(comm)
    fillGhost(Q, U, ρi, Yi, thermo, rank)
    fillSpec(ρi)

    if reaction
        if Cantera
            # CPU evaluation needed
            inputs_h = zeros(Float64, Nspecs+2, Nxp*Ny*Nz)
            Mem.pin(inputs_h)
            inputs = ROCArray(inputs_h)
        end

        dt2 = dt/2
    end

    for tt = 1:ceil(Int, Time/dt)
        if tt*dt > Time || tt > maxStep
            return
        end

        if reaction
            # Reaction Step
            if Cantera
                # CPU - cantera
                @roc groupsize=nthreads gridsize=ngroups pre_input_cpu(inputs, Q, ρi)
                copyto!(inputs_h, inputs)
                eval_cpu(inputs_h, dt2)
                copyto!(inputs, inputs_h)
                @roc groupsize=nthreads gridsize=ngroups post_eval_cpu(inputs, U, Q, ρi, thermo)
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
            else
                # GPU
                for _ = 1:sub_step
                    if stiff
                        @roc groupsize=nthreads gridsize=ngroups eval_gpu_stiff(U, Q, ρi, dt2/sub_step, thermo, react)
                    else
                        @roc groupsize=nthreads gridsize=ngroups eval_gpu(U, Q, ρi, dt2/sub_step, thermo, react)
                    end
                end
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
            end
        end

        # RK3
        for KRK = 1:3
            if KRK == 1
                copyto!(Un, U)
                copyto!(ρn, ρi)
            end

            @roc groupsize=nthreads gridsize=ngroups mixture(Q, ρi, Yi, λ, μ, D, thermo)
            @roc groupsize=nthreads gridsize=ngroups shockSensor(ϕ, Q)
            specAdvance(ρi, Q, Yi, Fp_i, Fm_i, Fx_i, Fy_i, Fz_i, Fd_x, Fd_y, Fd_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, D, Fhx, Fhy, Fhz, thermo, consts)
            flowAdvance(U, Q, Fp, Fm, Fx, Fy, Fz, Fv_x, Fv_y, Fv_z, dξdx, dξdy, dξdz, dηdx, dηdy, dηdz, dζdx, dζdy, dζdz, J, dt, ϕ, λ, μ, Fhx, Fhy, Fhz, consts)

            if KRK == 2
                @roc groupsize=nthreads gridsize=ngroups linComb(U, Un, Ncons, 0.25, 0.75)
                @roc groupsize=nthreads gridsize=ngroups linComb(ρi, ρn, Nspecs, 0.25, 0.75)
            elseif KRK == 3
                @roc groupsize=nthreads gridsize=ngroups linComb(U, Un, Ncons, 2/3, 1/3)
                @roc groupsize=nthreads gridsize=ngroups linComb(ρi, ρn, Nspecs, 2/3, 1/3)
            end

            @roc groupsize=nthreads gridsize=ngroups c2Prim(U, Q, ρi, thermo)
            exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
            exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
            MPI.Barrier(comm)
            fillGhost(Q, U, ρi, Yi, thermo, rank)
            fillSpec(ρi)
        end

        if reaction
            # Reaction Step
            if Cantera
                # CPU - cantera
                @roc groupsize=nthreads gridsize=ngroups pre_input_cpu(inputs, Q, ρi)
                copyto!(inputs_h, inputs)
                eval_cpu(inputs_h, dt2)
                copyto!(inputs, inputs_h)
                @roc groupsize=nthreads gridsize=ngroups post_eval_cpu(inputs, U, Q, ρi, thermo)
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
            else
                # GPU
                for _ = 1:sub_step
                    if stiff
                        @roc groupsize=nthreads gridsize=ngroups eval_gpu_stiff(U, Q, ρi, dt2/sub_step, thermo, react)
                    else
                        @roc groupsize=nthreads gridsize=ngroups eval_gpu(U, Q, ρi, dt2/sub_step, thermo, react)
                    end
                end
                exchange_ghost(Q, Nprim, rank, comm, Qsbuf_h, Qsbuf_d, Qrbuf_h, Qrbuf_d)
                exchange_ghost(ρi, Nspecs, rank, comm, dsbuf_h, dsbuf_d, drbuf_h, drbuf_d)
                MPI.Barrier(comm)
                fillGhost(Q, U, ρi, Yi, thermo, rank)
                fillSpec(ρi)
            end
        end

        if tt % 10 == 0
            if rank == 0
                printstyled("Step: ", color=:cyan)
                print("$tt")
                printstyled("\tTime: ", color=:blue)
                println("$(tt*dt)")
                flush(stdout)
            end
            if any(isnan, U)
                printstyled("Oops, NaN detected\n", color=:red)
                return
            end
        end
        # Output
        if plt_out && (tt % step_plt == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
            copyto!(Q_h, Q)
            copyto!(ρi_h, ρi)
            copyto!(ϕ_h, ϕ)

            # visualization file, in Float32
            mkpath("./PLT")
            fname::String = string("./PLT/plt", rank, "-", tt)

            rho = convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1])
            u =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2])
            v =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 3])
            w =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 4])
            p =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 5])
            T =   convert(Array{Float32, 3}, @view Q_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 6])
        
            YH2  = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 1])
            YO2  = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 2])
            YH2O = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 3])
            YH   = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 4])
            YOH  = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 6])
            YN2  = convert(Array{Float32, 3}, @view ρi_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG, 9])

            ϕ_ng = convert(Array{Float32, 3}, @view ϕ_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            x_ng = convert(Array{Float32, 3}, @view x_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            y_ng = convert(Array{Float32, 3}, @view y_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])
            z_ng = convert(Array{Float32, 3}, @view z_h[1+NG:Nxp+NG, 1+NG:Ny+NG, 1+NG:Nz+NG])

            vtk_grid(fname, x_ng, y_ng, z_ng; compress=plt_compress_level) do vtk
                vtk["rho"] = rho
                vtk["u"] = u
                vtk["v"] = v
                vtk["w"] = w
                vtk["p"] = p
                vtk["T"] = T
                vtk["phi"] = ϕ_ng
                vtk["YH2"] = YH2
                vtk["YO2"] = YO2
                vtk["YH2O"] = YH2O
                vtk["YH"] = YH
                vtk["YOH"] = YOH
                vtk["YN2"] = YN2
            end 
        end

        # restart file, in Float64
        if chk_out && (tt % step_chk == 0 || abs(Time-dt*tt) < dt || tt == maxStep)
            copyto!(Q_h, Q)
            copyto!(ρi_h, ρi)

            mkpath("./CHK")
            chkname::String = string("./CHK/chk", tt, ".h5")
            h5open(chkname, "w", comm) do f
                dset1 = create_dataset(
                    f,
                    "Q_h",
                    datatype(Float64),
                    dataspace(Nx_tot, Ny_tot, Nz_tot, Nprim, Nprocs);
                    chunk=(Nx_tot, Ny_tot, Nz_tot, Nprim, 1),
                    compress=chk_compress_level,
                    dxpl_mpio=:collective
                )
                dset1[:, :, :, :, rank + 1] = Q_h
                dset2 = create_dataset(
                    f,
                    "ρi_h",
                    datatype(Float64),
                    dataspace(Nx_tot, Ny_tot, Nz_tot, Nspecs, Nprocs);
                    chunk=(Nx_tot, Ny_tot, Nz_tot, Nspecs, 1),
                    compress=chk_compress_level,
                    dxpl_mpio=:collective
                )
                dset2[:, :, :, :, rank + 1] = ρi_h
            end
        end
    end
    if rank == 0
        printstyled("Done!\n", color=:green)
        flush(stdout)
    end
    MPI.Barrier(comm)
    return
end
