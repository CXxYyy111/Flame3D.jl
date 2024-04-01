# Only 1D-x direction MPI for now
include("solver.jl")

# LES
const LES_smag::Bool = false       # if use Smagorinsky model
const LES_wale::Bool = false        # if use WALE model

# thermal state
const γ::Float32 = 1.4
const Rg::Float32 = 287
const Cp::Float32 = Rg*γ/(γ-1)
const C_s::Float32 = 1.458f-6
const T_s::Float32 = 110.4
const Pr::Float32 = 0.7

# flow control
const Nprocs::SVector{3, Int64} = [1,1,1] # number of GPUs
const Iperiodic = (false, false, true)
const dt::Float32 = 1.5f-8             # dt for simulation, make CFL < 1
const Time::Float32 = 1f-3           # total simulation time
const maxStep::Int64 = 100         # max steps to run

const plt_out::Bool = true           # if output plt file
const step_plt::Int64 = 1000          # how many steps to save plt
const plt_compress_level::Int64 = 1  # output file compression level 0-9, 0 for no compression

const chk_out::Bool = false           # if checkpoint is made on save
const step_chk::Int64 = 2000          # how many steps to save chk
const chk_compress_level::Int64 = 1  # checkpoint file compression level 0-9, 0 for no compression
const restart::String = "./CHK/chk38000.h5"     # restart use checkpoint, file name "*.h5" or "none"

const average = false                 # if do average
const avg_step = 10                  # average interval
const avg_total = 1000                # total number of samples

# do not change 
const Ncons::Int64 = 5 # ρ ρu ρv ρw E 
const Nprim::Int64 = 6 # ρ u v w p T
# scheme constant
const WENOϵ::Float32 = eps(1f-12)
const hybrid_ϕ1::Float32 = 5f-2
const hybrid_ϕ2::Float32 = 10.f0
const UP7::SVector{7, Float32} = SVector(-3/420, 25/420, -101/420, 319/420, 214/420, -38/420, 4/420)

# load mesh info
const NG::Int64 = h5read("metrics.h5", "NG")
const Nx::Int64 = h5read("metrics.h5", "Nx")
const Ny::Int64 = h5read("metrics.h5", "Ny")
const Nz::Int64 = h5read("metrics.h5", "Nz")
const Nxp::Int64 = Nx ÷ Nprocs[1] # make sure it is integer
const Nyp::Int64 = Ny ÷ Nprocs[2] # make sure it is integer
const Nzp::Int64 = Nz ÷ Nprocs[3] # make sure it is integer
# here we use 512 threads/block and limit registers to 128
const maxreg::Int64 = 128
const nthreads::Tuple{Int32, Int32, Int32} = (8, 8, 8)
const nblock::Tuple{Int32, Int32, Int32} = (cld((Nxp+2*NG), 8), 
                                            cld((Nyp+2*NG), 8),
                                            cld((Nzp+2*NG), 8))


# Run the simulation
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nGPU = MPI.Comm_size(comm)
shmcomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
local_rank = MPI.Comm_rank(shmcomm)
comm_cart = MPI.Cart_create(comm, Nprocs; periodic=Iperiodic)
if nGPU != Nprocs[1]*Nprocs[2]*Nprocs[3] && rank == 0
    error("Oops, nGPU ≠ $Nprocs\n")
end
# set device on each MPI rank
# device!(local_rank)

time_step(rank, comm_cart)

MPI.Finalize()
