module SOptimizers

using NLOPT

include("OC/OC.jl")

include("ZPR/ZPR.jl")

include("MMA_GCMMA/abstract_basic_MMA.jl")
include("MMA_GCMMA/MMA_wrapper.jl")
include("MMA_GCMMA/GCMMA_wrapper.jl")

include("NLOPT/nlopt_wrapper.jl")

include("optimizer_interface.jl")

end # module SOptimizers
