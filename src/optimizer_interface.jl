
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
abstract type AbstractOptimizer end

#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
mutable struct NLOPT <: AbstractOptimizer
    model
    function NLOPT()
        return new(nothing)
    end
end
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
mutable struct MyMMA <: AbstractOptimizer
    model
    asyinit::Float64 
    asyinc::Float64 
    asydec::Float64
    move::Float64
    move_limit::Float64
    function MyMMA(asyinit::Float64=0.5, asyinc::Float64=1.2, asydec::Float64=0.7, move::Float64=0.5, move_limit::Float64=0.1)
        return new(nothing, move_limit, asyinit, asyinc, asydec, move, move_limit)
    end
end
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
mutable struct MyGCMMA <: AbstractOptimizer
    model
    asyinit::Float64 
    asyinc::Float64 
    asydec::Float64
    move::Float64
    move_limit::Float64
    innerit_max::Int64 
    function MyGCMMA(asyinit::Float64=0.5, asyinc::Float64=1.2, asydec::Float64=0.7, move::Float64=0.5, move_limit::Float64=0.1, innerit_max::Int64=30)
        return new(nothing, asyinit, asyinc, asydec, move, move_limit, innerit_max)
    end
end
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
mutable struct MyZPR <: AbstractOptimizer
    model
    move_limit::Float64
    function MyZPR(move_limit::Float64=0.1)
        return new(nothing, move_limit)
    end
end

#--------------------------------------------------------------------------
# 最適化アルゴリズム全体のインターフェースを定義
#--------------------------------------------------------------------------
function set_optimizer!(optimizer::NLOPT, num_x::Int64)
    optimizer.model = Opt(:LD_MMA, num_x)
end
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
function set_option!(optimizer::NLOPT, max_eval::Int64=200, xtol_rel::Float64=1.0e-08, ftol_rel::Float64=1.0e-08)
    optimizer.model.maxeval = max_eval
    optimizer.model.xtol_rel = xtol_rel
    optimizer.model.ftol_rel = ftol_rel
    optimizer.model.params["verbosity"] = 1
end
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
function optimize!(optimizer::NLOPT, physics::AbstractPhysics, opt_settings::OptimizationSettings, s0::Vector{Float64})
    
    # 最適化中の結果を出力する関数を追加(実際に制約を与える訳ではないので、大きな値を設定しておく)
    topology_printer = TopologyPrinter(physics, opt_settings)
    add_inequality_constraint!(optimizer.model, topology_printer, x -> limit_constant(x, 1.0e+20))

    #
    (minf, minx, ret) = optimize(optimizer.model, s0)
    
    # the number of function evaluations
    numevals = optimizer.model.numevals
    println("after $numevals iterations (returned $ret)")
    
    return minf, minx 
end

#--------------------------------------------------------------------------
# 最適化アルゴリズム全体のインターフェースを定義
#--------------------------------------------------------------------------
function set_optimizer!(optimizer::MyMMA, num_x::Int64)
    optimizer.model = MMA(num_x, optimizer.asyinit, optimizer.asyinc, optimizer.asydec, optimizer.move, optimizer.move_limit)
end
#--------------------------------------------------------------------------
# 最適化アルゴリズム全体のインターフェースを定義
#--------------------------------------------------------------------------
function set_option!(optimizer::MyMMA, max_eval::Int64)
    optimizer.model.max_eval = max_eval
end
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
function optimize!(optimizer::MyMMA, physics::AbstractPhysics, opt_settings::OptimizationSettings, s0::Vector{Float64})
    return my_optimize(optimizer.model, physics, opt_settings, s0)
end

#--------------------------------------------------------------------------
# 最適化アルゴリズム全体のインターフェースを定義
#--------------------------------------------------------------------------
function set_optimizer!(optimizer::MyGCMMA, num_x::Int64)
    optimizer.model = GCMMA(num_x, optimizer.asyinit, optimizer.asyinc, optimizer.asydec, optimizer.move, optimizer.move_limit, optimizer.innerit_max)
end
#--------------------------------------------------------------------------
# 最適化アルゴリズム全体のインターフェースを定義
#--------------------------------------------------------------------------
function set_option!(optimizer::MyGCMMA, max_eval::Int64)
    optimizer.model.max_eval = max_eval
end
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
function optimize!(optimizer::MyGCMMA, physics::AbstractPhysics, opt_settings::OptimizationSettings, s0::Vector{Float64})
    return my_optimize(optimizer.model, physics, opt_settings, s0)
end
#--------------------------------------------------------------------------
# 最適化アルゴリズム全体のインターフェースを定義
#--------------------------------------------------------------------------
function set_optimizer!(optimizer::MyZPR, num_x::Int64)
    optimizer.model = ZPR(num_x)
end
#--------------------------------------------------------------------------
# 最適化アルゴリズム全体のインターフェースを定義
#--------------------------------------------------------------------------
function set_option!(optimizer::MyZPR, max_eval::Int64=200)
    #
    optimizer.model.max_eval = max_eval
end
#--------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
function optimize!(optimizer::MyZPR, physics::AbstractPhysics, opt_settings::OptimizationSettings, s0::Vector{Float64})
    #
    return my_optimize(optimizer.model, physics, opt_settings, s0)
end