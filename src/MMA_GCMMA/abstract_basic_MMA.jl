
#--------------------------------------------------------------------------------------------------------
# MMAアルゴリズムの抽象型
#--------------------------------------------------------------------------------------------------------
abstract type AbstractMMA end
#--------------------------------------------------------------------------------------------------------
# 目的関数の設定
#--------------------------------------------------------------------------------------------------------
function set_objective_function!(opt::AbstractMMA, target::String, eval::AbstractEvaluateFunction, weight::Float64, s0::Vector{Float64})
    
    # 目的関数の初期値
    f0 = eval.f(s0)
    
    # 最小化問題の場合
    if target == "min"
        f_obj_min(x) = weight * eval.f(x) / f0
        df_obj_min(x) = weight * eval.df(x) / f0
        
        push!(opt.f_obj, (x) -> f_obj_min(x))
        push!(opt.df_obj, (x) -> df_obj_min(x))
    
    # 最大化問題の場合
    elseif target == "max"
        f_obj_max(x) = weight * (-eval.f(x)) / f0
        df_obj_max(x) = weight * (-eval.df(x)) / f0
    
        push!(opt.f_obj, (x) -> f_obj_max(x))
        push!(opt.df_obj, (x) -> df_obj_max(x))
    
    # どちらにも該当しない場合
    else
        error("You can only choose min or max for the objective function setting.")
    end
end
#--------------------------------------------------------------------------------------------------------
# 制約関数の設定
#--------------------------------------------------------------------------------------------------------
function add_inequality_constraint!(opt::AbstractMMA, eval::AbstractEvaluateFunction, f_limit::Float64)
    # 制約関数の定義
    f(x) = eval.f(x) - f_limit
    # 感度式の定義
    df(x) = eval.df(x)
    # 設定
    push!(opt.f_cons, (x) -> f(x))
    push!(opt.df_cons, (x) -> df(x))
    opt.m += 1
end
#--------------------------------------------------------------------------------------------------------
# 目的関数、制約関数の計算+それぞれの感度の計算を行う
#--------------------------------------------------------------------------------------------------------
function compute(opt::AbstractMMA, s::Vector{Float64})
   
    # 目的関数と感度の計算
    f0val = 0.0
    df0dx = zeros(Float64, length(s))
    for i in 1 : length(opt.f_obj)
        f0val += opt.f_obj[i](s)
        df0dx += opt.df_obj[i](s)
    end
   
    # 制約関数の計算
    fval = Vector{Float64}(undef, opt.m)
    dfdx = Matrix{Float64}(undef, opt.m, opt.n)
    for i in 1 : opt.m
        fval[i] = opt.f_cons[i](s)
        dfdx[i, :] = opt.df_cons[i](s)
    end
   
    return f0val, df0dx, fval, dfdx
end