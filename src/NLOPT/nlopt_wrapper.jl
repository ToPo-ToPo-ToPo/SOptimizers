
using NLopt
using Enzyme
#--------------------------------------------------------------------------------------------------------
# 目的関数を設定するためのwrapper
#--------------------------------------------------------------------------------------------------------
function set_objective_function!(opt, target::String, eval::AbstractEvaluateFunction, s0::Vector{Float64})
    # 目的関数の初期値
    f0 = eval.f(s0)
    
    # 目的関数の正規化
    f(x) = eval.f(x) / f0
    
    # 感度の正規化
    df(x) = eval.df(x) / f0
    
    # 最大化問題か最小化問題かの設定
    if target == "min"
        opt.min_objective = (x, g) -> change_snopt_format(x, g, f, df)
    elseif target == "max"
        opt.max_objective = (x, g) -> change_snopt_format(x, g, f, df)
    else
        throw(ErrorException("You can only choose min or max for the objective function setting."))
    end
end
#--------------------------------------------------------------------------------------------------------
# 制約関数を追加するためのwrapper
#--------------------------------------------------------------------------------------------------------
function add_inequality_constraint!(opt, eval::AbstractEvaluateFunction, f_limit::Float64)
    # 制約関数の定義
    f(x) = eval.f(x) - f_limit
    # 感度式の定義
    df(x) = eval.df(x)
    # 設定
    inequality_constraint!(opt, (x, g) -> change_snopt_format(x, g, f, df), 1.0e-04)
end
#--------------------------------------------------------------------------------------------------------
# NLOPTを使用するために必要な目的関数のフォーマットを定義
# 自動微分を使用している
#--------------------------------------------------------------------------------------------------------
function change_snopt_format(x::Vector, grad::Vector, f::Function, df::Function)
    # compute sensitivity
    if length(grad) > 0
        dfdx = df(x)
        for i in 1 : length(grad)
            grad[i] = dfdx[i]
        end
    end
    # compute evaluate function
    return f(x)
end
