
using Ipopt
using Enzyme
#--------------------------------------------------------------------------
# ipoptのモデルを管理する構造体
#--------------------------------------------------------------------------
mutable struct IpoptModel
    objective_function::Function                      # 目的関数
    constraint_functions::Vector{Function}            # 制約関数(複数考慮可能)
    constrain_lower_bounds::Vector{Float64}           # 制約関数の下限値をまとめた配列
    constrain_upper_bounds::Vector{Float64}           # 制約関数の上限値をまとめた配列
    lower_bounds::Vector{Float64}                     # 変数の下限値
    upper_bounds::Vector{Float64}                     # 変数の上限値
    #-------------------------------------------------------------
    # 内部コンストラクタ
    #-------------------------------------------------------------
    function IpoptModel()
        return new(nothing, Vector{Function}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}())
    end
end
#--------------------------------------------------------------------------
# 最適化を実行する関数
#--------------------------------------------------------------------------
function optimize(model::IpoptModel, x::Vector{Float64})

    # 制約関数式の更新
    eval_g_new(x, g) = eval_g(x, g, model.constraint_functions)

    # 目的関数の感度式の更新
    eval_grad_f_new(x, g) = eval_grad_f(x, g, model.objective_function)

    # 制約関数の感度式の更新
    eval_jac_g_new(x, rows, cols, values) = eval_jac_g(x, rows, cols, values, model.constraint_functions)

    # 実際のipoptの設定を入力
    prob = Ipopt.CreateIpoptProblem(
        length(x),                                         # 変数の数
        model.lower_bounds,                                # 変数の下限制約値
        model.upper_bounds,                                # 変数の上限制約値
        length(model.constraint_functions),                # 制約関数の数
        model.constrain_lower_bounds,                      # 制約関数の下限値 
        model.constrain_upper_bounds,                      # 制約関数の上限値 
        length(x) * length(model.constraint_functions),    # 制約関数の感度の数
        10,                                                # ヘッセ行列の非ゼロ成分の数
        model.objective_function,                          # 目的関数
        eval_g_new,                                        # 制約関数
        eval_grad_f_new,                                   # 目的関数の感度
        eval_jac_g_new,                                    # 制約関数の感度
        eval_h                                             # ヘッセ行列
    )
    
    # 変数を指定
    prob.x = x

    # 最適化の実行
    Ipopt.IpoptSolve(prob)

    # 最適化結果を返す
    return prob.obj_val, prob.x
end
#--------------------------------------------------------------------------
# 不等式制約を追加する関数
# 上限制約を与える関数とする
# 式は必ず以下のように与える
# f_cons(x) = f(x) - fmax <= 0
# 下限制約を与えたい場合は以下のようにする
# f_cons(x) = fmax - f(x) <= 0
#--------------------------------------------------------------------------
function add_inequality_constraint!(model::IpoptModel, f_cons::Function)
    # 関数を追加
    push!(model.constraint_functions, f_cons)
    # 制約関数の下限値を追加
    push!(model.constrain_lower_bounds, -2.0e+19)
    # 制約関数の上限値を追加
    push!(model.constrain_upper_bounds, 1.0e-04)
end
#--------------------------------------------------------------------------
# 制約関数の計算式
# すべての制約条件の数だけの配列に計算結果を定義する
# Bad: g    = zeros(2)  # Allocates new array
# OK:  g[:] = zeros(2)  # Modifies 'in place'
#--------------------------------------------------------------------------
function eval_g(x::Vector{Float64}, g::Vector{Float64}, f_cons::Vector{Function})
    for i in 1 : length(f_cons)
        g[i] = f_cons(x)[i]
    end
    return g
end
#--------------------------------------------------------------------------
# 目的関数の感度の計算式
# Bad: grad_f    = zeros(4)  # Allocates new array
# OK:  grad_f[:] = zeros(4)  # Modifies 'in place'
#--------------------------------------------------------------------------
function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64}, f_obj::Function)
    grad_f[:] = gradient(Reverse, f_obj, x)
    return grad_f
end
#--------------------------------------------------------------------------
# 制約関数の感度の計算式
# 疎行列形式で値を入力する
#--------------------------------------------------------------------------
function eval_jac_g(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, values::Union{Nothing,Vector{Float64}}, f_cons::Vector{Function})

    if values === nothing
        # Constraint (row) 1
        rows[1] = 1
        cols[1] = 1
        rows[2] = 1
        cols[2] = 2
        rows[3] = 1
        cols[3] = 3
        rows[4] = 1
        cols[4] = 4
        # Constraint (row) 2
        rows[5] = 2
        cols[5] = 1
        rows[6] = 2
        cols[6] = 2
        rows[7] = 2
        cols[7] = 3
        rows[8] = 2
        cols[8] = 4
    else
        # Constraint (row) 1
        values[1] = x[2] * x[3] * x[4]  # 1,1
        values[2] = x[1] * x[3] * x[4]  # 1,2
        values[3] = x[1] * x[2] * x[4]  # 1,3
        values[4] = x[1] * x[2] * x[3]  # 1,4
        # Constraint (row) 2
        values[5] = 2 * x[1]  # 2,1
        values[6] = 2 * x[2]  # 2,2
        values[7] = 2 * x[3]  # 2,3
        values[8] = 2 * x[4]  # 2,4
    end
    return
end
#--------------------------------------------------------------------------
# ヘッセ行列の計算式
# 特に値は定義せずとも最適化できる（ただし、関数自体の定義は必要）
#--------------------------------------------------------------------------
function eval_h(x::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Union{Nothing,Vector{Float64}})
    if values === nothing
        # Symmetric matrix, fill the lower left triangle only
        idx = 1
        for row in 1 : length(x)
            for col in 1 : row
                rows[idx] = row
                cols[idx] = col
                idx += 1
            end
        end
    else
        
    end
    return nothing
end