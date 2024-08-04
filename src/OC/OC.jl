
#--------------------------------------------------------------------------------------------------------
# OCアルゴリズムの構造体
#--------------------------------------------------------------------------------------------------------
mutable struct OC
    n::Int64                                 # 設計変数の総数
    m::Int64                                 # 制約条件の数
    maxeval::Int64                           # 最大ステップ数
    lower_bounds::Vector{Float64}            # 設計変数の下限制約値
    upper_bounds::Vector{Float64}            # 設計変数の上限制約値
    move_limit::Float64                      # ムーブリミット
    xtol_rel::Float64                        # 目的関数の変化量に対する制約値
    eta::Float64                             # 設計変数の更新パラメータ
    f_obj::Vector{Function}                  # 目的関数
    df_obj::Vector{Function}                 # 目的関数の感度
    f_cons::Vector{Function}                 # 制約関数
    df_cons::Vector{Function}                # 制約関数の感度
    #---------------------------------------------------------------------
    # 内部コンストラクタ
    #---------------------------------------------------------------------
    function OC(n::Int64, maxeval::Int64=200) 
        return new(n, 0, maxeval, Vector{Float64}(), Vector{Float64}(), 0.1, 1.0e-06, 0.5, Vector{Function}(), Vector{Function}(), Vector{Function}(), Vector{Function}())
    end
end
#--------------------------------------------------------------------------------------------------------
# ZPRアルゴリズムのメイン
#--------------------------------------------------------------------------------------------------------
function my_optimize(opt::OC, zIni::Vector{Float64}, nodes, elements, filter)

    # パラメータの初期化
    iter = 0
    zMin = opt.lower_bounds
    zMax = opt.upper_bounds
    zMin_inner = similar(zMin)
    zMax_inner = similar(zMax)
    move = opt.move_limit * (zMax - zMin)
    Tol = opt.xtol_rel
    z = copy(zIni)
    zCnd = similar(z)
    zNew = similar(zCnd)
    change = 2.0 * Tol

    # Compute cost functionals and sensitivities
    f0val, dfdz = compute_objective(opt, z)

    # フィルタリング
    filtered_z = filtering(filter, z)
    # vtkファイルに結果を書き出し
    vtk_name = "output/vtu_files/oc_opt_step_" * string(iter)
    vtk_datasets = []
    push!(vtk_datasets, VtkDatasets("design_variables", "CellData", z))
    push!(vtk_datasets, VtkDatasets("topology", "CellData", filtered_z))
    push!(vtk_datasets, VtkDatasets("objective_sensitivity", "CellData", dfdz))
    output_vtu(nodes, elements, vtk_datasets, vtk_name)

    # Outputs status in progress
    println("=====================================================================")
    println("                             OC                                      ")
    println("=====================================================================")
    println("iter      objective     |Violation of Const|      ||Δx||             ")
    println("---------------------------------------------------------------------")

    # main iteration
    while (iter < opt.maxeval) & (norm(change) > Tol)
        #
        iter += 1
        
        # Compute cost functionals and sensitivities
        f, dfdz = compute_objective(opt, z)
        g, dgdz = compute_constraint(opt, z)
        
        # Update design variable and analysis parameters
        # ラグランジュ乗数の初期化
        l1 = 0.0
        l2 = 1.0e+06
    
        # inner
        while l2 - l1 > 1.0e-04
            lmid = 0.5 * (l1 + l2)
            #
            for ii in 1 : opt.n
                B_in = - dfdz[ii] / (lmid * dgdz[opt.m, ii])
                if B_in < 0.0
                    B_in = 0.0
                end
                B = B_in^opt.eta
                zCnd[ii] = z[ii] * B
            end
            # min and max
            for ii in 1 : opt.n
                #
                min_val = z[ii] - move[ii]
                if min_val < zMin[ii]
                    zMin_inner[ii] = zMin[ii]
                else
                    zMin_inner[ii] = min_val
                end 
                #
                max_val = z[ii] + move[ii]
                if max_val > zMax[ii]
                    zMax_inner[ii] = zMax[ii]
                else
                    zMax_inner[ii] = max_val
                end
            end

            # update
            for ii in 1 : opt.n
                val = zCnd[ii]
                if val < zMin_inner[ii]
                    zNew[ii] = zMin_inner[ii]
                elseif zMax_inner[ii] < val
                    zNew[ii] = zMax_inner[ii]
                else
                    zNew[ii] = val
                end
            end
                
            # ラグランジュ乗数の更新
            g, _ = compute_constraint(opt, zNew)
            if g[opt.m] > 0.0
                l1 = lmid
            else
                l2 = lmid
            end
        end

        #
        change = norm(zNew .- z)

        # update
        z .= zNew
        f0val = f
        
        # Outputs status in progress
        if iter % 10 == 0
            println("---------------------------------------------------------------------")
            println("iter      objective     |Violation of Const|      ||Δx||             ")
            println("---------------------------------------------------------------------")
        end
        @printf("%3d      %.6e       %.6e       %.6e\n", iter, f, abs(maximum(g)), change)
        # フィルタリング
        filtered_z = filtering(filter, z)
        # vtkファイルに結果を書き出し
        vtk_name = "output/vtu_files/oc_opt_step_" * string(iter)
        vtk_datasets = []
        push!(vtk_datasets, VtkDatasets("design_variables", "CellData", z))
        push!(vtk_datasets, VtkDatasets("topology", "CellData", filtered_z))
        push!(vtk_datasets, VtkDatasets("objective_sensitivity", "CellData", dfdz))
        output_vtu(nodes, elements, vtk_datasets, vtk_name)
    end
    return f0val, z
end
#--------------------------------------------------------------------------------------------------------
# 目的関数の設定
#--------------------------------------------------------------------------------------------------------
function add_objective_function!(opt::OC, target::String, eval::AbstractEvaluateFunction, weight::Float64, s0::Vector{Float64})
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
function add_inequality_constraint!(opt::OC, eval::AbstractEvaluateFunction, f_limit::Float64)
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
# 目的関数とその感度の計算を行う
#--------------------------------------------------------------------------------------------------------
function compute_objective(opt::OC, s::Vector{Float64})
    # 目的関数と感度の計算
    f0val = 0.0
    df0dx = zeros(Float64, length(s))
    for i in 1 : length(opt.f_obj)
        f0val += opt.f_obj[i](s)
        df0dx += opt.df_obj[i](s)
    end
    #
    return f0val, df0dx
end
#--------------------------------------------------------------------------------------------------------
# 目的関数、制約関数の計算+それぞれの感度の計算を行う
#--------------------------------------------------------------------------------------------------------
function compute_constraint(opt::OC, s::Vector{Float64})
    # 制約関数の計算
    fval = Vector{Float64}(undef, opt.m)
    dfdx = Matrix{Float64}(undef, opt.m, opt.n)
    for i in 1 : opt.m
        fval[i] = opt.f_cons[i](s)
        dfdx[i, :] = opt.df_cons[i](s)
    end
    return fval, dfdx
end