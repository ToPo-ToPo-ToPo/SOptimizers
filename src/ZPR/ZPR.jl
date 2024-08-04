

#--------------------------------------------------------------------------------------------------------
# ZPRアルゴリズムの構造体
#--------------------------------------------------------------------------------------------------------
mutable struct ZPR
    n::Int64                                 # 設計変数の総数
    m::Int64                                 # 制約条件の数
    max_eval::Int64                          # 最大ステップ数
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
    function ZPR(n::Int64, max_eval::Int64=200) 
        return new(n, 0, max_eval, Vector{Float64}(), Vector{Float64}(), 0.1, 1.0e-06, 0.5, Vector{Function}(), Vector{Function}(), Vector{Function}(), Vector{Function}())
    end
end
#--------------------------------------------------------------------------------------------------------
# ZPRアルゴリズムのメイン
#--------------------------------------------------------------------------------------------------------
function my_optimize(opt::ZPR, physics, opt_settings, zIni::Vector{Float64})

    # パラメータの初期化
    iter = 0
    zMin = opt.lower_bounds
    zMax = opt.upper_bounds
    Tol = opt.xtol_rel
    z = copy(zIni)
    
    # 配列の初期化
    change = fill(2.0 * opt.xtol_rel, opt.m)
    c = fill(0.001, length(zIni))
    # 1つ前の制約条件における目的関数の感度?
    dfdz0 = zeros(length(zIni))

    # Compute cost functionals and sensitivities
    f0val, dfdz = compute_objective(opt, z)

    # フィルタリング
    filtered_z = filtering(opt_settings.filter, z)
    # vtkファイルに結果を書き出し
    vtk_name = physics.output_file_name * "_ZPR_" * string(iter)
    vtk_datasets = Vector{VtkDataset}()
    push!(vtk_datasets, VtkDataset("design_variables", "CellData", z))
    push!(vtk_datasets, VtkDataset("topology", "CellData", filtered_z))
    push!(vtk_datasets, VtkDataset("objective_sensitivity", "CellData", dfdz))
    output_vtu(physics.nodes, physics.elements, vtk_datasets, vtk_name)

    # Outputs status in progress
    println("=====================================================================")
    println("                              ZPR                                    ")
    println("=====================================================================")
    println("iter      objective     |Violation of Const|      ||Δx||             ")
    println("---------------------------------------------------------------------")

    # main iteration
    while (iter < opt.max_eval) & (norm(change) > Tol)
        #
        iter += 1
        
        # Compute cost functionals and sensitivities
        f, dfdz = compute_objective(opt, z)
        g, dgdz = compute_constraint(opt, z)
        
        # Update design variable and analysis parameters
        # 制約条件ごとに更新
        for j in 1 : opt.m
            # 初期化
            dfdz_eid = Vector{Float64}(undef, opt.n)
            dfdz0_eid = Vector{Float64}(undef, opt.n)
            c_eid = Vector{Float64}(undef, opt.n)
            dgdz_eid = Vector{Float64}(undef, opt.n)
            z_eid = Vector{Float64}(undef, opt.n)
            # 対象の設計変数のみを考慮したベクトルを作成
            for k in 1 : opt.n
                # 要素kにおける設計変数の種類jが保存されている位置を取得
                index = make_index(opt.m, j, k)
                # 値を設定
                dfdz_eid[k] = dfdz[index]
                dfdz0_eid[k] = dfdz0[index]
                c_eid[k] = c[index]
                dgdz_eid[k] = dgdz[j, index]
                z_eid[k] = z[index]
            end
            z_eid, dfdz0_eid, c_eid, change[j] = update_scheme_ss(opt, dfdz_eid, dfdz0_eid, c_eid, g[j], dgdz_eid, z_eid)
            # 全体の配列に戻す
            # 対象の設計変数のみを考慮したベクトルを作成
            for k in 1 : opt.n
                # 要素kにおける設計変数の種類jが保存されている位置を取得
                index = make_index(opt.m, j, k)
                # 値を設定
                z[index] = z_eid[k]
                dfdz0[index] = dfdz0_eid[k]
                c[index] = c_eid[k] 
            end
        end
        # Outputs status in progress
        if iter % 10 == 0
            println("---------------------------------------------------------------------")
            println("iter      objective     |Violation of Const|      ||Δx||             ")
            println("---------------------------------------------------------------------")
        end
        @printf("%3d      %.6e       %.6e       %.6e\n", iter, f, abs(maximum(g)), norm(change))
        # フィルタリング
        filtered_z = filtering(opt_settings.filter, z)
        # vtkファイルに結果を書き出し
        vtk_name = physics.output_file_name * "_ZPR_" * string(iter)
        vtk_datasets = Vector{VtkDataset}()
        push!(vtk_datasets, VtkDataset("design_variables", "CellData", z))
        push!(vtk_datasets, VtkDataset("topology", "CellData", filtered_z))
        push!(vtk_datasets, VtkDataset("objective_sensitivity", "CellData", dfdz))
        output_vtu(physics.nodes, physics.elements, vtk_datasets, vtk_name)
    end
    return f0val, z
end
#--------------------------------------------------------------------------------------------------------
# 目的関数の設定
#--------------------------------------------------------------------------------------------------------
function set_objective_function!(opt::ZPR, target::String, eval::AbstractEvaluateFunction, weight::Float64, s0::Vector{Float64})
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
function add_inequality_constraint!(opt::ZPR, eval::AbstractEvaluateFunction, f_limit::Float64)
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
function compute_objective(opt::ZPR, s::Vector{Float64})
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
function compute_constraint(opt::ZPR, s::Vector{Float64})
    # 制約関数の計算
    fval = Vector{Float64}(undef, opt.m)
    dfdx = Matrix{Float64}(undef, opt.m, opt.n)
    for i in 1 : opt.m
        fval[i] = opt.f_cons[i](s)
        dfdx[i, :] = opt.df_cons[i](s)
    end
    return fval, dfdx
end
#--------------------------------------------------------------------------------------------------------
# 設計変数を更新する部分
#--------------------------------------------------------------------------------------------------------
function update_scheme_ss(opt::ZPR, dfdz, dfdz0, c, g, dgdz, z0)
    # 設計変数の制約値
    zMin = opt.lower_bounds
    zMin_inner = similar(zMin)
    zMax = opt.upper_bounds
    zMax_inner = similar(zMax)
    zNew = similar(zMax)
    # move limit
    move = opt.move_limit * (zMax - zMin)
    eta = opt.eta
    # 更新幅
    dfndz = min.(-abs.(c) .* z0 .* eta, dfdz)
    #B = similar(dfndz)
    zCnd = similar(zNew)
    # Sensitivity separation
    dfpdz = dfdz - dfndz
    # ラグランジュ乗数の初期化
    l1 = 0.0
    l2 = 1.0e+06
    # 設計変数の補正量
    al = 0.75
    
    # inner
    while l2 - l1 > 1.0e-04
        lmid = 0.5 * (l1 + l2)
        #
        for ii in 1 : opt.n
            B = -dfndz[ii] / (dfpdz[ii] + lmid * dgdz[ii])
            zCnd[ii] = zMin[ii] + (z0[ii] - zMin[ii]) * B^eta
        end
        # min and max
        for ii in 1 : opt.n
            #
            min_val = z0[ii] - move[ii]
            if min_val < zMin[ii]
                zMin_inner[ii] = zMin[ii]
            else
                zMin_inner[ii] = min_val
            end 
            #
            max_val = z0[ii] + move[ii]
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
        checker = g + dot(dgdz, zNew - z0)
        if checker > 0.0
            l1 = lmid
        else
            l2 = lmid
        end
    end
    
    yk = dfdz - dfdz0
    sk = zNew - z0
    
    # 設計変数の更新 Damping scheme
    zNew .= al .* zNew .+ (1.0 - al) .* z0
    
    # 変化率の計算
    change = norm(zNew - z0) / norm(zMax - zMin)
    
    # パラメータの更新 Hessian approximation (PSB)
    c .= c .+ ((sk' * yk - c' * sk .^ 2.0) / sum(sk .^ 4.0)) .* (sk .^ 2.0)
    
    return zNew, dfdz, c, change
end