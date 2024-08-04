
using Printf
include("base_kit/gcmmasub.jl")
include("base_kit/subsolv.jl")
include("base_kit/kktcheck.jl")
include("base_kit/asymp.jl")
include("base_kit/raaupdate.jl")
include("base_kit/concheck.jl")
#--------------------------------------------------------------------------------------------------------
# MMAに関するデータをまとめた構造体
#--------------------------------------------------------------------------------------------------------
mutable struct GCMMA <: AbstractMMA
    n::Int64                                 # 設計変数の総数
    m::Int64                                 # 制約条件の数
    max_eval::Int64                          # 最大ステップ
    innerit_max::Int64                       # 内部ループの最大ステップ
    lower_bounds::Vector{Float64}            # 設計変数の下限制約値
    upper_bounds::Vector{Float64}            # 設計変数の上限制約値
    asyinit::Float64
    asyinc::Float64
    asydec::Float64
    move::Float64                            
    move_limit::Float64                      # ムーブリミット
    kkttol::Float64                          # 収束判定値
    xtol_rel::Float64                        # 設計変数の変化量に対する制約値
    ftol_rel::Float64                        # 目的関数の変化量に対する制約値
    f_obj::Vector{Function}                  # 目的関数
    df_obj::Vector{Function}                 # 目的関数の感度
    f_cons::Vector{Function}                 # 制約関数
    df_cons::Vector{Function}                # 制約関数の感度
    #---------------------------------------------------------------------
    # MMAに関するデータをまとめた構造体
    # 具体的な関数は別で定義
    #---------------------------------------------------------------------
    function GCMMA(
        n::Int64, asyinit::Float64, asyinc::Float64, asydec::Float64, move::Float64, move_limit::Float64, innerit_max::Int64, 
        max_eval::Int64=200, kkttol=1.0e-06, xtol_rel=0.0, ftol_rel=0.0
        )
        return new(
            n, 0, max_eval, innerit_max, Vector{Float64}(), Vector{Float64}(), 
            asyinit, asyinc, asydec, move, move_limit, 
            kkttol, xtol_rel, ftol_rel,
            Vector{Function}(), Vector{Function}(), 
            Vector{Function}(), Vector{Function}()
        )
    end
end
#--------------------------------------------------------------------------------------------------------
# 目的関数、制約関数の計算+それぞれの感度の計算を行う
#--------------------------------------------------------------------------------------------------------
function compute_inner_loop(opt::GCMMA, s::Vector{Float64})
    # 目的関数の計算
    f0val = 0.0
    for i in 1 : length(opt.f_obj)
        f0val += opt.f_obj[i](s)
    end
    # 制約関数の計算
    fval = Vector{Float64}(undef, opt.m)
    for i in 1 : opt.m
        fval[i] = opt.f_cons[i](s)
    end
    return f0val, fval
end
#--------------------------------------------------------------------------------------------------------
# MMAの最適化計算を実行
#--------------------------------------------------------------------------------------------------------
function my_optimize(opt::GCMMA, physics, opt_settings, xval::Vector{Float64})
    #
    m = opt.m
    n = opt.n
    epsimin = 0.0000001
    xold1   = copy(xval)
    xold2   = copy(xval)
    xmin    = opt.lower_bounds
    xmax    = opt.upper_bounds
    low     = Vector{Float64}(undef, n)
    upp     = Vector{Float64}(undef, n)
    c       = fill(1.0e+07, m)
    d       = fill(1.0, m)
    a0      = 1
    a       = zeros(Float64, m)
    # for gcmma param--------------
    raa0    = 0.01
    raa     = fill(0.01, m)
    raa0eps = 0.000001
    raaeps  = fill(0.000001, m)
    #------------------------------
    outeriter = 0
    maxoutit  = opt.max_eval
    kkttol  = opt.kkttol

    xmma = Vector{Float64}(undef, n)
    ymma = Vector{Float64}(undef, m)
    zmma = 0.0
    lam = Vector{Float64}(undef, m)
    xsi = Vector{Float64}(undef, n)
    eta = Vector{Float64}(undef, n)
    mu = Vector{Float64}(undef, m)
    zet = 0.0

    # パラメータの設定
    asyinit = opt.asyinit
    asyinc = opt.asyinc
    asydec = opt.asydec
    move = opt.move

    # Outputs status in progress
    println("=====================================================================")
    println("                           Basic MMA                                 ")
    println("=====================================================================")
    println("iter      objective         KKT-norm          ||Δx||                 ")
    println("---------------------------------------------------------------------")

    # 初期値を計算
    f0val, df0dx, fval, dfdx = compute(opt, xval)

    # フィルタリング
    filtered_x = filtering(opt_settings.filter, xval)
    # vtkファイルに結果を書き出し
    vtk_name = physics.output_file_name * "_gcmma_" * string(outeriter)
    vtk_datasets = []
    push!(vtk_datasets, VtkDataset("design_variables", "CellData", xval))
    push!(vtk_datasets, VtkDataset("topology", "CellData", filtered_x))
    push!(vtk_datasets, VtkDataset("objective_sensitivity", "CellData", df0dx))
    output_vtu(physics.nodes, physics.elements, vtk_datasets, vtk_name)

    # The outer iterations start:
    kktnorm = kkttol+10
    outit = 0
    while (kktnorm > kkttol) & (outit < maxoutit)
        # update step param
        outit = outit + 1
        outeriter = outeriter + 1

        # update
        xmin = max.(opt.lower_bounds, xval .- opt.move_limit)
        xmax = min.(opt.upper_bounds, xval .+ opt.move_limit)
        
        # The parameters low, upp, raa0 and raa are calculated:
        low, upp, raa0, raa = asymp(outeriter,n,xval,xold1,xold2,xmin,xmax,low,upp,raa0,raa,raa0eps,raaeps,df0dx,dfdx,asyinit,asyinc,asydec)
    
        # The GCMMA subproblem is solved at the point xval:
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(m,n,outeriter,epsimin,xval,xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d,move)
    
        # The user should now calculate function values (no gradients)
        # of the objective- and constraint functions at the point xmma
        # ( = the optimal solution of the subproblem).
        # The results should be put in f0valnew and fvalnew.
        f0valnew, fvalnew = compute_inner_loop(opt, xmma)

        # It is checked if the approximations are conservative:
        conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)
    
        # While the approximations are non-conservative (conserv=0),
        # repeated inner iterations are made:
        innerit=0
        if conserv == 0
            while (conserv == 0) & (innerit <= opt.innerit_max)
                innerit = innerit+1;
                
                # New values on the parameters raa0 and raa are calculated:
                raa0, raa = raaupdate(xmma,xval,xmin,xmax,low,upp,f0valnew,fvalnew,f0app,fapp,raa0,raa,raa0eps,raaeps,epsimin)
                
                # The GCMMA subproblem is solved with these new raa0 and raa:
                xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(m,n,outeriter,epsimin,xval,xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d,move)

                # The user should now calculate function values (no gradients)
                # of the objective- and constraint functions at the point xmma
                # ( = the optimal solution of the subproblem).
                # The results should be put in f0valnew and fvalnew:
                f0valnew, fvalnew = compute_inner_loop(opt, xmma)
    
                # It is checked if the approximations have become conservative:
                conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)
            end
        end

        # No more inner iterations. Some vectors are updated:
        xold2 = xold1
        xold1 = xval
        xval  = xmma
        
        # The user should now calculate function values and gradients
        # of the objective- and constraint functions at xval.
        # The results should be put in f0val, df0dx, fval and dfdx:
        f0val, df0dx, fval, dfdx = compute(opt, xval)

        # The residual vector of the KKT conditions is calculated:
        residu, kktnorm, residumax = kktcheck(m,n,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,xmin,xmax,df0dx,fval,dfdx,a0,a,c,d)

        # Outputs status in progress
        if outeriter % 10 == 0
            println("---------------------------------------------------------------------")
            println("iter      objective         KKT-norm          ||Δx||                 ")
            println("---------------------------------------------------------------------")
        end
        @printf("%3d      %.6e     %.6e     %.6e\n", outit, f0val, kktnorm, norm(xmma-xold1))

        # フィルタリング
        filtered_x = filtering(opt_settings.filter, xval)
        # vtkファイルに結果を書き出し
        vtk_name = physics.output_file_name * "_gcmma_" * string(outeriter)
        vtk_datasets = []
        push!(vtk_datasets, VtkDataset("design_variables", "CellData", xval))
        push!(vtk_datasets, VtkDataset("topology", "CellData", filtered_x))
        push!(vtk_datasets, VtkDataset("objective_sensitivity", "CellData", df0dx))
        output_vtu(physics.nodes, physics.elements, vtk_datasets, vtk_name)

    end

    return f0val, xval
end