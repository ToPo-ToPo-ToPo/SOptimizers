
using Printf
include("base_kit/mmasub.jl")
include("base_kit/subsolv.jl")
include("base_kit/kktcheck.jl")
#--------------------------------------------------------------------------------------------------------
# MMAに関するデータをまとめた構造体
#--------------------------------------------------------------------------------------------------------
mutable struct MMA <: AbstractMMA
    n::Int64                                 # 設計変数の総数
    m::Int64                                 # 制約条件の数
    max_eval::Int64                          # 最大ステップ
    lower_bounds::Vector{Float64}            # 設計変数の下限制約値
    upper_bounds::Vector{Float64}            # 設計変数の上限制約値
    move_limit::Float64                      # ムーブリミット
    asyinit::Float64
    asyinc::Float64
    asydec::Float64
    move::Float64                            # ムーブリミット
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
    function MMA(
        n::Int64, asyinit::Float64, asyinc::Float64, asydec::Float64, move::Float64, move_limit::Float64,
        max_eval::Int64=200, kkttol=1.0e-06, xtol_rel=0.0, ftol_rel=0.0
        )
        return new(
            n, 0, max_eval, Vector{Float64}(), Vector{Float64}(), 
            move_limit, asyinit, asyinc, asydec, move,
            kkttol, xtol_rel, ftol_rel, 
            Vector{Function}(), Vector{Function}(),
            Vector{Function}(), Vector{Function}()
        )
    end
end
#--------------------------------------------------------------------------------------------------------
# MMAの最適化計算を実行
#--------------------------------------------------------------------------------------------------------
function my_optimize(opt::MMA, physics, opt_settings, xval::Vector{Float64})
    m = opt.m
    n = opt.n
    epsimin = 0.0000001
    xold1   = xval
    xold2   = xval
    xmin    = opt.lower_bounds
    xmax    = opt.upper_bounds
    low     = Vector{Float64}(undef, n)
    upp     = Vector{Float64}(undef, n)
    c       = fill(1.0e+07, m)
    d       = fill(1.0, m)
    a0      = 1
    a       = zeros(Float64, m)
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

    # The iterations start:
    kktnorm = kkttol + 10
    outit = 0
    while (kktnorm > kkttol) & (outit < maxoutit)
        #
        outit = outit + 1
        outeriter = outeriter+1

        # update
        xmin = max.(opt.lower_bounds, xval .- opt.move_limit)
        xmax = min.(opt.upper_bounds, xval .+ opt.move_limit)
        
        # The MMA subproblem is solved at the point xval:
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = mmasub(m,n,outeriter,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,asyinit,asyinc,asydec,move)
        
        # Some vectors are updated:
        xold2 = xold1
        xold1 = xval
        xval  = xmma
        
        # The user should now calculate function values and gradients
        # of the objective- and constraint functions at xval.
        # The results should be put in f0val, df0dx, fval and dfdx.
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
        
    end

    return f0val, xval
end