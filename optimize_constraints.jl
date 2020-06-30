using ExprOptimization
using MAT
using Plots

## Grammar
const grammar = @grammar begin
    Real = a1
    Real = a2
    Real = b1
    Real = b2
    Real = c
    Real = Real * Real
    Real = Real + Real
    Real = Real - Real
    Real = Real / Real
    # Real = Real ^ Real
    Real = |(1.:1.:9.)
    # Bool = Bool | Bool
    Bool = Bool & Bool
    Bool = Real < Real
    Bool = Real > Real
end

const S = SymbolTable(grammar)
## Load data
work_file = matopen("../do_work_new.mat")
M_work = read(work_file, "doWork")
close(work_file)
work_file = matopen("../dont_work.mat")
M_dontwork = read(work_file, "dontWork")
close(work_file)

## Rearange generate
M_w = [M_work[i, :] for i in 1:size(M_work,1)]
M_dw = [M_dontwork[i, :] for i in 1:size(M_dontwork,1)]

## Loss function
function loss(tree::RuleNode, grammar::Grammar)
    global COUNT += 1
    if COUNT % 1000 == 0
        println("COUNT: $COUNT")
    end
    ex = get_executable(tree, grammar)
    los = 0.0
    try
        for row in M_w
            S[:a1], S[:a2], S[:b1], S[:b2], S[:c] = row
            los += abs2(Core.eval(S,ex) - true)
        end
        for row in M_dw[rand(1:length(M_dw),length(M_w))]
            S[:a1], S[:a2], S[:b1], S[:b2], S[:c] = row
            los += abs2(Core.eval(S,ex) - false)
        end
    catch err
        println(err)
        los = Inf
    end
    los
end

## Run
COUNT = 0
p_repr = 0.5
p_mute = 0.02
p_cross = 1 - p_mute - p_repr
p = GeneticProgram(2000,100,5,p_repr,p_cross,p_mute)
results_y = optimize(p, grammar, :Bool, loss)
y_loss = results_y.loss
my_expry = results_y.expr
