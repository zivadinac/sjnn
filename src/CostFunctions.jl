__precompile__()

module CostFunctions

export AbstractCostFunction, costFunctionVal, costFunctionDer, Hinge, costFunctionNumDer

# ActivationFunction interface
"Activation function type"
abstract AbstractCostFunction

"Function that calculates the value of specific activation function for all elements of given array"
function costFunctionVal
end

"Function that calculates derivative of specific activation function for all elements of given array"
function costFunctionDer
end

type Hinge <: AbstractCostFunction
end

function costFunctionVal{VarType <: AbstractFloat}(correctY, scores::Array{VarType}, ::Type{Val{Hinge}}, additionalParams...)
    delta = convert(VarType, additionalParams[1][1]); # additionalParams are tupled when passed to network's initParams! function
                                                      # so here we have Typle{Typle{...}} and we need [1] to get that tuple and than
                                                      # [i] on the tuple we got to get real value
    margins = max.(0, delta + scores - scores[correctY])
    margins[correctY] = 0;
    return sum(margins);
end

function costFunctionDer{VarType <: AbstractFloat}(correctY, scores::Array{VarType}, ::Type{Val{Hinge}}, additionalParams...)
    delta = convert(VarType, additionalParams[1][1]);
    ders = (x -> x > 0 ? VarType(1.0) : VarType(0.0)).(Array(delta + scores - scores[correctY]));
    ders[correctY] = 0;
    derY = -sum(ders);
    ders[correctY] = derY;

    return Array(ders);
end

function costFunctionNumDer{VarType <: AbstractFloat}(h, correctY, scores::Array{VarType}, ::Type{Val{Hinge}}, additionalParams...)
    xPlus = copy(scores);
    xMinus = copy(scores);
    grad = zeros(VarType, length(scores));

    for i in 1:length(scores)
        xPlus[i] = scores[i] + h;
        xMinus[i] = scores[i] - h;
        fxPlus = costFunctionVal(correctY, Array(xPlus), Val{Hinge}, additionalParams);
        fxMinus = costFunctionVal(correctY, Array(xMinus), Val{Hinge}, additionalParams);
        xPlus[i] = scores[i];
        xMinus[i] = scores[i];

        grad[i] = (fxPlus - fxMinus) / (2*h);
    end

    return grad;
end

end # module CostFunctions
