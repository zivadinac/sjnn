__precompile__()

module ActivationFunctions

export AbstractActivationFunction, activationFunctionVal, activationFunctionDer, ReLU, TanH, LeakyReLU, Identity

# ActivationFunction interface
"Activation function type"
abstract AbstractActivationFunction

"Function that calculates the value of specific activation function for all elements of given array"
function activationFunctionVal
end

"Function that calculates derivative of specific activation function for all elements of given array"
function activationFunctionDer
end

# ReLU activation function
type ReLU <: AbstractActivationFunction
end

function activationFunctionVal{VarType <: AbstractFloat}(x::Array{VarType}, ::Type{Val{ReLU}}, additionalParams...)
    eps = length(additionalParams) > 0 ? VarType(additionalParams[1][1]) : 0;
    return max.(eps, x);
end

function activationFunctionDer{VarType <: AbstractFloat}(x::Array{VarType}, ::Type{Val{ReLU}}, additionalParams...)
    eps = length(additionalParams) > 0 ? VarType(additionalParams[1][1]) : 0;
    return (y-> y > eps ? VarType(1) : VarType(0)).(Array(x));
end

# Leaky ReLU activation function
type LeakyReLU{Param} <: AbstractActivationFunction
end

function activationFunctionVal{VarType <: AbstractFloat}(x::Array{VarType}, ::Type{Val{LeakyReLU}}, additionalParams...)
    eps = length(additionalParams) > 0 ? VarType(additionalParams[1][1]) : 0;
    gamma = length(additionalParams) > 1 ? VarType(additionalParams[1][1]) : 0;
    return (y -> y > eps ? y : gama*y).(x);
end

function activationFunctionDer{VarType <: AbstractFloat}(x::Array{VarType}, ::Type{Val{LeakyReLU}}, additionalParams...)
    eps = length(additionalParams) > 0 ? VarType(additionalParams[1][1]) : 0;
    gamma = length(additionalParams) > 1 ? VarType(additionalParams[1][1]) : 0;
    return (y -> y > eps ? VarType(1) : gama).(x);
end

# Tanh activation function
type TanH <: AbstractActivationFunction
end

function activationFunctionVal{VarType <: AbstractFloat}(x::Array{VarType}, ::Type{Val{TanH}}, additionalParams...)
    return tanh.(x);
end

function activationFunctionDer{VarType <: AbstractFloat}(x::Array{VarType}, ::Type{Val{TanH}}, additionalParams...)
    return 1 - tanh.(x).^2; #this is elementwise -, it's OK
end

# Identity activation function
type Identity <: AbstractActivationFunction
end

function activationFunctionVal{VarType <: AbstractFloat}(x::Array{VarType}, ::Type{Val{Identity}}, additionalParams...)
    return x;
end

function activationFunctionDer{VarType <: AbstractFloat}(x::Array{VarType}, ::Type{Val{Identity}}, additionalParams...)
    return ones(VarType, size(x));
end

end # module ActivationFunctions
