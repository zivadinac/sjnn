__precompile__()

module Solvers

importall Layers
importall CostFunctions
using HDF5, JLD

export AbstractSolver, SGD, Adam, updatePass!, saveParams, loadParams

abstract AbstractSolver{VarType <: AbstractFloat}

type SGD{VarType <: AbstractFloat} <: AbstractSolver{VarType}
    _learningRate::VarType
    _totalEpochs::Int
    _dropFactor::VarType
    _dropInterval::Int

    function SGD(learningRate::VarType = VarType(0.01), totalEpochs::Int = 10, dropFactor::VarType = VarType(10.0), dropInterval::Int = 10)
        new(learningRate, totalEpochs, dropFactor, dropInterval);
    end
end

type Adam{VarType <: AbstractFloat} <: AbstractSolver{VarType}
    _learningRate::VarType
    _regCoef::VarType
    _totalEpochs::Int
    _beta1::VarType
    _beta2::VarType
    _eps::VarType
    _dropFactor::VarType
    _dropInterval::Int
    _m::Array{Array{VarType,2},1}
    _v::Array{Array{VarType,2},1}

    function Adam(layers::Array{AbstractLayer}
                  , learningRate::VarType, regCoef::VarType, totalEpochs::Int
                  , dropFactor::VarType, dropInterval::Int
                  , beta1::VarType, beta2::VarType, eps::VarType)
        m = []; v = [];

        for l in layers
            push!(m, zeros(VarType, size(l.weights)));
            push!(v, zeros(VarType, size(l.weights)));
        end

        new(learningRate, regCoef, totalEpochs, beta1, beta2, eps, dropFactor, dropInterval, m, v);
    end
end

function updatePass!{VarType <: AbstractFloat}(solver::AbstractSolver, epochNum::Int, input::Array{VarType}
                    , correctY::Int, outputGrad::Array{VarType}, layers::Array{AbstractLayer})

    outputLayer = layers[end];

    if length(layers) == 1
        deltaOld, derWeightsOld, derBiasesOld = backwardPass!(outputLayer, outputGrad);
        update!(solver, layers, 1, derWeightsOld, derBiasesOld);
        return;
    end

    deltaOld, derWeightsOld, derBiasesOld = backwardPass!(outputLayer, outputGrad);

    for i in reverse(eachindex(layers)[2:end-1])
        deltaNew, derWeightsNew, derBiasesNew = backwardPass!(layers[i], layers[i+1], deltaOld);

        update!(solver, layers, i+1, derWeightsOld, derBiasesOld);

        deltaOld = deltaNew;
        derWeightsOld = derWeightsNew;
        derBiasesOld = derBiasesNew;
    end

    deltaNew, derWeightsNew, derBiasesNew = backwardPass!(layers[1], layers[2], deltaOld);
    update!(solver, layers, 2, derWeightsOld, derBiasesOld);
    update!(solver, layers, 1, derWeightsNew, derBiasesNew);
end

function update!{VarType <: AbstractFloat}(solver::Adam, layers::Array{AbstractLayer}, layerIndex::Int, derWeights::Array{VarType}, derBiases::Array{VarType})
    beta1 = solver._beta1;
    beta2 = solver._beta2;
    eps = solver._eps;
    learningRate = solver._learningRate;
    regCoef = solver._regCoef;
    iterNum = solver._totalEpochs;

    dW = derWeights - regCoef * layers[layerIndex].weights;
    solver._m[layerIndex] = (beta1*solver._m[layerIndex] + (1-beta1)*dW);
    mt = solver._m[layerIndex] / (1-beta1^iterNum);
    solver._v[layerIndex] = (beta2*solver._v[layerIndex] + (1-beta2)*dW.^2);
    vt = solver._v[layerIndex] / (1-beta2^iterNum);

    layers[layerIndex].weights += -learningRate*mt ./ (sqrt(vt) + eps);
#    layers[layerIndex].weights = (1-learningRate*regCoef)*layers[layerIndex].weights -(learningRate*mt ./ (sqrt(vt) + eps));
end

function update!{VarType <: AbstractFloat}(solver::SGD, layers::Array{AbstractLayer}, layerIndex::Int, derWeights::Array{VarType}, derBiases::Array{VarType})
    learningRate = solver._learningRate;
    layer[layerIndex].weights -= learningRate*derWeights;
end

macro solverFileName(name, solverType)
    :(string($name, "/", $solverType, "_solver.jld"))
end


function saveParams(solver::Adam, location::String)
    sName = @solverFileName(location, "Adam");

    save(sName, "type", "Adam");
    save(sName, "learningRate", solver._learningRate);
    save(sName, "totalEpochs", solver._totalEpochs);
    save(sName, "beta1", solver._beta1);
    save(sName, "beta2", solver._beta2);
    save(sName, "eps", solver._eps);
    save(sName, "dropFactor", solver._dropFactor);
    save(sName, "dropInterval", solver._dropInterval);
    save(sName, "m", solver._m);
    save(sName, "v", solver._v);
end

function saveParams(solver::SGD, location::String)
    sName = @solverFileName(location, "SGD");

    save(sName, "type", "SGD");
    save(sName, "learningRate", solver._learningRate);
    save(sName, "totalEpochs", solver._totalEpochs);
    save(sName, "dropFactor", solver._dropFactor);
    save(sName, "dropInterval", solver._dropInterval);
end


function loadParams(solver::Adam, name::String)
    sName = @solverFileName(name, "Adam");

    solver._learningRate = load(sName)["learningRate"];
    solver._totalEpochs = load(sName)["totalEpochs"]
    solver._beta1 = load(sName)["beta1"];
    solver._beta2 = load(sName)["beta2"];
    solver._eps = load(sName)["eps"];
    solver._dropFactor = load(sName)["dropFactor"];
    solver._dropInterval = load(sName)["dropInterval"];
    solver._m = load(sName)["m"];
    solver._v = load(sName)["v"];
end

function loadParams(solver::SGD, name::String)
    sName = @solverFileName(name, "SGD");

    solver._learningRate = load(sName)["learningRate"];
    solver._totalEpochs = load(sName)["totalEpochs"]
    solver._dropFactor = load(sName)["dropFactor"];
    solver._dropInterval = load(sName)["dropInterval"];
end

end # module Solvers
