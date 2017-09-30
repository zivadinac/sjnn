__precompile__()

module NeuralNetworks

export NeuralNetwork, predict!, train!, saveParams, loadParams

importall Layers
importall CostFunctions
importall Solvers

type NeuralNetwork{VarType <: AbstractFloat, CostFunc <: AbstractCostFunction}
    name::String
    costFuncAddlParams # used for storing custom additional params for cost functions
    layers::Array{AbstractLayer}
    output::Array{VarType}
    predictedClass::Int

    function NeuralNetwork(networkName::String, additionalParams...)
        new(networkName, additionalParams);
    end
end

function forwardPass!{VarType <: AbstractFloat, CostFunc <: AbstractCostFunction}(
        input::Array{VarType}
        , net::NeuralNetwork{VarType, CostFunc}
    )
    output = input;
    for layer in net.layers
        output = forwardPass!(layer, output);
    end

    net.output = output;
    net.predictedClass = indmax(net.output);
    return net.predictedClass;
end

function predict!{VarType <: AbstractFloat, CostFunc <: AbstractCostFunction}(
        x::Array{VarType}
        , net::NeuralNetwork{VarType, CostFunc}
    )
    return forwardPass!(Array(x), net);
end

function train!{VarType <: AbstractFloat, CostFunc <: AbstractCostFunction}(
        solver::AbstractSolver{VarType}
        , net::NeuralNetwork{VarType, CostFunc}
        , trainingExamples::Array{Tuple{Array{VarType}, Number}}
        , folderPath::String = ""
    )
    regCoef = 1.0f0;
    correctNum::Int = 0;
    prevCorrectNum = correctNum;
    samplesNum = length(trainingExamples);
    p = 1;

    costFuncValByPass::Vector = [];
    avgCostFuncValByEpoch::Vector = [];
    avgLoss::VarType = 0.0;

    while p <= solver._totalEpochs && correctNum < samplesNum

        learningRateIncreased = false;
        if p > 1 && correctNum == prevCorrectNum
            solver._learningRate *= (1/solver._dropFactor);
            learningRateIncreased = true;
            println("Povećava u epohi $(p)");
        elseif mod(p, solver._dropInterval) == 0 || learningRateIncreased
                solver._learningRate *= solver._dropFactor;
                println("Smanjuje u epohi $(p)");
        end

        prevCorrectNum = correctNum;
        correctNum = 0;
        prevAvgLoss = avgLoss;
        avgLoss = 0.0;
        predictions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        avgConsPoint = max(1, length(avgCostFuncValByEpoch) - 4); # consider only up to last five epochs TODO export as parameter
#        avgAvgLoss = length(avgCostFuncValByEpoch) > 0 ? mean(avgCostFuncValByEpoch[avgConsPoint:end]) : 0;
        avgAvgLoss = length(avgCostFuncValByEpoch) > 0 ? mean(avgCostFuncValByEpoch) : 0;

        for te in trainingExamples
            x = Array(te[1]);
            y = te[2];

            pred = forwardPass!(x, net);
            predictions[pred] += 1;
            loss = costFunctionVal(y, net.output, Val{Hinge}, net.costFuncAddlParams);
            reg = regCoef * sum(l->norm(l.weights), net.layers); # l2 regularization, TODO expose regCoef and choosing as parameter
            loss += reg;
            avgLoss = avgLoss + loss;
            push!(costFuncValByPass, loss);

            if net.predictedClass == y
                correctNum+=1;
            end

            if loss > avgAvgLoss / 2.5f0 # TODO export 2.0 as parameter
                outputGrad = costFunctionDer(y, net.output, Val{Hinge}, net.costFuncAddlParams);
                updatePass!(solver, p, x, y, outputGrad, net.layers);
            end
            #println(string("Correct class: ", y, " predicted: ", pred, " net output: ", net.output, " loss: ", loss, " loss grad: ", outputGrad));
        end

        avgLoss = avgLoss / length(trainingExamples);
        push!(avgCostFuncValByEpoch, avgLoss);

        if correctNum == samplesNum
            break;
        end

        saveParams(net, string(folderPath, "/", net.name, "/epoch_", p));
        saveParams(solver, string(folderPath, "/", net.name, "/epoch_", p));
        println("Finished epoch $p at $(Dates.format(now(), "HH:MM:SS")), correct num: $correctNum");
        println("Predictions: $(predictions)");
        p+=1;
    end
    println("Finished at epoch $(p - 1), correctNum: $correctNum");

    return costFuncValByPass, avgCostFuncValByEpoch;
end

function saveParams(net::NeuralNetwork, location::String)
    mkpath(location);
    layerNum = 1;

    for layer in net.layers
        saveParams(layer, string(location, "/", net.name, "_layer_", layerNum));
        layerNum += 1;
    end
end

function loadParams(net::NeuralNetwork, networkFolder::String)
    files = sort(readdir(networkFolder));

    for i in 1:length(net.layers)
        layer = net.layers[i];
        loadParams(layer, string(networkFolder, "/", "layer_", i));
    end
end

end # module NeuralNetworks
