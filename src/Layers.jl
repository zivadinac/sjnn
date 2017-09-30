__precompile__()

module Layers

using ImageProcessing
using HDF5, JLD
importall ActivationFunctions

export AbstractLayer, FullyConnectedLayer, ConvolutionalLayer, forwardPass!, backwardPass!, saveParams, loadParams

abstract AbstractLayer{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction}

"Type that represents fully connected layer of a neural network."
type FullyConnectedLayer{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction} <: AbstractLayer{VarType, ActFunc}
    weights::Array{VarType, 2}
    biases::Array{VarType, 1}
    actFuncAddlParams # used for storing custom additional params for activation function
    output::Array{VarType, 1}
    input::Array{VarType}
    activations::Array{VarType, 1}
    derivatives::Array{VarType, 1}

    function FullyConnectedLayer(InputSize::Int, OutputSize::Int, additionalParams...)
        new(rand(VarType, OutputSize, InputSize), zeros(VarType, OutputSize), additionalParams); # weights and biases initialization
    end
end

"Type that represents convolutiona connected layer of a neural network. It takes input of size NxM "
type ConvolutionalLayer{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction, FilterRadius, Stride, FiltersNum} <: AbstractLayer{VarType, ActFunc}
    weights::Array{VarType, 2} # this actually represents filters, this name is for convinience with FC layers
#    biases::Array{VarType, 2}
    actFuncAddlParams # used for storing custom additional params for activation function
    inputSize::Tuple
    output::Array
    input::Array
    activations::Array
    derivatives::Array

    function ConvolutionalLayer(InputWidth::Int, InputHeight::Int, InputDepth::Int, additionalParams...)
        new(randn(VarType, FiltersNum, (FilterRadius*2 + 1)^2 * InputDepth).*sqrt(2.5f0/(InputWidth*InputHeight*InputDepth)), additionalParams, (InputWidth, InputHeight, InputDepth)); # weights initialization
    end
end

"Function that computes output values for given input and current weights in the layer.
This function sets the values of activations before applying activation function."
function forwardPass!{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction}(
        layer::FullyConnectedLayer{VarType, ActFunc}
        , input::Array{VarType}
    )
    layer.input = input;
    layer.activations = (layer.weights*layer.input) + layer.biases;
    layer.output = activationFunctionVal(layer.activations, Val{ActFunc}, layer.actFuncAddlParams);
    return layer.output;
end

"Function that computes output values for given input and current weights in the layer.
This function sets the values of activations before applying activation function."
function forwardPass!{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction, FilterRadius, FiltersNum, Stride}(
        layer::ConvolutionalLayer{VarType, ActFunc, FilterRadius, Stride, FiltersNum}
        , input::Array{VarType, 3}
    )
#    println("forward-layer: $(layer.inputSize)")
    iH, iW, iD = layer.inputSize;
    filterDiameter::Int = FilterRadius*2 + 1;
    outH::Int = div((iH - filterDiameter), Stride) + 1;
    outW::Int = div((iW - filterDiameter), Stride) + 1;

#    println("inputSize: $(size(input)), fd: $(filterDiameter), stride: $(Stride)")
    layer.input = Array(im2col(Array(input), (filterDiameter, filterDiameter, iD), (Stride, Stride, 1))); # ArrayFire{VarType, 3} is not subtype of AbstractArray{VarType, 3}
                                                                                                            # that's why input must be converted to Array
    layer.activations = (layer.weights*layer.input);
    aux = activationFunctionVal(layer.activations, Val{ActFunc}, layer.actFuncAddlParams);

    if ndims(aux) == 1 # pretty ugly, find another way
        aux = hcat(aux);
    end

    layer.derivatives = activationFunctionDer(Array(aux), Val{ActFunc}, layer.actFuncAddlParams); # activationFunctionDer(currentLayer.output, Val{ActFunc}, currentLayer.actFuncAddlParams);
    layer.output = reshape(aux, outH, outW, FiltersNum);

    return Array(layer.output);
end

"Function that computes error delta for currentLayer, derivatives of weihts and biases with respect to cost function.
This function sets the values of derivatives of activation function to the currentLayer."
function backwardPass!{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction, ActFunc2 <: AbstractActivationFunction}(
        currentLayer::FullyConnectedLayer{VarType, ActFunc}
        , nextLayer::FullyConnectedLayer{VarType, ActFunc2}
        , nextError::Array{VarType}
    )
    currentLayer.derivatives = activationFunctionDer(currentLayer.output, Val{ActFunc}, currentLayer.actFuncAddlParams);
    delta = nextLayer.weights'*nextError.*currentLayer.derivatives;
    derWeights = delta*currentLayer.input';
    derBiases = delta;

    return delta, derWeights, derBiases;
end

"Function that computes error delta for currentLayer, derivatives of weihts and biases with respect to cost function.
This function sets the values of derivatives of activation function to the currentLayer."
function backwardPass!{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction, FilterRadius, FiltersNum, Stride, ActFunc2 <: AbstractActivationFunction, FilterRadius2, Stride2, FiltersNum2}(
        currentLayer::ConvolutionalLayer{VarType, ActFunc, FilterRadius, Stride, FiltersNum}
        , nextLayer::ConvolutionalLayer{VarType, ActFunc2, FilterRadius2, Stride2, FiltersNum2}
        , nextError::Array{VarType}
    )
#    println("backward-layer: $(currentLayer.inputSize)")
    filterDiameter::Int = FilterRadius*2 + 1;
    iH, iW, iD = nextLayer.inputSize;

    ne = reshape(nextError, size(currentLayer.activations)).*currentLayer.derivatives;;
    derWeights = reshape(ne * currentLayer.input', size(currentLayer.weights));
    delta = currentLayer.weights' * ne
#    println("deltaSize: $(size(delta)), is: $(currentLayer.inputSize), stride: $(Stride)");
    delta = col2im(delta, (filterDiameter, filterDiameter, currentLayer.inputSize[3]), currentLayer.inputSize, (Stride, Stride, 1));
    derBiases = delta;

    return Array(delta), Array(derWeights), Array(derBiases);
end

"Function that computes error delta for lastLayer, derivatives of weihts and biases with respect to cost function.
This function sets the values of derivatives of activation function to the lastLayer."
function backwardPass!{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction}(
        lastLayer::FullyConnectedLayer{VarType, ActFunc}
        , outputGrad::Array{VarType}
    )
    lastLayer.derivatives = activationFunctionDer(lastLayer.output, Val{ActFunc}, lastLayer.actFuncAddlParams);
    delta = outputGrad.*lastLayer.derivatives;
    derWeights = delta*lastLayer.input';
    derBiases = delta;

    return delta, derWeights, derBiases;
end

"Function that computes error delta for lastLayer, derivatives of weihts and biases with respect to cost function.
This function sets the values of derivatives of activation function to the lastLayer."
function backwardPass!{VarType <: AbstractFloat, ActFunc <: AbstractActivationFunction, FilterRadius, FiltersNum, Stride}(
        lastLayer::ConvolutionalLayer{VarType, ActFunc, FilterRadius, Stride, FiltersNum}
        , outputGrad::Array{VarType}
    )
#    println("backward-layer: $(lastLayer.inputSize)")
#    println("og: $(outputGrad)")
    filterDiameter::Int = FilterRadius*2 + 1;
    iH, iW, iD = lastLayer.inputSize;
    ne = reshape(outputGrad, size(lastLayer.activations)).*lastLayer.derivatives;
    derWeights = reshape(ne * lastLayer.input', size(lastLayer.weights));
    delta = hcat(lastLayer.weights' * ne);
#    println("deltaSize: $(size(delta)), is: $(lastLayer.inputSize), stride: $(Stride)");
    delta = col2im(delta, (filterDiameter, filterDiameter, lastLayer.inputSize[3]), lastLayer.inputSize, (Stride, Stride, 1));
    derBiases = delta;

    return Array(delta), Array(derWeights), Array(derBiases);
end

macro weightsFileName(name)
    :(string($name, "_weights.jld"))
end

macro biasesFileName(name)
    :(string($name, "_biases.jld"))
end

"Function that saves weights and biases of a layer. It appends _weights and _biases to provided filename"
function saveParams(layer::AbstractLayer, name::String)
    wName = @weightsFileName name;
    bName = @biasesFileName name;

    save(wName, "data", layer.weights);
    #save(bName, "data", layer.biases);
    return;
end

"Function that loads weights and biases of a layer from file. It appends _weights and _biases to provided filename"
function loadParams(layer::AbstractLayer, name::String)
    wName = @weightsFileName name;
    bName = @biasesFileName name;

    layer.weights = load(wName)["data"];
    if isfile(bName)
        layer.biases = load(bName)["data"];
    end
    return layer;
end

end # module Layers
