__precompile__()

module Helpers

export display, loadClasses_Cifar10, getClass_Cifar10, loadBatch_Cifar10, formatBatch_Cifar10, padElement, PadType, Full, Bottom, Right, BottomRight, test

using GZip
using Images
import NeuralNetworks

const Cifar10BatchLength = 10000;
const Cifar10ElementBytesNum = 3072;
const Cifar10BatchBytesNum = Cifar10BatchLength*(Cifar10ElementBytesNum+1); # bytes for image and a byte for class
@enum PadType Full = 1 Bottom = 2 Right = 3 BottomRight = 4;

function display(element, classes)
    println(getClass_Cifar10(classes, element[2]));
    Images.display(Images.colorim(element[1]));
end

function loadClasses_Cifar10(fileName::String)
    file = open(fileName);
    classes = [];

    for line in eachline(file)
        push!(classes, line);
    end

    close(file);

    return classes;
end

function getClass_Cifar10(classes, classNumber)
    return classes[classNumber]; # classes in CIFAR-10 have numeric labels in the range 0-9
end

function loadBatch_Cifar10(fileName::String)
    file = GZip.open(fileName, "r");
    data = reinterpret(UInt8, read(file));
    close(file);

    return data;
end

function formatBatch_Cifar10(data::Array{UInt8, 1}, padSize::Int, padType::PadType = Full)
    if length(data) != Cifar10BatchBytesNum
        throw("Wrong data length");
    end

    formatedBatch = [];

    for i in 1:(Cifar10ElementBytesNum+1):Cifar10BatchBytesNum
        push!(formatedBatch, formatElement_Cifar10(data, i, padSize, padType));
    end

    return formatedBatch;
end

function formatElement_Cifar10(data::Array{UInt8, 1}, position::Int, padSize::Int, padType::PadType = Full, elementLength::Int = Cifar10ElementBytesNum)
    element::Array{Float32} = reshape(data[position + 1:position + elementLength], 32, 32, 3);

    # transpose all channels because data is stored in row major order but Julia works in column major way
    # this is not necessary because all images would be transposed, but it is nicer :D
    element[:, :, 1] = element[:, :, 1]';
    element[:, :, 2] = element[:, :, 2]';
    element[:, :, 3] = element[:, :, 3]';

    return (padElement(element, padSize, padType)./typemax(UInt8), data[position] + 1); # data[position] represents class of the element, + 1 is beacuse it is from [0,9]
end

# Adds additional zeros around element
function padElement{VarType <: Number}(element::AbstractArray{VarType}, padSize::Int, padType::PadType = Full)
    if padSize <= 0
        return element;
    end

    height, width, depth = size(element);

    topPad, rightPad, bottomPad, leftPad = calculatePads(padSize, padType);

    paddedElement = zeros(VarType, height + topPad + bottomPad, width + leftPad + rightPad, depth);
    for i in 1:depth
        paddedElement[topPad+1:end-bottomPad, leftPad+1:end-rightPad, i] = element[:, :, i];
    end

    return paddedElement;
end

function calculatePads(padSize::Int, padType::PadType)
    if padSize <= 0
        return 0, 0, 0, 0;
    end

    if padType == Full
        return padSize, padSize, padSize, padSize;
    end

    if padType == Right
        return 0, padSize, 0, 0;
    end

    if padType == Bottom
        return 0, 0, padSize, 0;
    end

    if padType == BottomRight
        return 0, padSize, padSize, 0;
    end

    throw(ArgumentError("Unknown pad type."))
end

function test(net, data, dataName)
println("Testing $(net.name) on $(length(data)) examples from $(dataName).")
br = 0;
brs = zeros(10);

for i in 1:length(data)
    res = NeuralNetworks.predict!(data[i][1], net);

    if res != data[i][2]
        brs[data[i][2]] += 1;
    else
        br = br + 1;
    end
end

println("Broj tačnih: $(br). Broj pogrešnih: $(brs).")

end

end # module Helpers
