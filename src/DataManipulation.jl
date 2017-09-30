__precompile__()

module DataManipulation

export normalize

function normalize{VarType <: AbstractFloat}(data::Array, ::Type{Val{VarType}})
    vectors = [];

    for datum in data
        push!(vectors, datum[1]);
    end

    mean = Base.mean(vectors);

    normalizedData = Array{Tuple{Array{VarType}, Number}, 1}();

    for datum in data
        zc = datum[1]-mean;
        normalized = convert(Array{VarType}, zc); # zc./std(zc)
        push!(normalizedData, (Array(normalized), datum[2]));
    end

    return normalizedData;
end

end # module DataManipulation
