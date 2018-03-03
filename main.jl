using OhMyJulia
using StatsBase
using PyCall
using Fire

unshift!(PyVector(pyimport("sys")["path"]), @__DIR__)

function get_data()
    text = open(readstring, "D:/xi-presentation/clean/hand.txt")
    V = unique(text)
    data = [findfirst(V, c) for c in text]
    data, V
end

function train(model, data, epoch, seqlen, batchsize=256)
    loss = 0

    for e in 1:epoch
        model[:reset](batchsize)
        indices = rand(1:length(data)-seqlen+1, batchsize)
        list = [data[indices .+ i].-1 for i in 0:seqlen-1]
        loss += model[:learn](list)
    end

    loss / epoch
end

function sample_softmax(x, t=.5)
    w = ProbabilityWeights(exp.(x / t))
    sample(1:length(x), w)
end

function predict(model, V)
    model[:reset](1)
    c = findfirst(V, '\n')
    print("\n  >>>> ")
    for j in 1:1024
        p = model[:predict](c-1)[:]
        c = sample_softmax(p)
        print(V[c])
    end
    println()
end

@main function main(model; epoch::Int=50)
    data, V = get_data()
    model = pyimport(model)[:Model](length(V), "gpu")
    for i in 1:epoch
        loss = @time train(model, data, 1000, min(20+i, 64))
        println("\n=== epoch $i, loss: $loss ===")
        predict(model, V)
        try
            model[:save]("D:/xi-presentation/model/$model.model")
        end
    end
end