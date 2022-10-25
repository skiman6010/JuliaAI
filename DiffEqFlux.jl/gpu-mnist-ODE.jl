using DiffEqFlux, DifferentialEquations, NNlib, MLDataUtils, Printf
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets
using CUDA
CUDA.allowscalar(false)

function loadmnist(batchsize = bs)
	# Use MLDataUtils LabelEnc for natural onehot conversion
  	onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
	# Load MNIST
    mnist = MNIST(split = :train)
    imgs, labels_raw = mnist.features, mnist.targets
	# Process images into (H,W,C,BS) batches
	x_train = Float32.(reshape(imgs,size(imgs,1),size(imgs,2),1,size(imgs,3))) |> gpu
	x_train = batchview(x_train,batchsize)
	# Onehot and batch the labels
	y_train = onehot(labels_raw) |> gpu
	y_train = batchview(y_train,batchsize)
	return x_train, y_train
end

# Main
const bs = 128
x_train, y_train = loadmnist(bs)

down = Flux.Chain(Flux.flatten, Flux.Dense(784, 20, tanh)) |> gpu

nn = Flux.Chain(Flux.Dense(20, 10, tanh),
           Flux.Dense(10, 10, tanh),
           Flux.Dense(10, 20, tanh)) |> gpu


nn_ode = NeuralODE(nn, (0.f0, 1.f0), Tsit5(),
                   save_everystep = false,
                   reltol = 1e-3, abstol = 1e-3,
                   save_start = false) |> gpu

fc  = Flux.Chain(Flux.Dense(20, 10)) |> gpu

function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
end

# Build our overall model topology
model = Flux.Chain(down,
              nn_ode,
              DiffEqArray_to_Array,
              fc) |> gpu;

# To understand the intermediate NN-ODE layer, we can examine it's dimensionality
img, lab = train_dataloader.data[1][:, :, :, 1:1], train_dataloader.data[2][:, 1:1]

x_d = down(img)

# We can see that we can compute the forward pass through the NN topology
# featuring an NNODE layer.
x_m = model(img)

classify(x) = argmax.(eachcol(x))

function accuracy(model, data; n_batches = 100)
    total_correct = 0
    total = 0
    for (i, (x, y)) in enumerate(collect(data))
        # Only evaluate accuracy for n_batches
        i > n_batches && break
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# burn in accuracy
accuracy(model, train_dataloader)

loss(x, y) = logitcrossentropy(model(x), y)

# burn in loss
loss(img, lab)

opt = ADAM(0.05)
iter = 0

callback() = begin
    global iter += 1
    # Monitor that the weights do infact update
    # Every 10 training iterations show accuracy
    if iter % 10 == 1
        train_accuracy = accuracy(model, train_dataloader) * 100
        test_accuracy = accuracy(model, test_dataloader;
                                 n_batches = length(test_dataloader)) * 100
        @printf("Iter: %3d || Train Accuracy: %2.3f || Test Accuracy: %2.3f\n",
                iter, train_accuracy, test_accuracy)
    end
end

# Train the NN-ODE and monitor the loss and weights.
Flux.train!(loss, Flux.params(down, nn_ode.p, fc), train_dataloader, opt, cb = callback)

# explanation: https://docs.sciml.ai/DiffEqFlux/stable/examples/mnist_neural_ode/