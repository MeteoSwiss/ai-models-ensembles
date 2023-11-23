# Ensemble Weather Forecasts with ECWFM's AI Models

This is a comprehensive wrapper around the ECWFM's AI models. The primary goal is to facilitate the execution of these AI models within a pipeline, thereby generating ensemble weather forecasts.

## Core Functionality

The heart of this repository is the `runscript.sh`, a bash script that orchestrates the entire pipeline. It requires three arguments:

1. **Start Datetime**: The start datetime of the forecast in the format `YYYYMMDDHHMM`
2. **Model**: The model to be used (options: FourCastNet, GraphCast, Pangu)
3. **Ensemble Members**: Number of ensemble members to generate

Upon receiving these arguments, the script initiates the pipeline for the given model and start datetime.

## Some Theory

### Complex Numbers in Pytorch

The Fourier Neural Operator operates in the Fourier space and utilizes complex numbers. However, in PyTorch, complex numbers are typically represented as a pair of real numbers, where the first number is the real part and the second number is the imaginary part. This is why you see the weights as real numbers.

In the world of sine and cosine functions, complex numbers can be represented using Euler's formula, which states:

$$e^{ix} = \cos(x) + i\sin(x)$$

Here, 'x' is a real number, 'i' is the imaginary unit, and 'e' is the base of the natural logarithm. The real part of the complex number is cos(x), and the imaginary part is sin(x).

This formula shows the deep relationship between exponential functions, complex numbers, and trigonometric functions. It's the basis for expressing any complex number in polar form as:

$$r(\cos(\theta) + i\sin(\theta))$$

where 'r' is the magnitude (or modulus) of the complex number, and 'θ'

### SpectralAttentionS2 Layer

In the `SpectralAttentionS2` layer, the weights are stored in the `w` attribute as a `ParameterList` containing three `Parameter` objects. Each `Parameter` object contains a tensor of weights with shape `[256x512x2]`, `[512x512x2]`, and `[512x512x2]` respectively. The last dimension of size 2 represents the real and imaginary parts of the complex weights.

```
(0): FourierNeuralOperatorBlock(
  (norm0): InstanceNorm2d(256, eps=1e-06, momentum=0.1, affine=True, track_running_stats=False)
  (filter_layer): SpectralFilterLayer(
    (filter): SpectralAttentionS2(
      (w): ParameterList(
          (0): Parameter containing: [torch.float32 of size 256x512x2]
          (1): Parameter containing: [torch.float32 of size 512x512x2]
          (2): Parameter containing: [torch.float32 of size 512x512x2]
      )
      (drop): Identity()
      (activation): ComplexReLU(
        (act): LeakyReLU(negative_slope=0.0)
      )
    )
  )
)
```

### Fourier Space and Weights

In the context of the `SpectralAttentionS2` layer in the `FourierNeuralOperatorBlock`, it's not straightforward to map the weights directly to specific frequencies. This is because the weights in this layer are used to learn global convolutions in the Fourier space, and they don't directly represent the amplitude or phase of specific frequencies.

In the context of Fourier space, each layer of weights can be thought of as learning to capture different frequency components of the input data. The combination of these layers allows the model to capture a wide range of frequencies and to model complex spectral patterns.

It's important to note that the exact number of layers and their sizes are hyperparameters of the model and can be adjusted based on the specific task and data. The choice of three layers in this case is likely based on empirical performance on the training data.

