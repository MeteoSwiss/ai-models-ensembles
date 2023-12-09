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

$$r*e^{i\theta} = r*(\cos(\theta) + i\sin(\theta))$$

where 'r' is the magnitude (or modulus) of the complex number, and 'θ' is the phase angle.

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

In the SpectralFilterLayer of the FourierNeuralOperatorBlock, the complex weights in the SpectralAttentionS2 layer represent the parameters that the model learns during training to perform convolutions in the Fourier (frequency) domain.

The real part of the complex weight can be interpreted as the magnitude of the corresponding frequency component. In other words, it determines the "strength" or "intensity" of a particular frequency in the Fourier space.
The imaginary part of the complex weight can be interpreted as the phase shift of the corresponding frequency component. This means it determines the "shift" or "offset" of a particular frequency in the Fourier space.

The weights are complex because they operate in the Fourier space, where signals are represented as a combination of complex exponential functions (which can be thought of as rotating vectors). The real and imaginary parts of these weights can be interpreted as the magnitude and phase shift of the corresponding frequency component, respectively.

Each Parameter object in the ParameterList contains a tensor of weights with shape [256x512x2], [512x512x2], and [512x512x2] respectively. The last dimension of size 2 represents the real and imaginary parts of the complex weights.

The exact interpretation of these weights depends on the specific operations performed by the SpectralAttentionS2 layer. However, in general, these weights are used to learn global convolutions in the Fourier space, and they don't directly represent the amplitude or phase of specific frequencies. Instead, they learn to capture different frequency components of the input data, allowing the model to model complex spectral patterns.

When a wave represented by r*e^(iΘ) is multiplied by a complex weight, the effect is a change in the amplitude and phase of the wave.
Let's denote the complex weight as a + bi, where a and b are the real and imaginary parts of the weight, respectively. The multiplication of the wave by the weight in the complex plane is as follows:

(r*e^(iΘ)) * (a + bi) = r*a*e^(iΘ) + r*b*i*e^(iΘ)

This can be rewritten using Euler's formula as:

= r*a*(cos(Θ) + i*sin(Θ)) + r*b*i*(cos(Θ) + i*sin(Θ))
= (r*a*cos(Θ) - r*b*sin(Θ)) + i*(r*a*sin(Θ) + r*b*cos(Θ))

The result is a new complex number, where (r*a*cos(Θ) - r*b*sin(Θ)) is the real part and (r*a*sin(Θ) + r*b*cos(Θ)) is the imaginary part.
The real part can be interpreted as the new amplitude (or magnitude) of the wave, and the imaginary part can be interpreted as the new phase of the wave.
In other words, the multiplication by the complex weight results in a wave with a modified amplitude and phase. This is the fundamental operation that allows the Fourier Neural Operator to learn and model complex spectral patterns in the data.

Let's consider a wave represented in Euler's form as A * e^(iωt), where A is the amplitude, ω is the angular frequency, and t is time. 
Now, let's multiply this wave by a complex number in polar form, B * e^(iφ), where B is the magnitude and φ is the phase angle.
The result is (A * B) * e^(i(ωt + φ)).
Here's what happens to the wave:

1. **Amplitude**: The amplitude of the wave is multiplied by the magnitude of the complex number. If A was the original amplitude, the new amplitude is now A * B.
2. **Frequency**: The frequency of the wave remains unchanged. This is because the multiplication does not affect the ωt term, which determines the frequency of the wave.
3. **Phase**: The phase of the wave is shifted by the argument of the complex number. If the original phase was ωt, the new phase is now ωt + φ. This represents a phase shift of φ.

So, multiplying a wave in Euler form by a complex number in polar form changes the amplitude and phase of the wave, but leaves the frequency unchanged.

Let's consider a wave represented in the standard form as A(cos(ωt) + i*sin(ωt)), where A is the amplitude, ω is the angular frequency, and t is time. This is equivalent to the Euler's form A * e^(iωt) due to Euler's formula.
Now, let's multiply this wave by a complex number in standard form, which is B(cos(φ) + i*sin(φ)). This is equivalent to the polar form B * e^(iφ).

The multiplication of the two complex numbers in standard form is:

(A * cos(ωt) + i * A * sin(ωt)) * (B * cos(φ) + i * B * sin(φ))

When we multiply these out, we use the distributive property and the fact that i^2 = -1:

= A * B * (cos(ωt) * cos(φ) - sin(ωt) * sin(φ)) + i * A * B * (cos(ωt) * sin(φ) + sin(ωt) * cos(φ))

Using the angle addition formulas for sine and cosine, we can simplify this to:

= A * B * cos(ωt + φ) + i * A * B * sin(ωt + φ)

This is the standard form of the resulting complex wave, which has an amplitude of A * B and a phase shift of φ. The frequency ω remains unchanged.