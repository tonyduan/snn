### Self-Normalizing Neural Networks

---

Implementation of SNNs [1] in PyTorch. 

SNNs keep the neuron activations in the network near zero mean and unit variance, by employing the following tools.

1. SELU activation
<p align="center"><img alt="$$&#10;\mathrm{SELU}(x) = \begin{cases} \lambda x &amp; \mbox{if } x &gt; 0 \\ \lambda\alpha e^x -\lambda \alpha &amp; \mbox{if }x \leq 0\end{cases}&#10;$$" src="svgs/214d782c168d980bd6ae9120d95f536c.svg" align="middle" width="247.65407864999997pt" height="49.315569599999996pt"/></p>

2. Initialization of weights
<p align="center"><img alt="$$&#10;w_{i}^{(\ell)} \sim \mathcal{N}(0, \frac{1}{d^\ell}),\quad\text{where } d^\ell \text{ is the layer dimensionality}&#10;$$" src="svgs/60c436bec86a6fad3492148d239481f2.svg" align="middle" width="394.4523594pt" height="32.990165999999995pt"/></p>

3. Alpha-dropout (though dropout is rarely necessary in my experience)

4. Scale input features to zero-mean, unit variance.

Constants are chosen appropriately to be:
<p align="center"><img alt="$$&#10;\alpha = 1.6733 \quad \lambda = 1.0507.&#10;$$" src="svgs/fbababb0b12617bfa0d29d7e9fb7256f.svg" align="middle" width="176.32998899999998pt" height="11.4155283pt"/></p>

Note that I found Adamax to have best empirical performance when optimizing.

#### Usage

See the `examples/` folder for examples. Below we show a regression task with an 8-layer SNN for the concrete compression dataset [2]. The left pane shows the regression QQ-plot and the right pane shows the (roughly standard normally distributed) activations of the layers in the network. 

![ex_model](examples/ex_reg.png "Example model output")

#### References

[1] G. Klambauer, T. Unterthiner, A. Mayr, & S. Hochreiter, Self-Normalizing Neural Networks. In I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett,eds., Advances in Neural Information Processing Systems 30 (Curran Associates, Inc., 2017), pp. 971–980.

[2] I. Yeh, Modeling of strength of high performance concrete using artificial neural networks. In Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998). 

#### License

This code is available under the MIT License.
