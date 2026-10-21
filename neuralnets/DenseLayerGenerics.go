package neuralnets

import (
	"fmt"
	"math/rand/v2"
)


type Neuron struct {
	weights []*Value
	bias *Value
	label string
}

type DenseLayer struct {
	neurons []*Neuron
	label string
	activationType string
	supplementaryData []float64
}

type DenseNetwork struct {
	layers []*DenseLayer
}

type DenseNetworkSpecification struct {
	// note that this two slices should have same length, 
	// but first layer is input and so has no activation type
	LayerSizes []int
	LayerActivations []string
}

func newNeuron(size int, l string) *Neuron {

	newWeights := []*Value{}
	for i := 0; i < size; i++ {
		newWeights = append(newWeights,
			NewValue(rand.Float64()*2 - 1, fmt.Sprintf("%s-w%d", l, i+1)),
		)
	}

	return &Neuron{
		weights: newWeights,
		bias: NewValue(rand.Float64()*2 - 1, fmt.Sprintf("%s-bias", l)),
		label: l,
	}

}

func newDenseLayer(nInputs, nOutputs int, label, activationType string) *DenseLayer {

	ns := []*Neuron{}

	for i := 0; i < nOutputs; i++ {
		ns = append(ns, newNeuron(
			nInputs, fmt.Sprintf("layer-%s-neuron-%d", label, i+1),
		))
	}

	return &DenseLayer{
		neurons: ns,
		label: fmt.Sprintf("layer-%s", label),
		activationType: activationType,
	}

}


func (n Neuron) print() {

	fmt.Println()
	fmt.Println(n.label)
	for _, w := range n.weights {
		fmt.Printf("| %s : %f |\n", w.label, w.data)
	}
	fmt.Printf("%s : %f\n", n.bias.label, n.bias.data)

}


func (n Neuron) Call(input []*Value, activationType string) *Value {

	proc := n.bias
	var new_inter *Value

	for i, w := range n.weights {
		new_inter = input[i].times(w, "")
		proc = proc.add(new_inter, fmt.Sprintf("intermediate-sum-weight-%d",i+1))
	}

	switch activationType {
		case "tanh":
			return proc.tanh(fmt.Sprintf("%s-tanh-activation",n.label))
		case "relu":
			return proc.relu(fmt.Sprintf("%s-relu-activation",n.label))
		default:
			proc.label = fmt.Sprintf("%s-linear-activation",n.label)
			return proc
	}
	
}

func (l *DenseLayer) Call(inputs []*Value) []*Value {

	outs := []*Value{}

	if l.activationType != "az-output" {
		for _, n := range l.neurons {
			outs = append(outs, n.Call(inputs, l.activationType))
		}
	} else {
		// for an alphazero output layer, we allow the first n-1 nodes to be 
		// simple linear nodes (which we later softmax into a policy distribution)
		for i := 0; i < len(l.neurons) - 1; i++ {
			outs = append(outs, l.neurons[i].Call(inputs, "linear"))
		}
		// and the final node is a tanh value node
		outs = append(outs, l.neurons[len(l.neurons)-1].Call(inputs, "tanh"))
	}

	return outs

}

func (m *DenseNetwork) CallTraining(inputs []*Value) []*Value {

	outs := inputs

	for i := 0; i < len(m.layers); i++ {
		outs = m.layers[i].Call(outs)
	}

	return outs

}

func (n Neuron) flatten() []*Value {
	return append(n.weights, n.bias)
}

func (l DenseLayer) flatten() []*Value {
	out := []*Value{}
	for _, n := range l.neurons {
		out = append(out, n.flatten()...)
	}
	return out
}

func (m DenseNetwork) flatten() []*Value {
	// completely flattens the MLP, giving a 
	// slice containing all weights and biases

	out := []*Value{}
	for _, l := range m.layers {
		out = append(out, l.flatten()...)
	}
	return out
}

func (m DenseNetwork) ClearGrads() {
	// gradient descent

	params := m.flatten()

	for _, p := range params {
		p.grad = 0
	}

}

func (m DenseNetwork) gradientDescent(learningRate, weightDecay float64) {
	// gradient descent

	params := m.flatten()

	for _, p := range params {
		p.data -= learningRate * (p.grad + 2.0 * weightDecay * p.data)
	}

}