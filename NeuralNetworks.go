package main

import (
	"fmt"
	"math/rand"
)


type Neuron struct {
	weights []*Value
	bias *Value
	label string
}

func newNeuron(size int, l string) *Neuron {

	newWeights := []*Value{}
	for i := 0; i < size; i++ {
		newWeights = append(newWeights,
			newValue(rand.Float64()*2 - 1, fmt.Sprintf("%s-w%d", l, i+1)),
		)
	}

	return &Neuron{
		weights: newWeights,
		bias: newValue(rand.Float64()*2 - 1, fmt.Sprintf("%s-bias", l)),
		label: l,
	}

}

func (n Neuron) Call(input []*Value) *Value {

	proc := n.bias
	var new_inter *Value

	for i, w := range n.weights {
		new_inter = input[i].times(w, "")
		proc = proc.add(new_inter, fmt.Sprintf("intermediate-sum-weight-%d",i+1))
	}

	return proc.tanh(fmt.Sprintf("%s-activation",n.label))

}

func (n Neuron) print() {

	fmt.Println()
	fmt.Println(n.label)
	for _, w := range n.weights {
		fmt.Printf("| %s : %f |\n", w.label, w.data)
	}
	fmt.Printf("%s : %f\n", n.bias.label, n.bias.data)

}

func (n Neuron) flatten() []*Value {
	return append(n.weights, n.bias)
}

type Layer struct {
	neurons []*Neuron
	label string
}

func newLayer(nInputs, nOutputs int, label string) *Layer {

	ns := []*Neuron{}

	for i := 0; i < nOutputs; i++ {
		ns = append(ns, newNeuron(
			nInputs, fmt.Sprintf("layer-%s-neuron-%d", label, i+1),
		))
	}

	return &Layer{
		neurons: ns,
		label: fmt.Sprintf("layer-%s", label),
	}

}

func (l *Layer) Call(inputs []*Value) []*Value {

	outs := []*Value{}

	for _, n := range l.neurons {
		outs = append(outs, n.Call(inputs))
	}

	return outs

}

func (l Layer) flatten() []*Value {
	out := []*Value{}
	for _, n := range l.neurons {
		out = append(out, n.flatten()...)
	}
	return out
}

type MLP struct {
	layers []*Layer
}

func newMLP(structure []int) *MLP {

	ls := []*Layer{}

	for i := 0; i < len(structure)-1; i++ {
		ls = append(ls, 
			newLayer(structure[i], structure[i+1], fmt.Sprintf("%d",i+1)),
		)
	}

	return &MLP{
		layers: ls,
	}

}

func (m *MLP) Call(inputs []*Value) []*Value {

	outs := inputs

	for i := 0; i < len(m.layers); i++ {
		outs = m.layers[i].Call(outs)
	}

	return outs

}

func (m *MLP) generateLoss(xs [][]*Value, ys []*Value) *Value {
	// requires an MLP with a one final outputs node
	// loss function is TSE

	var out *Value

	for i, xset := range xs {
		yAct := m.Call(xset)[0]
		k := ys[i].sub(yAct, "diff")
		j := k.pow(2, "powd")
		if out == nil {
			j.label = "new-total-loss-1"
			out = j
		} else {
			out = out.add(j, fmt.Sprintf("new-total-loss-%d", i+1))
		}
	}
	out.label = "Final-total-loss"

	return out

}

func (m MLP) flatten() []*Value {
	// completely flattens the MLP, giving a 
	// slice containing all weights and biases

	out := []*Value{}
	for _, l := range m.layers {
		out = append(out, l.flatten()...)
	}
	return out
}

func (m MLP) ClearGrads() {
	// gradient descent

	params := m.flatten()

	for _, p := range params {
		p.grad = 0
	}

}

func (m MLP) GD(learningRate float64) {
	// gradient descent

	params := m.flatten()

	for _, p := range params {
		p.data -= learningRate * p.grad
	}

}


func (m MLP) SGD(learningRate float64, numSteps int, xs [][]*Value, ys []*Value) {

	fmt.Println()

	for i := 0; i <= numSteps; i++ {

		loss := m.generateLoss(xs, ys)
		loss.backwardsPropagate()
		m.GD(learningRate)

		if i % 100 == 0 {
			fmt.Printf("After descent %d, loss is %f\n", i, loss.data)
		}
		m.ClearGrads()

	}
	fmt.Printf("After finishing, loss is %f\n", m.generateLoss(xs, ys).data)

}

func (m MLP) printLoss(xs [][]*Value, ys []*Value) {

	for i, xset := range xs {
		fmt.Printf("%d | Modelled: %s | Expected: %s\n", i+1, m.Call(xset)[0].printf(), ys[i].printf())
	}
	loss := m.generateLoss(xs, ys)
	fmt.Printf("Total Loss: %s \n",loss.printf())

}