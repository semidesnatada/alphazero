package demos

func ValueDemo() {
	// Two Inputs
// 	x1 := neuralnets.NewValue(2.0, "x1")
// 	x2 := neuralnets.NewValue(0.0, "x2")

// 	// Two weights
// 	w1 := neuralnets.NewValue(-3.0, "w1")
// 	w2 := neuralnets.NewValue(1.0, "w2")
	
// 	// An offset / bias to add
// 	b := neuralnets.NewValue(6.8813735870195432, "b")
	
// 	// Products of inputs and weights
// 	x1w1 := x1.Times(w1, "x1w1")
// 	x2w2 := x2.Times(w2, "x2w2")

// 	// Sum of products
// 	x1w1x2w2 := x1w1.add(x2w2, "x1w1x2w2")

// 	// Adding the bias
// 	n := x1w1x2w2.add(b, "n")
	
// 	// Normalising to between -1 and 1
// 	output := n.tanh("output")

// 	// Printing all of the nodes in the tree, then calculating the gradients for all nodes.
// 	output.printAll()
// 	output.backwardsPropagate()
// 	output.printAll()

// 	// Creating a neuron which takes three inputs (and therefore has four parameters).
// 	NN := neuralnets.NewNeuron(3, "first-neuron")
// 	NN.print()

// 	// Some random inputs to be fed into the neuron.
// 	inputs := []*neuralnets.Value{
// 		neuralnets.NewValue(2.0, "x1"),
// 		neuralnets.NewValue(3.0, "x2"),
// 		neuralnets.NewValue(-1.0, "x3"),
// 	}

// 	// Activation is the value of the neuron once the inputs have been processed through it.
// 	activation := NN.Call(inputs)
// 	activation.print()
// 	activation.printAll()

// 	// Creating a fully connected MLP with 41 parameters, then pushing the inputs the full way through.
// 	mm := neuralnets.NewMLP([]int{3, 4, 4, 1})
// 	outs := mm.Call(inputs)
// 	outs[0].Print()
// 	fmt.Println(outs)
}