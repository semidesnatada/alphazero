package main

import "fmt"

func main() {

	// fmt.Println()

	// x1 := newValue(2.0, "x1")
	// x2 := newValue(0.0, "x2")

	// w1 := newValue(-3.0, "w1")
	// w2 := newValue(1.0, "w2")
	
	// b := newValue(6.8813735870195432, "b")
	
	// x1w1 := x1.times(w1, "x1w1")
	// x2w2 := x2.times(w2, "x2w2")

	// x1w1x2w2 := x1w1.add(x2w2, "x1w1x2w2")

	// n := x1w1x2w2.add(b, "n")
	
	// output := n.tanh("output")

	// output.printAll()
	// output.backwardsPropagate()
	// output.printAll()

	// NN := newNeuron(3, "first-neuron")
	// NN.print()

	// inputs := []*Value{
	// 	newValue(2.0, "x1"),
	// 	newValue(3.0, "x2"),
	// 	newValue(-1.0, "x3"),
	// }

	// activation := NN.Call(inputs)

	// activation.print()
	// activation.printAll()


	mm := newMLP([]int{3, 4, 4, 1})

	// outs := mm.Call(inputs)
	// outs[0].print()
	// fmt.Println(outs)

	xs := [][]*Value{
		{
			newValue(2.0, "x1"),
			newValue(3.0, "x2"),
			newValue(-1.0, "x3"),
		},
		{
			newValue(3.0, "x1"),
			newValue(-1.0, "x2"),
			newValue(0.5, "x3"),
		},
		{
			newValue(0.5, "x1"),
			newValue(1.0, "x2"),
			newValue(1.0, "x3"),
		},
		{
			newValue(1.0, "x1"),
			newValue(1.0, "x2"),
			newValue(-1.0, "x3"),
		},
	}
	ys := []*Value{
		newValue(1.0, "y1"),
		newValue(-1.0, "y2"),
		newValue(-1.0, "y3"),
		newValue(1.0, "y4"),
	}

	fmt.Println("\nPre-training")
	for _, xset := range xs {
		mm.Call(xset)[0].print()
	}

	loss := mm.generateLoss(xs, ys)
	loss.print()

	mm.SGD(0.05, 500, xs, ys)

	fmt.Println("\nPost-training")
	for _, xset := range xs {
		mm.Call(xset)[0].print()
	}

	loss = mm.generateLoss(xs, ys)
	loss.print()
	fmt.Println()
}