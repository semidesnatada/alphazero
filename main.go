package main

import "fmt"

func main() {

	fmt.Println()

	x1 := newValue(2.0, "x1")
	x2 := newValue(0.0, "x2")

	w1 := newValue(-3.0, "w1")
	w2 := newValue(1.0, "w2")
	
	b := newValue(6.8813735870195432, "b")
	
	x1w1 := x1.times(w1, "x1w1")
	x2w2 := x2.times(w2, "x2w2")

	x1w1x2w2 := x1w1.add(x2w2, "x1w1x2w2")

	n := x1w1x2w2.add(b, "n")
	
	output := n.tanh("output")

	output.printAll()
	output.backwardsPropagate()
	output.printAll()


}