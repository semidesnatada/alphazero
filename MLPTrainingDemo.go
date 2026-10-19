package main

import "fmt"

func SimpleBinaryClassifierMLP() {
	mm := newMLP([]int{3, 4, 4, 1})
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
	mm.printLoss(xs, ys)

	mm.SGD(0.05, 500, xs, ys)

	fmt.Println("\nPost-training")
	mm.printLoss(xs, ys)
}