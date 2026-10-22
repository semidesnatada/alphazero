package demos

import (
	"alphazero/neuralnets"
	"fmt"
)

func SimpleBinaryClassifierMLP() {
	binaryMLP := neuralnets.NewDenseNetwork(
		neuralnets.DenseNetworkSpecification{
			LayerSizes: []int{3, 4, 4, 1},
			LayerActivations: []string{"", "tanh", "tanh", "tanh"},
	})
	xs := [][]*neuralnets.Value{
		{
			neuralnets.NewValue(2.0, "x1"),
			neuralnets.NewValue(3.0, "x2"),
			neuralnets.NewValue(-1.0, "x3"),
		},
		{
			neuralnets.NewValue(3.0, "x1"),
			neuralnets.NewValue(-1.0, "x2"),
			neuralnets.NewValue(0.5, "x3"),
		},
		{
			neuralnets.NewValue(0.5, "x1"),
			neuralnets.NewValue(1.0, "x2"),
			neuralnets.NewValue(1.0, "x3"),
		},
		{
			neuralnets.NewValue(1.0, "x1"),
			neuralnets.NewValue(1.0, "x2"),
			neuralnets.NewValue(-1.0, "x3"),
		},
	}
	ys := []*neuralnets.Value{
		neuralnets.NewValue(1.0, "y1"),
		neuralnets.NewValue(-1.0, "y2"),
		neuralnets.NewValue(-1.0, "y3"),
		neuralnets.NewValue(1.0, "y4"),
	}

	fmt.Println("\nPre-training")
	binaryMLP.PrintBinaryMLPLoss(xs, ys)
	binaryMLP.BinaryMLPStochasticGradientDescent(
		0.05, //learning rate
		0.0, //weight decay
		500, //number of training cycles
		xs, //training inputs
		ys, //training outputs / actuals
	)

	fmt.Println("\nPost-training")
	binaryMLP.PrintBinaryMLPLoss(xs, ys)
}