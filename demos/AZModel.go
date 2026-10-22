package demos

import (
	"alphazero/neuralnets"
	"fmt"
	"os"
)

func AlphaZeroModelDemo() {

	az := neuralnets.NewDenseNetwork(neuralnets.DenseNetworkSpecification{
		LayerSizes: []int{
			9, 16, 16, 10,
		},
		LayerActivations: []string{
			"", "relu", "relu", "az-output",
		},
	})

	// az.Inspect()

	// Pre training
	BoardState := []int{-1, 0, 1, -1, 0, 0, 0, 0, 0}
	testInput := make([]*neuralnets.Value, len(BoardState))
	for i, square := range BoardState {
		testInput[i] = neuralnets.NewValue(float64(square), fmt.Sprintf("board-square-%d", i))
	}

	modelOutputs := az.CallAZFinal(testInput)
	neuralnets.PrintAZModelOutputs(modelOutputs, [2]int{3, 3})

	buffer := neuralnets.CreateReplayBuffer(1024)

	buffer.PopulateWithTest(9, 9)

	az.AZStochasticGradientDescent(
		0.01,
		0.001,
		buffer,
		128,
		32,
		1,
	)

	// Post Training
	modelOutputs = az.CallAZFinal(testInput)
	neuralnets.PrintAZModelOutputs(modelOutputs, [2]int{3, 3})

	fname := "tester-alpha-zero-22-Oct-2026.json"
	dErr := az.DumpNetworkToFile(fname)
	if dErr != nil {
		fmt.Println(dErr.Error())
		os.Exit(1)
	}

	by, bErr := neuralnets.CreateNetworkFromFile(fname)
	if bErr != nil {
		fmt.Println(bErr.Error())
		os.Exit(1)
	}

	modelOutputs = by.CallAZFinal(testInput)
	neuralnets.PrintAZModelOutputs(modelOutputs, [2]int{3, 3})

	// With random initial weights and random training data, we see that 
	// the neural net trains towards giving every move equal weight, as we expect.

}