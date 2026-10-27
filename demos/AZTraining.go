package demos

import (
	"alphazero/games"
	"alphazero/neuralnets"
	"fmt"
	"log"
	"os"
)

func NACModelTesting() {

	// develop tests based on unambiguous best choices, use instead of random opponent play
	// check that game engine works. winner detection,
	// check that MCTS properly recognises immediate wins / loss prevention
	// Output loss function optimisation over time.
	// 

	filename := "new-model-26-oct-v17.json"
	err := TrainAZModel(filename)
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}
	
	// ExploreBuffer(filename)

	// model, err := neuralnets.CreateNetworkFromFile(filename)
	// if err != nil {
	// 	fmt.Println(err.Error())
	// 	os.Exit(1)
	// }

	// TestModel(10000, model, true)

	// ExploreModelPredictions(model, 800)

}

type NACtest struct {
	board games.NACBoard
	choice int
	passed bool
}

func ExploreModelPredictions(model *neuralnets.DenseNetwork, depth int) {

	tests := []NACtest{
		{
			board: games.NACBoard{
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
			},
			choice: 0,
		},
		{
			board: games.NACBoard{
				-1, 0, 0,
				0, 0, 0,
				0, 0, 0,
			},
			choice: 4,
		},
		{
			board: games.NACBoard{
				0, 0, 0,
				0, -1, 0,
				0, 0, 0,
			},
			choice: 0,
		},
		{
			board: games.NACBoard{
				1, 0, -1,
				0, -1, 0,
				0, 0, 0,
			},
			choice: 6,
		},
		{
			board: games.NACBoard{
				1, 0, -1,
				-1, -1, 0,
				1, 0, 0,
			},
			choice: 5,
		},
		{
			board: games.NACBoard{
				1, 1, 0,
				-1, 0, 1,
				-1, 0, -1,
		},
			choice: 2,
			},
		{
			board: games.NACBoard{
				1, 1, 0,
				-1, 0, 0,
				0, 0, -1,
		},
			choice: 2,
			},
		{
			board: games.NACBoard{
				1, 0, 0,
				-1, 0, -1,
				0, 0, 1,
		},
			choice: 4,
			},
		{
			board: games.NACBoard{
				1, 0, 0,
				-1, 0, -1,
				0, 1, 0,
		},
			choice: 4,
			},
		{
			board: games.NACBoard{
				-1, 0, 0,
				1, 0, 1,
				0, -1, 0,
		},
			choice: 4,
			},
		{
			board: games.NACBoard{
				0, -1, 0,
				0, 0, 0,
				1, -1, 1,
			},
			choice: 4,
		},
		{
			board: games.NACBoard{
				1, -1, 0,
				0, 0, 0,
				1, -1, 0,
			},
			choice: 3,
		},
		{
			board: games.NACBoard{
				0, -1, 1,
				0, 0, 0,
				1, -1, 0,
			},
			choice: 4,
		},
		
	}
	var testsPassed int
	for i, test := range tests {
		// if i != 8 {continue}
		val, err := games.MCTSMoveSelector(test.board, depth, model, 1)
		// fmt.Println(val)
		if val == test.choice {
			test.passed = true
			testsPassed ++
		}
		vals, err := games.MCTSMoveSelectorDistribution(test.board, depth, model, 1)
		if err != nil {
			fmt.Println("Error at testing stage", err.Error())
			os.Exit(1)
		}

		// rawVals := model.CallAZFinalFloat(test.board.ConvertToValues())

		if test.passed {
			fmt.Printf("Test %d passed\n", i)
		} else {
			fmt.Printf("Test %d failed\n", i)
		}
		
		test.board.PrintBoard()
		fmt.Println("Normalised policy")
		for i := 0; i < 3; i ++ {
			fmt.Printf(" %f | %f | %f\n", vals[3*i], vals[3*i + 1], vals[3*i + 2])
			fmt.Println()
		}
		// fmt.Println("Raw model outputs")
		// for i := 0; i < 3; i ++ {
		// 	fmt.Printf(" %f | %f | %f\n", rawVals[3*i], rawVals[3*i + 1], rawVals[3*i + 2])
		// }
		// fmt.Printf("Game value = %f\n\n", rawVals[9])
	}

	fmt.Printf(" %d / %d tests passed\n", testsPassed, len(tests))

}

func ExploreBuffer(modelFilename string) {
	net, err := neuralnets.CreateNetworkFromFile(modelFilename)
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	dirichletScale := 0.25
	dirichletAlpha := 0.3

	buff := neuralnets.CreateReplayBuffer(1000)

	gamesToExplore := 7
	explorationDepth := 400

	PopulateBuffer(
		buff,
		gamesToExplore,
		explorationDepth,
		net,
		dirichletScale,
		dirichletAlpha,
	)
	
	InspectBuffer(buff, 25)
}

func TrainAZModel(outputFilename string) error {

	az := neuralnets.NewDenseNetwork(neuralnets.DenseNetworkSpecification{
		LayerSizes: []int{
			// 9, 16, 16, 10,
			9, 32, 32, 10,
			// 9, 64, 64, 64, 10,
		},
		LayerActivations: []string{
			"", "relu", "relu", "az-output",
			// "", "relu", "relu", "relu", "az-output",
		},
	})

	// Model + Training parameters
	
	// current test (used for model v13 (10 cycles) + v14 (1 cycle) + v15 (10 cycles) + v16 (10 cycles))
	// test_runs := 1_000
	// dirichletScale := 0.3
	// dirichletAlpha := 0.5
	// explorationSteps := 800
	// MCTSgames := 25
	// bufferSize := 4000
	// numTrainingCycles := 10
	// learningRate := 0.0005
	// weightDecay := 0.00005
	// batchSize := 32
	// numBatches := 16
	// cyclesPerBatch := 1

	// next test
	// test_runs := 1_000
	// dirichletScale := 0.25
	// dirichletAlpha := 0.4
	// explorationSteps := 200
	// MCTSgames := 100
	// bufferSize := 10_000
	// numTrainingCycles := 5
	// learningRate := 0.001
	// weightDecay := 0.0005
	// batchSize := 64
	// numBatches := 100
	// cyclesPerBatch := 1

	test_runs := 10_000
	dirichletScale := 0.2
	dirichletAlpha := 0.5
	explorationSteps := 800
	MCTSgames := 50
	bufferSize := 8000
	numTrainingCycles := 20
	learningRate := 0.0001
	weightDecay := 0.00005
	batchSize := 64
	numBatches := 32
	cyclesPerBatch := 1

	// Pre-training
	fmt.Println("\nExpected Values pre-training")
	TestModel(test_runs, az, true)

	buf := neuralnets.CreateReplayBuffer(bufferSize)
	evs := []float64{}

	for i := 0; i < numTrainingCycles; i ++ {

		PopulateBuffer(
			buf,
			MCTSgames,
			explorationSteps,
			az,
			dirichletScale,
			dirichletAlpha,
		)
		// fmt.Printf("number of matches in buffer: %d . We expect c. %d items after %d MCTS searches\n",
		// 			buf.Content, MCTSgames * 8, MCTSgames)
		
		az.AZStochasticGradientDescent(
			learningRate,
			weightDecay,
			buf,
			batchSize,
			numBatches,
			cyclesPerBatch,
		)

		// ev := TestModel(test_runs, az, false)
		ev := -1.0
		// Post-training
		// if i % 1 == 0 {
			fmt.Printf("Training cycle %d complete. Current proper ev = %f\n", i, ev)
		// }
		if i % 5 == 0 {
			learningRate *= 0.98
			weightDecay *= 0.98
		}
		evs = append(evs, ev)
		// if ev >= 0.75 {
		// 	break
		// }

	}

	fmt.Println("\nExpected Values post-training")
	TestModel(test_runs, az, true)

	fmt.Println(evs)

	err := az.DumpNetworkToFile(outputFilename)
	if err != nil {
		return err
	}

	return nil


}

func InspectBuffer(buf *neuralnets.ReplayBuffer, noToInspect int) {

	frames := buf.Content
	for i := 0; i < noToInspect; i++ {
		history := buf.Histories[i]
		b := make(games.NACBoard, len(history.BoardState))
		copy(b, history.BoardState)

		fmt.Println()
		fmt.Printf("Board number %d / %d\n", i + 1, frames)
		fmt.Printf("Current player is: %d, Game value is: %f\n", history.CurrentPlayer, history.AssessedGameValue)
		fmt.Println(history.MCTSPolicy)
		b.PrintBoard()
	}

}

func PopulateBuffer(
		buf *neuralnets.ReplayBuffer,
		MCTSgames, explorationSteps int,
		model *neuralnets.DenseNetwork,
		dirichletScale, dirichletAlpha float64,
	) {

	for i := 0; i < MCTSgames; i ++ {
		err := games.MCTSBufferBuilder(
			explorationSteps,   	// exploration steps per move
			buf, 					// replay buffer
			model,  				// model
			dirichletScale,
			dirichletAlpha,
		)
		if err != nil {
			fmt.Printf("something wrong with MCTS, %s\n", err.Error())
			os.Exit(1)
		}
	}
}

func TestModel(runs int, model *neuralnets.DenseNetwork, print bool) float64 {

	TestRandomOpponents(runs, print)
	v := TestAZModelAgainstRandom(runs, model, print)
	v = TestProperAZModelAgainstRandom(runs/10, model, print)
	TestTwoAZModels(model, print)

	return v
}

func TestRandomOpponents(runs int, print bool) {

	// Random game simulations
	ev := 0.0
	for i := 0; i < runs; i ++ {
		gg := games.NACNewGame()
		errg := gg.PlayMatch(
			games.RandomMoveSelector,
			games.RandomMoveSelector,
			nil,
			false,
		)
		if errg != nil {
			log.Fatal("Issue playing the match: %w", errg)
		}
		ev += gg.GameValue
	}
	ev /= float64(runs)
	if print {
		fmt.Printf("ev of a game with both players randomly selecting moves is: %f\n", ev)
		fmt.Println("	With completely random play, EV is exactly 0.296886.")
	}
	// fmt.Println()
	// Where P1 wins = +1, P2 wins = -1, Draw = 0; EV = 0.296886.
	// We expect a P1 neural net to have a stronger EV vs a random P2
	// We expect a P1 random player to have a weaker EV vs a neural net P2
	// We expect a P1 neural net and P2 neural net to have EV = 0.0 as this is the value of the game under perfect play.

}

func TestAZModelAgainstRandom(runs int, model *neuralnets.DenseNetwork, print bool) float64 {

	ev := 0.0
	for i := 0; i < runs; i ++ {
		gg := games.NACNewGame()
		errg := gg.PlayMatch(
			games.ModelBasedMoveSelector,
			games.RandomMoveSelector,
			model,
			false,
		)
		if errg != nil {
			log.Fatal("Issue playing the match: %w", errg)
		}
		ev += gg.GameValue
	}
	ev /= float64(runs)
	if print {
		fmt.Printf("ev of a game with first player an AZ model and second randomly selecting moves is: %f\n", ev)
		fmt.Println("	For a trained model, we expect an EV of approx. 0.994792. Or something like that")
	}
	// fmt.Println()

	return ev

}

func TestProperAZModelAgainstRandom(runs int, model *neuralnets.DenseNetwork, print bool) float64 {

	ev := 0.0
	for i := 0; i < runs; i ++ {
		gg := games.NACNewGame()
		errg := gg.PlayMatch(
			games.ModelPlusMCTSBasedMoveSelector,
			games.RandomMoveSelector,
			model,
			false,
		)
		if errg != nil {
			log.Fatal("Issue playing the match: %w", errg)
		}
		ev += gg.GameValue
		// if i % 10 == 0 {
		// 	fmt.Printf("Finished proper run %d of %d\n", i, runs)
		// }
	}
	ev /= float64(runs)
	if print {
		fmt.Printf("ev of a game with first player a properly configured AZ model and second randomly selecting moves is: %f\n", ev)
		fmt.Println("	For a trained model, we expect an EV of approx. 0.994792. Or something like that")
	}
	// fmt.Println()

	return ev

}

func TestTwoAZModels(model *neuralnets.DenseNetwork, print bool) {
	// AZ models are deterministic, so we do not repeat runs.
	// 
	ev := 0.0
	gg := games.NACNewGame()
	errg := gg.PlayMatch(
		games.ModelBasedMoveSelector,
		games.ModelBasedMoveSelector,
		model,
		false,
	)
	if errg != nil {
		log.Fatal("Issue playing the match: %w", errg)
	}
	ev += gg.GameValue
	ev /= float64(1)
	if print {
		fmt.Printf("ev of a game with two players using AZ models to select moves is: %f\n", ev)
		fmt.Println("	With well trained models, ev should equal zero. Perfect play by both players results in a draw.")
	}
	// fmt.Println()

}
