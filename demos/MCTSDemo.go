package demos

import (
	"alphazero/games"
	"alphazero/neuralnets"
	"fmt"
	"os"
)

func TestMCTS() {


	g := games.NACNewGame()
	g.CurrentHead.Board.PrintBoard()

	dirichletScale := 0.25
	dirichletAlpha := 0.3

	buf := neuralnets.CreateReplayBuffer(100)
	az := neuralnets.NewDenseNetwork(neuralnets.DenseNetworkSpecification{
		LayerSizes: []int{
			9, 16, 16, 10,
		},
		LayerActivations: []string{
			"", "relu", "relu", "az-output",
		},
	})

	for i := 0; i < 10; i ++ {
		err := games.MCTSBufferBuilder(
			1000,   		// exploration steps per move
			buf, 			// replay buffer
			az,  			// model
			dirichletScale, // weights the noise when searching
			dirichletAlpha,			// noise source, to prevent deterministic search
		)
		if err != nil {
			fmt.Printf("something wrong with MCTS, %s\n", err.Error())
			os.Exit(1)
		}
	}

	frames := buf.Content
	for i := 0; i < frames; i++ {
		history := buf.Histories[i]
		b := make(games.NACBoard, len(history.BoardState))
		copy(b, history.BoardState)

		fmt.Println()
		fmt.Printf("Board number %d / %d\n", i + 1, frames)
		fmt.Printf("Current player is: %d, Game value is: %f\n", history.CurrentPlayer, history.AssessedGameValue)
		fmt.Println(history.MCTSPolicy)
		b.PrintBoard()
	}

	// g.CurrentHead.Board.PrintBoard()



}

func TestSingleMCTSRun() {

	g := games.NACNewGame()
	g.CurrentHead.Board.PrintBoard()

	dirichletScale := 0.25
	dirichletAlpha := 0.3

	buf := neuralnets.CreateReplayBuffer(100)
	az := neuralnets.NewDenseNetwork(neuralnets.DenseNetworkSpecification{
		LayerSizes: []int{
			9, 16, 10,
		},
		LayerActivations: []string{
			"", "relu", "az-output",
		},
	})

	err := games.MCTSBufferBuilder(
		10,   			// exploration steps per move
		buf, 			// replay buffer
		az,  			// model
		dirichletScale, // weights the noise when searching
		dirichletAlpha,	// noise source, to prevent deterministic search
	)
	if err != nil {
		fmt.Printf("something wrong with MCTS, %s\n", err.Error())
		os.Exit(1)
	}
	

	frames := buf.Content
	for i := 0; i < frames; i++ {
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