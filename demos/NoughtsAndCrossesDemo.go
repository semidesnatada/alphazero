package demos

import (
	"alphazero/games"
	"fmt"
	"log"
)

func NoughtsAndCrossesDemo() {

	g := games.NACNewGame()
	g.CurrentHead.Board.PrintBoard()
	// fmt.Println(g.CurrentHead.PossibleMovesMap)

	// g.MakeMove(0)
	// g.CurrentHead.Board.PrintBoard()
	// fmt.Println(g.CurrentHead.PossibleMovesMap)
	
	// g.MakeMove(0)
	// g.CurrentHead.Board.PrintBoard()
	// fmt.Println(g.CurrentHead.PossibleMovesMap)

	// g.MakeMove(4)
	// g.CurrentHead.Board.PrintBoard()
	// fmt.Println(g.CurrentHead.PossibleMovesMap)

	// g.MakeMove(1)
	// g.CurrentHead.Board.PrintBoard()
	// fmt.Println(g.CurrentHead.PossibleMovesMap)

	// g.MakeMove(2)
	// g.CurrentHead.Board.PrintBoard()
	// fmt.Println(g.CurrentHead.PossibleMovesMap)

	// g.MakeMove(6)
	// g.CurrentHead.Board.PrintBoard()
	// fmt.Println(g.CurrentHead.PossibleMovesMap)

	// g.MakeMove(3)
	// g.CurrentHead.Board.PrintBoard()
	// fmt.Println(g.CurrentHead.PossibleMovesMap)

	// Match between two random players
	err := g.PlayMatch(
		games.RandomMoveSelector,
		games.RandomMoveSelector,
		nil,
		true,
	)
	if err != nil {
		log.Fatal("Issue playing the match: %w", err)
	}

	// Random game simulations
	ev := 0.0
	runs := 100_000
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
	fmt.Printf("ev of a game with randomly selected moves is: %f\n", ev)
	// Where P1 wins = +1, P2 wins = -1, Draw = 0; EV = 0.296886.
	// We expect a P1 neural net to have a stronger EV vs a random P2
	// We expect a P1 random player to have a weaker EV vs a neural net P2
	// We expect a P1 neural net and P2 neural net to have EV = 0.0 as this is the value of the game under perfect play.

}