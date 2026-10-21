package demos

import (
	game "alphazero/games"
	"fmt"
)

func ConnectFourDemo() {

	g := game.NewGame()
	g.PrintBoard()

	g.PlayMove(0)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PlayMove(1)
	g.PrintBoard()

	g.PlayMove(2)
	g.PlayMove(3)
	g.PlayMove(2)
	g.PlayMove(1)
	g.PlayMove(6)
	g.PlayMove(6)
	g.PlayMove(5)
	g.PlayMove(6)
	g.PlayMove(5)
	g.PlayMove(6)
	g.PlayMove(5)
	g.PlayMove(6)
	g.PlayMove(6)
	g.PlayMove(5)

	g.PrintBoard()

	fmt.Printf("%49b\n", g.CurrentHead.BoardState.Playing)
	fmt.Printf("%49b\n", g.CurrentHead.BoardState.AllPlayed)


}