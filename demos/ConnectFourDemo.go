package demos

import (
	"alphazero/games"
	"fmt"
)

func ConnectFourDemo() {

	g := games.C4NewGame()
	g.CurrentHead.Board.PrintBoard()

	g.MakeMove(0)
	g.MakeMove(1)
	g.MakeMove(1)
	g.MakeMove(1)
	g.MakeMove(1)
	g.MakeMove(1)
	g.MakeMove(1)
	g.MakeMove(1)
	g.MakeMove(1)
	g.MakeMove(1)
	g.MakeMove(1)

	g.CurrentHead.Board.PrintBoard()

	g.MakeMove(2)
	g.MakeMove(3)
	g.MakeMove(2)
	g.MakeMove(1)
	g.MakeMove(6)
	g.MakeMove(6)
	g.MakeMove(5)
	g.MakeMove(6)
	g.MakeMove(5)
	g.MakeMove(6)
	g.MakeMove(5)
	g.MakeMove(6)
	g.MakeMove(6)
	g.MakeMove(5)

	g.CurrentHead.Board.PrintBoard()

	fmt.Printf("%49b\n", g.CurrentHead.Board.Playing)
	fmt.Printf("%49b\n", g.CurrentHead.Board.AllPlayed)


}