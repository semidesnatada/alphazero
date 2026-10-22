package games

import (
	"errors"
	"fmt"
)

func C4NewGame() C4GameState {

	return C4GameState{
		CurrentHead: &C4GameStateNode{
			PossibleMovesMap: map[int]bool{
				0:true,1:true,2:true,3:true,4:true,5:true,6:true,
			},
			TimesVisited: 0,
			Wins: 0,
			Board: C4Board{
				Playing: 0,
				AllPlayed: 0,
			},
		},
		PrevState: nil,
		GameOver: false,
	}

}

func (bs C4Board) PrintBoard() {

	playing := bs.Playing
    allPlayed := bs.AllPlayed

    for row := 5; row >= 0; row-- {
        for col := 0; col < 7; col++ {
            bit := uint64(1) << (col*7 + row)
            switch {
				case playing&bit != 0:
					fmt.Print(" X ")
				case allPlayed&bit != 0:
					fmt.Print(" O ")
				default:
					fmt.Print(" . ")
            }
        }
        fmt.Println()
	}
    fmt.Println(" 0  1  2  3  4  5  6")

}


func (bs C4Board) PossibleMoves() map[int]bool {
	valids := make(map[int]bool)
	for i := 0; i <= 6; i++ {
		bit := uint64(1) << (i*7 + 5)
		if bit & bs.AllPlayed == 0 {
			valids[i] = true
		} else {
			valids[i] = false
		}
	}

	return valids
}

func (bs C4Board) IsGameOver() bool {
    bits := bs.Playing
    for _, shift := range []uint64{1, 7, 6, 8} {
        pairs := bits & (bits >> shift)
        if pairs&(pairs>>(2*shift)) != 0 {
			return true
        }
    }
    return false
}

func (gs *C4GameState) MakeMove(colMove int) error {

	if possible := gs.CurrentHead.PossibleMovesMap[colMove]; !possible || gs.GameOver {
		if !gs.GameOver {
			fmt.Printf("move not possible in col %d\n", colMove)
		} else {
			fmt.Println("game is over")
		}
		return errors.New("move not possible")
	}

	gs.PrevState = gs.CurrentHead
	board := gs.PrevState.Board

	for row := 0; row < 6; row++ {
        bit := uint64(1) << (colMove*7 + row)
        if board.AllPlayed&bit == 0 {
            board.Playing |= bit
            board.AllPlayed |= bit
			if board.IsGameOver() {
				gs.GameOver = true
				fmt.Println("game has been won")
			}
			board.Playing ^= board.AllPlayed
			break
        }
    }

	newHead := &C4GameStateNode{
		PossibleMovesMap: board.PossibleMoves(),
		TimesVisited: 0,
		Wins: 0,
		Board: board,
	}
	gs.CurrentHead = newHead

	return nil
} 

