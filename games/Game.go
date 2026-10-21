package game

import (
	"errors"
	"fmt"
)

type GameState struct {
	CurrentHead *GameStateNode
	PrevState *GameStateNode
	GameOver bool
}

type GameStateNode struct {
	PossibleMoves map[int]bool
	TimesVisited int
	Wins int
	BoardState BoardState
}

type BoardState struct {
	Playing uint64
	AllPlayed uint64
}

func (bs *BoardState) checkWin() bool {
    bits := bs.Playing
    for _, shift := range []uint64{1, 7, 6, 8} {
        pairs := bits & (bits >> shift)
        if pairs&(pairs>>(2*shift)) != 0 {
			return true
        }
    }
    return false
}

func NewGame() GameState {

	return GameState{
		CurrentHead: &GameStateNode{
			PossibleMoves: map[int]bool{
				0:true,1:true,2:true,3:true,4:true,5:true,6:true,
			},
			TimesVisited: 0,
			Wins: 0,
			BoardState: BoardState{
				Playing: 0,
				AllPlayed: 0,
			},
		},
		PrevState: nil,
		GameOver: false,
	}

}

func (gs GameState) PrintBoard() {

	playing := gs.CurrentHead.BoardState.Playing
    allPlayed := gs.CurrentHead.BoardState.AllPlayed

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

func (gs *GameState) PlayMove(colMove int) error {

	if possible := gs.CurrentHead.PossibleMoves[colMove]; !possible || gs.GameOver {
		if !gs.GameOver {
			fmt.Printf("move not possible in col %d\n", colMove)
		} else {
			fmt.Println("game is over")
		}
		return errors.New("move not possible")
	}

	gs.PrevState = gs.CurrentHead
	bs := gs.PrevState.BoardState

	for row := 0; row < 6; row++ {
        bit := uint64(1) << (colMove*7 + row)
        if bs.AllPlayed&bit == 0 {
            bs.Playing |= bit
            bs.AllPlayed |= bit
			if bs.checkWin() {
				gs.GameOver = true
				fmt.Println("game has been won")
			}
			bs.Playing ^= bs.AllPlayed
			break
        }
    }

	newHead := &GameStateNode{
		PossibleMoves: bs.getValidMoves(),
		TimesVisited: 0,
		Wins: 0,
		BoardState: bs,
	}
	gs.CurrentHead = newHead

	return nil
} 

func (bs BoardState) getValidMoves() map[int]bool {
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