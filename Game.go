package main

import (
	"errors"
	"fmt"
)

type GameState struct {
	currentHead *GameStateNode
	prevState *GameStateNode
	gameOver bool
}

type GameStateNode struct {
	possibleMoves map[int]bool
	timesVisited int
	wins int
	boardState boardState
}

type boardState struct {
	playing uint64
	allPlayed uint64
}

func (bs *boardState) checkWin() bool {
    bits := bs.playing
    for _, shift := range []uint64{1, 7, 6, 8} {
        pairs := bits & (bits >> shift)
        if pairs&(pairs>>(2*shift)) != 0 {
			return true
        }
    }
    return false
}

func newGame() GameState {

	return GameState{
		currentHead: &GameStateNode{
			possibleMoves: map[int]bool{
				0:true,1:true,2:true,3:true,4:true,5:true,6:true,
			},
			timesVisited: 0,
			wins: 0,
			boardState: boardState{
				playing: 0,
				allPlayed: 0,
			},
		},
		prevState: nil,
		gameOver: false,
	}

}

func (gs GameState) printBoard() {

	playing := gs.currentHead.boardState.playing
    allPlayed := gs.currentHead.boardState.allPlayed

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

func (gs *GameState) playMove(colMove int) error {

	if possible := gs.currentHead.possibleMoves[colMove]; !possible || gs.gameOver {
		if !gs.gameOver {
			fmt.Printf("move not possible in col %d\n", colMove)
		} else {
			fmt.Println("game is over")
		}
		return errors.New("move not possible")
	}

	gs.prevState = gs.currentHead
	bs := gs.prevState.boardState

	for row := 0; row < 6; row++ {
        bit := uint64(1) << (colMove*7 + row)
        if bs.allPlayed&bit == 0 {
            bs.playing |= bit
            bs.allPlayed |= bit
			if bs.checkWin() {
				gs.gameOver = true
				fmt.Println("game has been won")
			}
			bs.playing ^= bs.allPlayed
			break
        }
    }

	newHead := &GameStateNode{
		possibleMoves: bs.getValidMoves(),
		timesVisited: 0,
		wins: 0,
		boardState: bs,
	}
	gs.currentHead = newHead

	return nil
} 

func (bs boardState) getValidMoves() map[int]bool {
	valids := make(map[int]bool)
	for i := 0; i <= 6; i++ {
		bit := uint64(1) << (i*7 + 5)
		if bit & bs.allPlayed == 0 {
			valids[i] = true
		} else {
			valids[i] = false
		}
	}

	return valids
}