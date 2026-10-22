package games

import (
	"alphazero/neuralnets"
	"errors"
	"fmt"
)

func NACNewGame() NACGameState {

	return NACGameState{
		CurrentHead: &NACGameStateNode{
			PossibleMovesMap: map[int]bool{
				0:true,1:true,2:true,
				3:true,4:true,5:true,
				6:true,7:true,8:true,
			},
			TimesVisited: 0,
			Wins: 0,
			Board: NACBoard{
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
			},
			// current player is the player who will choose the move on this node
			CurrentPlayer: 1,
		},
		PrevState: nil,
		GameOver: false,
	}

}

func (nb NACBoard) PossibleMoves() map[int]bool {

	moves := make(map[int]bool)

	for i, val := range nb {
		if val == 0 {
			moves[i] = true
		}
	}

	return moves
}

func (nb NACBoard) IsGameOver() (bool, int) {

	// verts
	for i := 0; i < 3; i ++ {
		if sum := nb[i] + nb[i+3] + nb[i+6]; sum == 3 || sum == -3 {
			return true, sum / 3
		}
	}		

	// horizs
	for i := 0; i < 3; i ++ {
		if sum := nb[i * 3] + nb[i * 3 + 1] + nb[i * 3 + 2]; sum == 3 || sum == -3 {
			return true, sum / 3
		}
	}

	// diags
	if sum := nb[0] + nb[4] + nb[8]; sum == 3 || sum == -3 {
		return true, sum / 3
	}
	if sum := nb[2] + nb[4] + nb[6]; sum == 3 || sum == -3 {
		return true, sum / 3
	}

	emptySquares := 0
	for i := 0; i < 9; i ++ {
		if x := nb[i]; x == 0 {
			emptySquares ++
		}
	}
	if emptySquares == 0 {
		return true, 0
	}

	return false, 0
}

func (nb NACBoard) PrintBoard() {

	// requires a 3x3 game board

	for i := 0; i < 3; i ++ {
		fmt.Printf("Row %d : ", i)
		for j := 0; j < 3; j ++ {
			var val string
			if stat := nb[i * 3 + j]; stat == 1 {
				val = " X"
			} else if stat == -1 {
				val = " O"
			} else {
				val = " ."
			}
			fmt.Printf(" %s ", val)
		}
		fmt.Printf("\n")
	}
	fmt.Println("Col   :   1   2   3  ")

}

func (gs *NACGameState) MakeMove(move int) error {

	if possible := gs.CurrentHead.PossibleMovesMap[move]; !possible || gs.GameOver {
		if !gs.GameOver {
			return fmt.Errorf("move not possible in row %d col %d\n", move / 3, move % 3)
		} else {
			// handles draws too
			return fmt.Errorf("game is already over")
		}
	}

	gs.PrevState = gs.CurrentHead
	board := gs.PrevState.Board
	
	board[move] = gs.CurrentHead.CurrentPlayer

	if over, val := board.IsGameOver(); over {
		gs.GameOver = true
		gs.GameValue = float64(val)
		// fmt.Printf("game is over after playing in row: %d, col: %d\n", move / 3, move % 3)
	}

	newHead := &NACGameStateNode{
		PossibleMovesMap: board.PossibleMoves(),
		TimesVisited: 0,
		Wins: 0,
		Board: board,
		CurrentPlayer: -1 * gs.CurrentHead.CurrentPlayer,
	}
	gs.CurrentHead = newHead

	return nil
}

func (b NACBoard) ConvertToValues() []*neuralnets.Value {

	out := make([]*neuralnets.Value, len(b))

	for i, square := range b {
		out[i] = neuralnets.NewValue(float64(square), fmt.Sprintf("square-%d", i))
	}

	return out

}

func (gs *NACGameState) PlayMatch(
	moveSelectorP1 func(map[int]bool, NACBoard, *neuralnets.DenseNetwork) (int, error),
	moveSelectorP2 func(map[int]bool, NACBoard, *neuralnets.DenseNetwork) (int, error),
	print bool,
) error {

	for ;; {
		var move int
		var mErr error

		if gs.CurrentHead.CurrentPlayer == 1 {
			move, mErr = moveSelectorP1(gs.CurrentHead.PossibleMovesMap, gs.CurrentHead.Board, nil)
		} else if gs.CurrentHead.CurrentPlayer == -1 {
			move, mErr = moveSelectorP2(gs.CurrentHead.PossibleMovesMap, gs.CurrentHead.Board, nil)
		}
		
		if mErr != nil {
			return fmt.Errorf("Error selecting move: %w", mErr)
		}

		if err := gs.MakeMove(move); err != nil {
			return fmt.Errorf("Error processing move: %w", err)
		}
		if gs.GameOver {
			if print {
				gs.CurrentHead.Board.PrintBoard()
				fmt.Printf("Game result is: %f\n", gs.GameValue)
			}
			break
		}
	}

	return nil

}

func RandomMoveSelector(validMoves map[int]bool, b NACBoard, n *neuralnets.DenseNetwork) (int, error) {
	// A standard move selector function which select only random moves	
	// Note that when used, the input n should == nil

	if len(validMoves) <= 0 {
		return -1, errors.New("no valid moves to select from")
	}

	// if validMoves is not empty, always returns a result
	move := randomKey(validMoves)

	return move, nil
}

func ModelBasedMoveSelector(validMoves map[int]bool, b NACBoard, n *neuralnets.DenseNetwork) (int, error) {
	// A selector which chooses the move the provided model gives highest weight
	//  n should be an AZ network.

	if len(validMoves) <= 0 {
		return -1, errors.New("no valid moves to select from")
	}

	inputs := b.ConvertToValues()

	movePs := n.CallAZFinalFloat(inputs)

	tot := 0.0
	for i, moveP := range movePs {
		if _, ok := validMoves[i]; ok {
			tot += moveP
		} else {
			movePs[i] = 0.0
		}
	}
	argMax := -1
	currMax := 0.0
	for i := 0; i < len(movePs); i ++ {
		if movePs[i] > currMax {
			currMax = movePs[i]
			argMax = i
		}
		// not actually necessary to normalise, seeing as we only select largest value
		// movePs[i] /= tot
	}

	_, ok := validMoves[argMax]
	if !ok {
		return -1, errors.New("an error identifying the best move via model")
	}

	return argMax, nil
}