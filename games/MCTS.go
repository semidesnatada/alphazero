package games

import (
	"alphazero/neuralnets"
	"fmt"
	"math"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/stat/distmv"
)

// for reference
// type neuralnets.BufferItem struct {
// 	BoardState []int
// 	MCTSPolicy []float64
// 	AssessedGameValue float64
// }

func MCTSBufferBuilder(
		explorationStepsPerMove int,
		b *neuralnets.ReplayBuffer,
		model *neuralnets.DenseNetwork,
		dirichletScale, dirichletAlpha float64,
	) error {

	// implements the MCTS algorithm for one match of self play.
	//  add all new match frames to the replay buffer for training.

	gameTree := NACNewGame()

	// only valid for the NAC case, where board size == move options
	numMoveOptions := len(gameTree.CurrentHead.Board)

	gameStates := []neuralnets.BufferItem{}

	// continue playing until we reach an end state
	for ; !gameTree.GameOver ; {

		explorationRoot := NewMCTSRoot(&gameTree, model, dirichletScale, dirichletAlpha)
		// the exploration tree is initialised with the current state being assessed by the model and having a blank exploration slice

		// before we make each move, populate the MCTS tree
		for i := 0; i < explorationStepsPerMove; i ++ {
			explorationRoot.MCTSExplore(model)
		}

		// explorationRoot.Print(0)
		// os.Exit(1)

		// choose to make move which was visited the most in the exploration phase.
		maxVisits := 0
		argMax := -1
		// place to store count of each move visited
		explorationResults := make([]int, numMoveOptions)
		for i, child := range explorationRoot.children {
			result := child.visitCount
			explorationResults[i] = result
			if maxVisits < result {
				maxVisits = result
				argMax = i
			}
		}

		originalBoard := make(NACBoard, len(gameTree.CurrentHead.Board))
		copy(originalBoard, gameTree.CurrentHead.Board)
		// make the move
		err := gameTree.MakeMove(argMax)
		if err != nil {
			return fmt.Errorf("Error making move in MCTS cycle: %w", err)
		}

		totalVisits := 0
		for _, count := range explorationResults {
			totalVisits += count
		}
		if totalVisits != explorationStepsPerMove {
			fmt.Printf("Total visits = %d while exploration steps is set to be %d\n", totalVisits, explorationStepsPerMove)
		}
		// store the count of exploration visits, normalised to sum to 1 (prob distribution) to the game state
		explorationResultsNormed := make([]float64, len(explorationResults))
		for k, result := range explorationResults {
			explorationResultsNormed[k] = float64(result) / float64(totalVisits)
		}
		// invert all boards from the perspective of player 2 to aid training.
		normedBoard := make([]int, len(originalBoard))
		currPlay := gameTree.CurrentHead.CurrentPlayer
		for i, val := range originalBoard {
			normedBoard[i] = -1 * currPlay * val
		}
		newRecord := neuralnets.BufferItem{
				BoardState: normedBoard,
				MCTSPolicy: explorationResultsNormed,
				CurrentPlayer: currPlay,
			}
		gameStates = append(gameStates, 
			newRecord,
		)

		// fmt.Println(newRecord)

	}

	// now add all new training data from the completed match to buffer
	for _, newState := range gameStates {
		// need to invert every other newState, so that the neural net always plays as player with value 1
		// (so they get a mixture of training data from the perspective of player 1 and player 2)
		// renormedBoard := make(NACBoard, len(newState.BoardState))
		// for i, item := range newState.BoardState {
		// 	renormedBoard[i] = newState.CurrentPlayer * item
		// }
		b.AddItem(
			// renormedBoard,
			newState.BoardState,
			newState.MCTSPolicy,
			newState.CurrentPlayer * newState.CurrentPlayer,
			-1 * float64(newState.CurrentPlayer) * gameTree.GameValue,
			false,
		)
	}

	return nil

}

func MCTSSingleRunInspector(
		board NACBoard,
		explorationStepsPerMove int,
		model *neuralnets.DenseNetwork,
		currentPlayer int,
	) (int, error) {

	// implements the MCTS algorithm for one match of self play.
	//  add all new match frames to the replay buffer for training.

	// only valid for the NAC case, where board size == move options
	numMoveOptions := 9

	g := NACNewGame()
	// explorationRoot := NewMCTSRoot(&gameTree, model, 0, 0)
	p := NewMCTSRoot(&g, model, 0, 0)
	// fmt.Println(b)
	explorationRoot := NewMCTSNode(p, board, model)

	valids := board.PossibleMoves()
	childs := make(map[int]*MCTSNode)
	for i := range valids {
		childBoard := make(NACBoard, len(board))
		copy(childBoard, board)
		childBoard[i] = -1 * currentPlayer
		childs[i] = NewMCTSNode(explorationRoot, childBoard, model)
	}

	explorationRoot.children = childs

	// the exploration tree is initialised with the current state being assessed by the model and having a blank exploration slice

	// before we make each move, populate the MCTS tree
	for i := 0; i < explorationStepsPerMove; i ++ {
		explorationRoot.MCTSExplore(model)
	}

	// choose to make move which was visited the most in the exploration phase.
	maxVisits := 0
	argMax := -1
	// place to store count of each move visited
	explorationResults := make([]int, numMoveOptions)
	for i, child := range explorationRoot.children {
		result := child.visitCount
		explorationResults[i] = result
		if maxVisits < result {
			maxVisits = result
			argMax = i
		}
	}

	return argMax, nil

}

func MCTSMoveSelector(
		board NACBoard,
		explorationStepsPerMove int,
		model *neuralnets.DenseNetwork,
		currentPlayer int,
	) (int, error) {

	// implements the MCTS algorithm for one match of self play.
	//  add all new match frames to the replay buffer for training.

	// only valid for the NAC case, where board size == move options
	numMoveOptions := 9

	g := NACNewGame()
	// explorationRoot := NewMCTSRoot(&gameTree, model, 0, 0)
	p := NewMCTSRoot(&g, model, 0, 0)
	p.playerVal = currentPlayer * -1
	// fmt.Println(b)
	explorationRoot := NewMCTSNode(p, board, model)

	valids := board.PossibleMoves()
	childs := make(map[int]*MCTSNode)
	for i := range valids {
		childBoard := make(NACBoard, len(board))
		copy(childBoard, board)
		childBoard[i] = currentPlayer
		childs[i] = NewMCTSNode(explorationRoot, childBoard, model)
	}

	explorationRoot.children = childs

	// the exploration tree is initialised with the current state being assessed by the model and having a blank exploration slice

	// before we make each move, populate the MCTS tree
	for i := 0; i < explorationStepsPerMove; i ++ {
		explorationRoot.MCTSExplore(model)
	}

	// explorationResults := make([]int, numMoveOptions)
	// for i, child := range explorationRoot.children {
	// 	result := child.visitCount
	// 	explorationResults[i] = result
	// 	}

	// choose to make move which was visited the most in the exploration phase.
	maxVisits := 0
	argMax := -1
	// place to store count of each move visited
	explorationResults := make([]int, numMoveOptions)
	for i, child := range explorationRoot.children {
		result := child.visitCount
		explorationResults[i] = result
		if maxVisits < result {
			maxVisits = result
			argMax = i
		}
	}

	return argMax, nil

}

func MCTSMoveSelectorDistribution(
		board NACBoard,
		explorationStepsPerMove int,
		model *neuralnets.DenseNetwork,
		currentPlayer int,
	) ([]float64, error) {

	// implements the MCTS algorithm for one match of self play.
	//  add all new match frames to the replay buffer for training.

	// only valid for the NAC case, where board size == move options
	numMoveOptions := 9

	g := NACNewGame()
	// explorationRoot := NewMCTSRoot(&gameTree, model, 0, 0)
	p := NewMCTSRoot(&g, model, 0, 0)
	p.playerVal = currentPlayer * -1
	// fmt.Println(b)
	explorationRoot := NewMCTSNode(p, board, model)

	valids := board.PossibleMoves()
	childs := make(map[int]*MCTSNode)
	for i := range valids {
		childBoard := make(NACBoard, len(board))
		copy(childBoard, board)
		childBoard[i] = currentPlayer
		childs[i] = NewMCTSNode(explorationRoot, childBoard, model)
	}

	explorationRoot.children = childs

	// the exploration tree is initialised with the current state being assessed by the model and having a blank exploration slice

	// before we make each move, populate the MCTS tree
	for i := 0; i < explorationStepsPerMove; i ++ {
		explorationRoot.MCTSExplore(model)
	}

	// explorationRoot.Print(0)
	// os.Exit(1)

	// choose to make move which was visited the most in the exploration phase.
	sumVisits := 0
	// place to store count of each move visited
	explorationResults := make([]int, numMoveOptions)
	for i, child := range explorationRoot.children {
		result := child.visitCount
		explorationResults[i] = result
		sumVisits += result
	}

	dist := make([]float64, 9)
	for i, val := range explorationResults {
		dist[i] = float64(val) / float64(sumVisits)
	}

	return dist, nil

}

func NewMCTSRoot(
		game *NACGameState,
		model *neuralnets.DenseNetwork,
		dirichletScale, dirichletAlpha float64,
	) *MCTSNode {

	priors := model.CallAZFinalFloat(game.CurrentHead.Board.ConvertToValues())
	// should add dirichlet noise to the first n-1 values in priors; not yet implemented

	
	// Mask priors
	board := game.CurrentHead.Board
	invalids := board.ImpossibleMoves()
	for i := range invalids {
		priors[i] = 0.0
	}
	norm := 0.0
	for i := 0; i < len(priors) - 1; i ++ {
		norm += priors[i]
	}
	for i := 0; i < len(priors) - 1; i ++ {
		priors[i] /= norm
	}

	if dirichletScale > 0.0 {
		// Adding Dirichlet Noise.
		possibles := board.PossibleMoves()
		dirichletParameters := make([]float64, len(possibles))
		for i := range dirichletParameters {
			dirichletParameters[i] = dirichletAlpha
		}
		source := rand.New(rand.NewSource(int64(rand.Intn(100))))
		dirichletDist := distmv.NewDirichlet(dirichletParameters, source)
		noises := dirichletDist.Rand(nil)
		j := 0
		// note, that this is ranging over a map, which is not ordered.
		//  so to reproduce deterministically, we must change the below to 
		//  an ordered iteration. But it's not important now.
		for i := range possibles {
			priors[i] = priors[i] * (1.0 - dirichletScale) + dirichletScale * noises[j]
			j ++
		}
	}
		
	// Creating the new node.
	out := MCTSNode{
		parent: nil,
		visitCount: 0,
		priorProbs: priors[:len(priors) - 1],
		priorValue: priors[len(priors) - 1],
		cumValue: 0,
		playerVal: game.CurrentHead.CurrentPlayer,
		boardState: board,
	}

	valids := board.PossibleMoves()
	childs := make(map[int]*MCTSNode)
	for i := range valids {
		childBoard := make(NACBoard, len(board))
		copy(childBoard, board)
		childBoard[i] = -1 * game.CurrentHead.CurrentPlayer
		childs[i] = NewMCTSNode(&out, childBoard, model)
	}

	out.children = childs

	return &out

}

func NewMCTSNode(parentNode *MCTSNode, board NACBoard, model *neuralnets.DenseNetwork) *MCTSNode {

	priors := model.CallAZFinalFloat(board.ConvertToValues())
	// should add dirichlet noise to the first n-1 values in priors; not yet implemented
	
	// Mask priors
	invalids := board.ImpossibleMoves()
	for i := range invalids {
		priors[i] = 0.0
	}
	norm := 0.0
	for i := 0; i < len(priors) - 1; i ++ {
		norm += priors[i]
	}
	for i := 0; i < len(priors) - 1; i ++ {
		priors[i] /= norm
	}

	out := MCTSNode{
		parent: parentNode,
		children: make(map[int]*MCTSNode),
		visitCount: 0,
		priorProbs: priors[:len(priors) - 1],
		priorValue: priors[len(priors) - 1],
		playerVal: parentNode.playerVal * -1,
		boardState: board,
	}

	return &out

}

type MCTSNode struct {
	parent *MCTSNode
	children map[int]*MCTSNode
	visitCount int
	priorProbs []float64
	priorValue float64
	cumValue float64
	playerVal int
	boardState NACBoard
}

func (n *MCTSNode) Print(offset int) {
	// if offset > 2 {return}

	off := ""
	for i := 0; i < offset; i ++ {
		off += " "
	}

	fmt.Printf(
		"%s- vists: %d | prior value: %f | cum value: %f | player: %d | value/visits: %f\n",
		// | priors: %v\n",
		off, n.visitCount, n.priorValue, n.cumValue, n.playerVal, float64(n.cumValue)/float64(n.visitCount), ///n.priorProbs,
	)
	for _, child := range n.children {
		child.Print(offset + 1)
	}

}

func (n *MCTSNode) MCTSExplore(model *neuralnets.DenseNetwork) {

	n.visitCount ++

	// Apply PUCT to the root of the tree
	// Select the child of the current node which has the highest PUCT score.
	found, moveChoice := n.PUCT(1.25, model)
	if !found {
		n.boardState.PrintBoard()
		fmt.Println(n.boardState.IsGameOver())
		os.Exit(1)
	}

	childToExplore := n.children[moveChoice]
	gameIsOver, gameValue := childToExplore.boardState.IsGameOver()

	if len(childToExplore.children) == 0 || gameIsOver {
		// if the selected node does not already have its children included in the tree, or it is a terminal node, then it is a leaf node - stop
		// 	 -identify the valid moves of the leaf and add them to the tree
		var propValue float64
		if len(childToExplore.children) == 0 {
			validMoves := childToExplore.boardState.PossibleMoves()
			for i := range validMoves {
				newChildBoard := make(NACBoard, len(childToExplore.boardState))
				copy(newChildBoard, childToExplore.boardState)
				newChildBoard[i] = childToExplore.playerVal
				childToExplore.children[i] = NewMCTSNode(
					childToExplore,
					newChildBoard,
					model,
				)
			}
			propValue = childToExplore.priorValue
		}
		if gameIsOver {
			propValue = float64(gameValue)
			childToExplore.priorValue = propValue
			childToExplore.cumValue += propValue
		}
		childToExplore.visitCount ++
		// 	 -propagate the NN assessed value of the game back up the path taken from the root
		for k := childToExplore; k != nil ; k = k.parent {
			k.cumValue += -1 * propValue * float64(k.playerVal) * float64(n.playerVal)
		}

	} else {
		// else, apply PUCT to this node and choose the node with the highest PUCT score., aka apply the same function to this node.
		// fmt.Println("game status ,", gameIsOver)
		// childToExplore.boardState.IsGameOver()
		childToExplore.MCTSExplore(model)
	}

}

func (n *MCTSNode) PUCT(cP float64, model *neuralnets.DenseNetwork) (bool, int) {

	boardSize := len(n.boardState)
	possibles := n.boardState.PossibleMoves()

	// pucts := make([]float64, boardSize)
	max := math.Inf(-2)
	argMax := -1

	for i := 0; i < boardSize; i ++ {

		if _, isValid := possibles[i]; !isValid {
			continue
		}

		var exploitTerm, exploreTerm float64
		var childVisitCount float64

		if child, exists := n.children[i]; exists && child != nil {
			childVisitCount = float64(child.visitCount)
			if childVisitCount > 0 {
				exploitTerm = child.cumValue / childVisitCount
			}
		} else {
			childVisitCount = 0.0
			exploitTerm = 0.0
		}

		exploreTerm = cP * n.priorProbs[i] * math.Sqrt(float64(n.visitCount)) / (1 + childVisitCount)
		
		puct := exploitTerm + exploreTerm
		// pucts[i] = v
		if puct > max {
			max = puct
			argMax = i
		}
		
	}
	if argMax > -1 {
		return true, argMax
	}
	return false, argMax
	

}

	// 		1. initialise tree for use in MCTS
	//      2. initialise the game board (used binary bits; inversion switches player perspective)
	//      3. initialise empty list of game states
	//      4. for a loop until a match is over
	// 			- get list of valid moves from game board
	// 			- mask any invalid moves in probability distribution and prevent MCTS exploration to these invalid moves.
	// 			- use neural net policy head to assess a choice probability for each valid move. Add dirichlet noise to this.
	// 			
	// 			- run MCTS exploration, for a set number of steps (c. 100)
	// 				. use PUCT values to deterministically find the next leaf node (always choose highest PUCT)
	// 				. check if terminal state. if terminal, game is over so propagate result back up. ensure sign alternates.
	// 				. if not terminal, use neural net to evaluate the state, yielding move choice probabilities and expected 
	// 					game value at this leaf node. do not add dirichlet noise again. store at node
	// 				. if not terminal, propagate neural net assessed value back up the tree. ensure sign of value is properly 
	// 					alternated across each layer (to reflect different players)

	// 			- combine record of visits during MCTS to valid choices into final move choice distribution
	// 			- add board state and final move selection probabilities to game state list
	// 			- make move by selecting from the final move choice distribution (use temperature model so more likely 
	// 				to choice outside top option earlier in match).
	// 			- invert board state so next player can play
	// 			- rebase MCTS tree to next 
	// 			- check if game is over, if so, break and store outcome
	//      5. add / update each board state, final move choice probabilities (MCTS), and eventual match outcome data to 
	// 			storage (replay buffer), accounting for differing players on each move