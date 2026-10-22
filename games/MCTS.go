package games

func MCTS() {

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
	
}