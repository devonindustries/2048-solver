import numpy as np

from GameLogic import *

# --------
# CONSTANTS
# --------
directionList = [_ for _ in 'lurd']

# --------
# CLASSES
# --------
class State:
    '''
    Stores a state along with all the possible next moves.
    '''
    def __init__(self, grid):

        # Store the grid for the state
        self.grid = grid

        # Determine the next possible moves for the state.
        # If there are no legal moves, keep empty
        self.legalMoves = []
        for move in directionList:
            if not (play(self.grid, move) == self.grid).all():
                self.legalMoves.append(move)

        # Set a losing flag
        self.losing = len(self.legalMoves) == 0

class MCTSNode:
    '''
    Stores a node for the Monte-Carlo Tree Search algorithm for the game 2048.
    '''
    def __init__(self, state, parentNode=None, move=None, explorationFactor=1.5):

        self.state = state # Store the state corresponding to this 
        self.parentNode = parentNode # Points to the parent node
        # self.prob = prob # The local probability of this node being selected
        self.move = move # The previous move that was played to reach this node
        self.exploration_factor = explorationFactor

        self.children = [] # Store all of the next nodes with their played directions
        self.visits = 0
        self.score = 0

    def UCB1(self):
        '''
        Calculates the individual UCB1 score for a node.
        '''
        return self.score / self.visits + self.exploration_factor * np.sqrt(2 * np.log(self.visits) / self.visits) if self.visits > 0 else np.inf

    def select(self):
        '''
        Returns the individual node with the highest UCB score.
        '''
        ucbs = {_ : _.UCB1() for _ in self.children}
        return max(ucbs, key = ucbs.get)

    def selectDirection(self):
        '''
        Returns the direction which has the highest UCB score.
        '''

        # Create a temporary dictionary for UCB scores. The indices are as follows:
        # 0 : The total score
        # 1 : The total number of visits
        stats = {x:[0,0] for x in self.state.legalMoves}
        ucbs = {}
        
        # Loop through the child nodes and group according to the direction played
        for child in self.children:

            # Add the scores and visits
            stats[child.move][0] += child.score
            stats[child.move][1] += child.visits

        # Calculate the UCB scores
        for move in stats.keys():
            if stats[move][1] == 0:
                ucbs[move] = np.inf
            else:
                ucbs[move] = stats[move][0] / stats[move][1] + self.exploration_factor * np.sqrt(2 * np.log(stats[move][1]) / stats[move][1])

        # Return the direction with the greatest score, breaking ties uniformly randomly
        return max(ucbs, key=ucbs.get)


    def expand(self):
        '''
        Reveal the next possible directions given a selected state.
        '''

        # Loop through each of the legal moves and expand the node
        for move in self.state.legalMoves:
            
            # Get the basis state after playing a move
            newState = play(self.state.grid, move)

            # Generate a class in the children list based on all the next possible states
            possibleSpawns = simulateSpawns(newState)
            getChildNodes = lambda l: [MCTSNode(
                state = State(_),
                parentNode = self,
                move = move,
                explorationFactor=self.exploration_factor
            ) for _ in l]

            # Expand the node
            self.children += getChildNodes(possibleSpawns[0])
            self.children += getChildNodes(possibleSpawns[1])

        
    def simulate(self, depth):
        '''
        Simulates until the next biggest square has been reached. Constrained by some computational budget.
        '''

        # Select the starting node to begin with
        node = self.state

        # Set the stopping condition to the maximum of the next biggest square, or the 16 tile
        nextSquare = max(np.max(self.state.grid) * 2, 8)

        # Incorporate a computational budget counter
        depthCount = 1

        # Pick random moves until we reach a terminating condition
        while not node.losing:

            # Play a random move
            nextMove = np.random.choice(node.legalMoves)
            basisNode = play(node.grid, nextMove)
            
            # Spawn a new tile according to the probability distribution
            selectionIndex = 0 if np.random.random() < 0.9 else 1
            possibleStates = simulateSpawns(basisNode)[selectionIndex]
            nextGrid = possibleStates[np.random.choice([_ for _ in range(len(possibleStates))])]

            if np.max(nextGrid) == nextSquare:
                # If we found the next square, count as win
                return 1
            elif depthCount > depth:
                # If we did not, count it as neither loss nor win
                return 0

            # Set the node to the newly generated node
            node = State(nextGrid)
            depthCount += 1

        # If the loop was broken, count a loss
        return -1

    def backpropagate(self, score):
        '''
        Traverse back up the tree updating the scores and visits.
        '''
        self.visits += 1
        self.score += score
        if self.parentNode:
            self.parentNode.backpropagate(score)

# --------
# FUNCTIONS
# --------
def Search(rootGrid, N, simulationDepth):

    # Define the root node based on the root state
    rootNode = MCTSNode(rootGrid)

    # Run for N simulations
    for _ in range(N):

        # Start again from the root node on every iteration
        node = rootNode

        # Select child nodes whilst they exist
        while node.children:
            node = node.select()

        # Expand the tree
        node.expand()

        # Play a random simulation
        score = -1
        if len(node.children) > 0:
            child_node = np.random.choice(node.children)
            score = child_node.simulate(simulationDepth)

        # Back-propagate through the tree to update the scores
        child_node.backpropagate(score)

    # Select the optimal move based on the combined UCT formula
    return rootNode.selectDirection()