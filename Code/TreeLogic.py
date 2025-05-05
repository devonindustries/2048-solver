from GameLogic import *

# With the existing game logic, start the simulation process
class Node:
    '''
    A node class to form a tree for Expectimax / Minimax Tree Searching. Also considers:

    - Tree Depth
    - Score Matrices

    '''
    def __init__(self, state, prob=None):

        # Save the score matrix
        self.scoreMatrix = np.matrix([
            [-16,-12,-8,-5],
            [-1,-2,-3,-4],
            [1,2,3,4],
            [16,12,8,5]
        ])

        # Store the state matrix
        self.state = state

        # Store the relative probability of this event occuring.
        # Set to none if node is root.
        self.prob = prob

        # Also store the score given according to some
        # rule. Set to none if node is root.
        self.score = np.sum(np.multiply(state, self.scoreMatrix)) - (magnitude(self.state) ** 2) / score(self.state) if not(self.__isLosing()) else -np.inf

        # Store lists corresponding to possible states
        # after the following actions have been performed
        self.children = {}

        # Calculate the scores of each individual state according to some method.
        # For example, the expectimax search
        self.nextScores = {}

    def __repr__(self):
        return(f'{str(self.state)} with score {self.score}')
    
    def __isLosing(self):
        # Checks if the state is a losing state
        return np.array([(self.state == play(self.state, _)).all() for _ in 'lurd']).all()

    def __expandDir(self, dir):
        # Play the state in the intended direction, and then
        # generate a new Node for each state in the corresponding list
        baseState = play(self.state, dir)

        # Only continue if this is a valid move
        if (baseState == self.state).all(): return -1

        nextSpawns = simulateSpawns(baseState)

        # Do nothing if there are no possible spawns for this move
        if len(nextSpawns[0]) == 0: return -1

        # Generate the relative probabilities for each 
        # next state, given that a direction has been played.
        probs = [
            0.9 / len(nextSpawns[0]),
            0.1 / len(nextSpawns[1])
        ]

        # Add the children to the dictionary for this state expansion.
        self.children[dir] = [
            [Node(newSquare, probs[_]) for newSquare in nextSpawns[_]]
            for _ in range(2)
        ]

    def __zeroIfEmpty(self, l):
        return 0 if len(l) == 0 else l

    def expand(self, depth=1, epsilon=0.1, metricCode='minimax'):
        '''
        Search through all of the next possible states for all directions.

        - A depth of 0 means that only one expansion is performed.
        - Epsilon controls the probability of expanding a random branch.
        - Metric controls whether we are using the minimax or expectimax algorithm.
        '''

        # Map the metric to an actual function
        metric = {
            'minimax' : np.min,
            'expectimax' : np.mean
        }[metricCode]

        # Expand for each direction
        for dir in [_ for _ in 'lurd']:
            self.__expandDir(dir)
        
        # Loop through each direction and calculate the scores
        for d in self.children.keys():
            self.nextScores[d] = 0.9 * metric(
                [_.score for _ in self.children[d][0]]
            ) + 0.1 * metric(
                [_.score for _ in self.children[d][1]]
            )

        # Now pick the optimal direction / a random direction to expand in on the next iteration
        d = self.nextScores
        if len(d) == 0: return 'x', -np.infty # Return negative infinity on loss
        nextDir = max(d, key=d.get) if np.random.random() > epsilon else np.random.choice(list(d.keys()))

        # Check the depth
        if depth == 0:
            # If we have gone far enough, return the best direction and score
            return nextDir, d[nextDir]
        
        else:
            # Make a dictionary to hold all of the scores for each direction
            c2dict, c4dict = [{x:[] for x in ['l', 'u', 'r', 'd', 'x']} for _ in range(2)]

            # Make a final dictionary to optimise
            fdict = dict()

            # Expand each of the children, and aggregate the maximum scores for each direction
            for child2 in self.children[nextDir][0]:
                res = child2.expand(depth=depth-1, epsilon=epsilon, metricCode=metricCode)
                c2dir, c2score = res
                c2dict[c2dir].append(c2score)

            for child4 in self.children[nextDir][1]:
                res = child4.expand(depth=depth-1, epsilon=epsilon, metricCode=metricCode)
                c4dir, c4score = res
                c4dict[c4dir].append(c4score)

            # Compile a final score for each direction from both dictionaries according to the metric
            for dir in [_ for _ in 'lurd']:
                fdict[dir] = 0.9 * metric(self.__zeroIfEmpty(c2dict[dir])) + 0.1 * metric(self.__zeroIfEmpty(c4dict[dir]))

            # Maximise the dictionary, and return the pair
            nextDir = max(fdict, key=fdict.get)
            
            # If the currently calculated scores are better, use them
            if nextDir in self.nextScores.keys():    
                if fdict[nextDir] < self.nextScores[nextDir]: 
                    fdict[nextDir] = self.nextScores[nextDir]
                else:
                    self.nextScores[nextDir] = fdict[nextDir]

            # Return the final direction and score
            return nextDir, fdict[nextDir]