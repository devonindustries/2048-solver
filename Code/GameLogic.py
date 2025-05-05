# IMPORTS
# --------
import numpy as np

# CONSTANTS
# --------
rotMap = {
    'l' : 2,
    'u' : 3,
    'r' : 0,
    'd' : 1
}

# FUNCTIONS
# --------
def blank():
    '''
    Creates a blank game grid
    '''
    return np.matrix([[0 for x in range(4)] for y in range(4)])

def score(s):
    '''
    Calculates the score of the matrix.
    '''
    return np.sum(s)

def magnitude(s):
    '''
    Calculates how many squares are occupied in the state matrix.
    '''
    return np.sum(s!=0)

def makeState(n, x, y):
    '''
    Create a blank state with the number n in the (x,y)th square.
    '''
    tmp = blank()
    tmp[x,y] = n
    return tmp

# Store the next state matrix using the previously defined function
# --------
allNextStates = [
    [makeState(n,x,y) for x in range(4) for y in range(4)] for n in [2,4]
]

def simulateSpawns(s):
    '''
    A function to simulate all of the next possible spawn states.
    '''

    nextStates = [[], []]

    # Loop through the next states and only add them to the states list if the magnitude upon addition is different
    for i in range(16):
        if magnitude(s) != magnitude(s + allNextStates[0][i]):
            nextStates[0].append(s + allNextStates[0][i])
            nextStates[1].append(s + allNextStates[1][i])

    return nextStates

def combineNums(row):
    '''
    A function to combine two numbers together on the same row.
    '''

    # Return the row if it is of length 1
    if len(row) == 1:
        return [0] * 3 + row

    newRow = []
    while len(row) > 1:
        x = row.pop()
        if x == row[-1]:
            newRow.append(x * 2)
            row.pop()
        else:
            newRow.append(x)

    if len(row) != 0:
        newRow.append(row.pop())

    newRow.reverse()
    return [0] * (4 - len(newRow)) + newRow

def play(s, dir):
    '''
    A function to play the game in a given direction. Uses the "rotMap" variable.
    '''

    # 1. Normalise the matrix so that the numbers travel right
    rot = rotMap[dir]
    s_prime = np.rot90(s, rot)

    # 2. Solve the matrix in the right hand direction
    s_new = []
    for row in s_prime:
        s_new.append( # Append to the new list
            combineNums( # Perform the function to add adjacent numbers
                row[row!=0].tolist()[0] # Filter out gaps on the row
            )
        )

    # 3. Return it to its original orientation
    return np.rot90(np.matrix(s_new), -rot)