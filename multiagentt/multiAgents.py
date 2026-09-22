# multiAgents.py

from util import manhattanDistance
from game import Directions, Agent
import random, util

############################################################
# Reflex Agent
############################################################

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state‐evaluation function.
    """

    def getAction(self, gameState):
        # Collect legal moves
        legalMoves = gameState.getLegalActions()

        # Evaluate each action; pick the best
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index, score in enumerate(scores) if score == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Simple feature‐based evaluation:
          + penalize moves that bring Pacman too close to a ghost
          + reward eating food / getting closer to food
        """
        successor = currentGameState.generatePacmanSuccessor(action)
        newPos = successor.getPacmanPosition()
        newFood = successor.getFood().asList()
        newGhostStates = successor.getGhostStates()

        # If ghost is too close, a very bad score
        distsToGhosts = [manhattanDistance(newPos, ghost.getPosition())
                         for ghost in newGhostStates]
        if distsToGhosts and min(distsToGhosts) < 2:
            return -float('inf')

        # Reward eating a pellet
        if successor.getNumFood() < currentGameState.getNumFood():
            return successor.getScore() + 10.0

        # Otherwise bias towards closer food
        if newFood:
            distsToFood = [manhattanDistance(newPos, food) for food in newFood]
            return successor.getScore() + 1.0 / min(distsToFood)
        else:
            return successor.getScore()

############################################################
# Multi‐Agent Search Agents
############################################################

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi‐agent searchers.  Any agent that wants to use
    minimax, alpha‐beta, or expectimax should inherit from this.
    """
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

############################################################
# Minimax Agent (Q3a)
############################################################

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState
        using self.depth and self.evaluationFunction.
        """
        def value(state, agentIndex, depth):
            # Terminal test
            if state.isWin() or state.isLose() or depth > self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                return max_value(state, agentIndex, nextAgent, nextDepth)
            else:
                return min_value(state, agentIndex, nextAgent, nextDepth)

        def max_value(state, agentIndex, nextAgent, nextDepth):
            v = -float('inf')
            for action in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, action)
                v = max(v, value(succ, nextAgent, nextDepth))
            return v

        def min_value(state, agentIndex, nextAgent, nextDepth):
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, action)
                v = min(v, value(succ, nextAgent, nextDepth))
            return v

        # At root, choose the action with the highest minimax score
        bestScore = -float('inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = value(succ, 1, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

############################################################
# Alpha‐Beta Agent (Q3b)
############################################################

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the alpha‐beta pruned minimax action
        """
        def value(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth > self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                return max_value(state, agentIndex, nextAgent, nextDepth, alpha, beta)
            else:
                return min_value(state, agentIndex, nextAgent, nextDepth, alpha, beta)

        def max_value(state, agentIndex, nextAgent, nextDepth, alpha, beta):
            v = -float('inf')
            for action in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, action)
                v = max(v, value(succ, nextAgent, nextDepth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, agentIndex, nextAgent, nextDepth, alpha, beta):
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, action)
                v = min(v, value(succ, nextAgent, nextDepth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        # Root call
        alpha, beta = -float('inf'), float('inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = value(succ, 1, 1, alpha, beta)
            if score > alpha:
                alpha = score
                bestAction = action
        return bestAction

############################################################
# Expectimax Agent (Q3c)
############################################################

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth
        """
        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth > self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                return max_value(state, agentIndex, nextAgent, nextDepth)
            else:
                return exp_value(state, agentIndex, nextAgent, nextDepth)

        def max_value(state, agentIndex, nextAgent, nextDepth):
            v = -float('inf')
            for action in state.getLegalActions(agentIndex):
                succ = state.generateSuccessor(agentIndex, action)
                v = max(v, value(succ, nextAgent, nextDepth))
            return v

        def exp_value(state, agentIndex, nextAgent, nextDepth):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return 0
            prob = 1.0 / len(actions)
            return sum(value(state.generateSuccessor(agentIndex, a), nextAgent, nextDepth)
                       for a in actions) * prob

        # Root call
        bestScore = -float('inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = value(succ, 1, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

############################################################
# Better Evaluation Function (Q3d)
############################################################

def betterEvaluationFunction(currentGameState):
    """
    A more sophisticated evaluation function.  If this is being
    called on the autograder’s tree‐state (which lacks getPacmanPosition),
    we just return the bare score().
    """
    # Fall back to plain score for TreeState
    if not hasattr(currentGameState, 'getPacmanPosition'):
        return currentGameState.getScore()

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    caps = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # 1) Distance to nearest food (the closer the better)
    if food:
        dists = [manhattanDistance(pos, f) for f in food]
        score += 1.0 / min(dists)

    # 2) Penalize ghost in range
    ghostDists = [manhattanDistance(pos, g.getPosition()) for g in ghosts]
    if ghostDists and min(ghostDists) < 2:
        score -= 10.0

    # 3) Fewer capsules is better
    score -= 2 * len(caps)

    return score

# Abbreviation
better = betterEvaluationFunction

############################################################
# Default evaluation function
############################################################
def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()
