# searchAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).

"""
This file contains all of the agents that can be selected to control Pacman.
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# Provided code below; do not modify this section
#######################################################

class SearchAgent(Agent):
    """
    A general search agent.  Uses a search function (dfs, bfs, ucs, or astar)
    to compute a path, then returns actions to follow that path.
    """
    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Get the search function
        if fn not in dir(search):
            raise AttributeError(fn + ' is not in search.py')
        func = getattr(search, fn)
        if 'heuristic' in func.__code__.co_varnames:
            # Provide heuristic
            if heuristic in globals():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function')
            self.searchFunction = lambda x: func(x, heuristic=heur)
        else:
            self.searchFunction = func
        # Get the search problem class
        if prob not in globals():
            raise AttributeError(prob + ' is not a problem type')
        self.searchType = globals()[prob]

    def registerInitialState(self, state):
        """Compute the path to goal and store actions."""
        if not self.searchFunction:
            raise Exception("No search function provided")
        problem = self.searchType(state)
        self.actions = self.searchFunction(problem)
        if self.actions is None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print(f'Path found with total cost of {totalCost}')
        if hasattr(problem, '_expanded'):
            print(f'Nodes expanded: {problem._expanded}')

    def getAction(self, state):
        if not hasattr(self, 'actionIndex'):
            self.actionIndex = 0
        if self.actionIndex < len(self.actions):
            action = self.actions[self.actionIndex]
            self.actionIndex += 1
            return action
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem for finding paths to a specific point.
    State space is (x,y) positions.
    """
    def __init__(self, gameState, costFn=lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start is not None:
            self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: unexpected layout')
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = (state == self.goal)
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if hasattr(__main__, '_display') and hasattr(__main__._display, 'drawExpandedCells'):
                __main__._display.drawExpandedCells(self._visitedlist)
        return isGoal

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x+dx), int(y+dy)
            if not self.walls[nextx][nexty]:
                cost = self.costFn((nextx,nexty))
                successors.append(((nextx,nexty), action, cost))
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)
        return successors

    def getCostOfActions(self, actions):
        if actions is None:
            return 999999
        x,y = self.startState
        cost = 0
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x+dx), int(y+dy)
            if self.walls[x][y]:
                return 999999
            cost += self.costFn((x,y))
        return cost

#######################################################
# Heuristics for PositionSearchProblem
#######################################################

def manhattanHeuristic(position: Tuple[int,int], problem: PositionSearchProblem, info: dict = {}) -> float:
    """The Manhattan distance heuristic."""
    x1,y1 = position
    x2,y2 = problem.goal
    return abs(x1-x2) + abs(y1-y2)


def euclideanHeuristic(position: Tuple[int,int], problem: PositionSearchProblem, info: dict = {}) -> float:
    """The Euclidean distance heuristic."""
    x1,y1 = position
    x2,y2 = problem.goal
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5

#######################################################
# Deliverable 2: CornersProblem and its heuristic
#######################################################

class CornersProblem(search.SearchProblem):
    """Find a path through all four corners of a layout."""
    def __init__(self, startingGameState: pacman.GameState):
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right,1), (right,top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0
        # State: (position, visitedCorners)
        self.startState = (self.startingPosition, frozenset())

    def getStartState(self):
        return self.startState

    def isGoalState(self, state: Any) -> bool:
        _, visited = state
        return len(visited) == 4

    def getSuccessors(self, state: Any) -> List[Tuple[Any, Any, float]]:
        successors = []
        position, visited = state
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x+dx), int(y+dy)
            if not self.walls[nextx][nexty]:
                nextPos = (nextx, nexty)
                nextVisited = visited
                if nextPos in self.corners and nextPos not in visited:
                    nextVisited = visited | {nextPos}
                successors.append(((nextPos, nextVisited), action, 1))
        self._expanded += 1
        return successors

    def getCostOfActions(self, actions) -> float:
        if actions is None:
            return 999999
        x,y = self.startingPosition
        cost = 0
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x+dx), int(y+dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


def cornersHeuristic(state: Any, problem: CornersProblem) -> float:
    """Heuristic: max Manhattan distance to any unvisited corner."""
    position, visited = state
    unvisited = [c for c in problem.corners if c not in visited]
    if not unvisited:
        return 0
    return max(util.manhattanDistance(position, c) for c in unvisited)

class AStarCornersAgent(SearchAgent):
    """Agent for CornersProblem using A* and cornersHeuristic."""
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

#######################################################
# Deliverable 2: FoodSearchProblem and its heuristic
#######################################################

class FoodSearchProblem:
    """Find a path that collects all the food in the game."""
    def __init__(self, startingGameState: pacman.GameState):
        self.startingGameState = startingGameState
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self._expanded = 0
        self.heuristicInfo = {}

    def getStartState(self) -> Any:
        return self.start

    def isGoalState(self, state) -> bool:
        return state[1].count() == 0

    def getSuccessors(self, state) -> List[Tuple[Any, Any, float]]:
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x+dx), int(y+dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions) -> float:
        if actions is None:
            return 999999
        x,y = self.start[0]
        cost = 0
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x+dx), int(y+dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    """Agent for FoodSearchProblem using A* and foodHeuristic."""
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points using BFS.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1] and not walls[x2][y2]
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))


def foodHeuristic(state: Tuple[Tuple[int,int], Any], problem: FoodSearchProblem) -> float:
    """
    Heuristic: maximum maze distance to any remaining food dot, with caching.
    """
    position, foodGrid = state
    foods = foodGrid.asList()
    if not foods:
        return 0
    # compute and cache maze distances
    distances = []
    for food in foods:
        key = (position, food)
        if key not in problem.heuristicInfo:
            problem.heuristicInfo[key] = mazeDistance(position, food, problem.startingGameState)
        distances.append(problem.heuristicInfo[key])
    return max(distances)
