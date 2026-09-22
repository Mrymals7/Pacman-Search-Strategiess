"""
search.py
----------

Implement generic search algorithms for Pacman here.
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem but doesn't implement any of the methods.
    You do not need to change anything in this class.
    """
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        state: Search state
        For a given state, should return a list of triples
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
        actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Returns a list of actions that reaches the goal.
    """
    stack = util.Stack()
    stack.push((problem.getStartState(), []))
    visited = set()

    while not stack.isEmpty():
        state, path = stack.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    stack.push((successor, path + [action]))
    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    Returns a list of actions that reaches the goal.
    """
    queue = util.Queue()
    queue.push((problem.getStartState(), []))
    visited = set([problem.getStartState()])

    while not queue.isEmpty():
        state, path = queue.pop()
        if problem.isGoalState(state):
            return path
        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                visited.add(successor)
                queue.push((successor, path + [action]))
    return []


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    Returns a list of actions that reaches the goal.
    """
    pq = util.PriorityQueue()
    start = problem.getStartState()
    pq.push((start, []), 0)
    best_cost = {start: 0}

    while not pq.isEmpty():
        state, path = pq.pop()
        cost_so_far = problem.getCostOfActions(path)
        if problem.isGoalState(state):
            return path
        if cost_so_far > best_cost.get(state, float('inf')):
            continue
        for successor, action, stepCost in problem.getSuccessors(state):
            new_path = path + [action]
            new_cost = cost_so_far + stepCost
            if new_cost < best_cost.get(successor, float('inf')):
                best_cost[successor] = new_cost
                pq.push((successor, new_path), new_cost)
    return []


def nullHeuristic(state, problem=None):
    """
    A trivial heuristic function.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    Returns a list of actions that reaches the goal.
    """
    pq = util.PriorityQueue()
    start = problem.getStartState()
    start_h = heuristic(start, problem)
    pq.push((start, []), start_h)
    best_cost = {start: 0}

    while not pq.isEmpty():
        state, path = pq.pop()
        cost_so_far = problem.getCostOfActions(path)
        if problem.isGoalState(state):
            return path
        if cost_so_far > best_cost.get(state, float('inf')):
            continue
        for successor, action, stepCost in problem.getSuccessors(state):
            new_path = path + [action]
            new_cost = cost_so_far + stepCost
            f = new_cost + heuristic(successor, problem)
            if new_cost < best_cost.get(successor, float('inf')):
                best_cost[successor] = new_cost
                pq.push((successor, new_path), f)
    return []


# Abbreviations
bfs = breadthFirstSearch  # noqa
dfs = depthFirstSearch    # noqa
ucs = uniformCostSearch   # noqa
astar = aStarSearch       # noqa
