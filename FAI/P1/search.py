# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import searchAgents
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
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

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def expand(problem: SearchProblem, node, final_state, frontier):
    """Given a search problem, a node and its final state (and frontier), expand the children and add to the frontier."""

    successors = problem.getSuccessors(final_state)
    for i in range(len(successors)):
        new_node = node + [successors[i]]
        frontier.push(new_node)

    return frontier ### to do: necessary???

def computeTotalCost(node):
    sum = 0

    for state in node[1:]:
        sum += state[2]

    return sum

def expandQueue(problem: SearchProblem, node, final_state, frontier, heuristic = None):
    """Given a search problem, a node and its final state (and frontier), expand the children and add to the frontier.
        Slight variation due to PriorityQueue requiring to save the priority (total cost of a path).
        Heuristic is a heuristic function, like Manhattan distance: see searchAgents.py"""

    successors = problem.getSuccessors(final_state)
    current_priority = computeTotalCost(node)

    for i in range(len(successors)):
        new_node = node + [successors[i]]
        new_priority = current_priority + successors[i][2]
        if heuristic is not None:
            # When we want to use a heuristic, compute it and then add it to the priority
            heuristic_value = heuristic(successors[i][0], problem)
            new_priority += heuristic_value

        frontier.push(new_node, new_priority)



    return frontier ### to do: necessary???

def compileActions(goal_node):
    """When we found a solution, in the goal node, compile the necessary actions to let Pacman win the game
        Should be called right before returning in the search algorithms"""
    actions = []

    for state in goal_node[1:]:
        actions.append(state[1])

    return actions

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Create empty frontier
    frontier = util.Stack()

    # Add start node to the frontier
    start_state = problem.getStartState()
    start_node = [start_state]
    frontier.push(start_node)

    # Initialize the 'reached' set
    reached = set()

    print("Set up done. Frontier is now", frontier.list)

    while not frontier.isEmpty():
        # Pop next node
        node = frontier.pop()

        # Get the final state of this node. Watch out: start node is NOT a triplet, separate case
        if node == start_node:
            final_state = start_state
        else:
            final_state = node[-1][0]

        # Check if the final state is a goal state
        if problem.isGoalState(final_state):
            ## Test:
            #print("Solution found, winning node: ", node)
            actions = compileActions(node)
            return actions

        # Expand children of the node and add final state to reached set
        if final_state not in reached:
            expand(problem, node, final_state, frontier)
            reached.add(final_state)

    print("No solution was found. Something is wrong?")
    return False



def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # Create empty frontier
    frontier = util.Queue()

    # Add start node to the frontier
    start_state = problem.getStartState()
    start_node = [start_state]
    frontier.push(start_node)

    # Initialize the 'reached' set
    reached = set()

    print("Set up done. Frontier is now", frontier.list)

    while not frontier.isEmpty():
        # Pop next node
        node = frontier.pop()

        # Get the final state of this node. Watch out: start node is NOT a triplet, separate case
        if node == start_node:
            final_state = start_state
        else:
            final_state = node[-1][0]

        # Check if the final state is a goal state
        if problem.isGoalState(final_state):
            ## Test:
            #print("Solution found, winning node: ", node)
            actions = compileActions(node)
            return actions

        # Expand children of the node and add final state to reached set
        if final_state not in reached:
            expand(problem, node, final_state, frontier)
            reached.add(final_state)

    print("No solution was found. Something is wrong?")
    return False

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Create empty frontier
    frontier = util.PriorityQueue()

    # Add start node to the frontier
    start_state = problem.getStartState()
    start_node = [start_state]
    frontier.push(start_node, 0)

    # Initialize the 'reached' set
    reached = set()

    #print("Set up done. Frontier is now", frontier.list)

    while not frontier.isEmpty():
        # Pop next node, save its priority
        node = frontier.pop()

        # Get the final state of this node. Watch out: start node is NOT a triplet, separate case
        if node == start_node:
            final_state = start_state
        else:
            final_state = node[-1][0]

        # Check if the final state is a goal state
        if problem.isGoalState(final_state):
            ### Testing purpose: print the solution when found
            #print("Solution found, winning node: ", node)
            actions = compileActions(node)
            return actions

        # Expand children of the node and add final state to reached set
        if final_state not in reached:
            expandQueue(problem, node, final_state, frontier)
            reached.add(final_state)

    print("No solution was found. Something is wrong?")
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Create empty frontier
    frontier = util.PriorityQueue()

    # Add start node to the frontier
    start_state = problem.getStartState()
    start_node = [start_state]
    frontier.push(start_node, 0)

    # Initialize the 'reached' set
    reached = set()

    #print("Set up done. Frontier is now", frontier.list)

    while not frontier.isEmpty():
        # Pop next node, save its priority
        node = frontier.pop()

        # Get the final state of this node. Watch out: start node is NOT a triplet, separate case
        if node == start_node:
            final_state = start_state
        else:
            final_state = node[-1][0]

        # Check if the final state is a goal state
        if problem.isGoalState(final_state):
            ## Test:
            #print("Solution found, winning node: ", node)
            actions = compileActions(node)
            return actions

        # Expand children of the node and add final state to reached set
        if final_state not in reached:
            expandQueue(problem, node, final_state, frontier, heuristic)
            reached.add(final_state)

    print("No solution was found. Something is wrong?")
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
