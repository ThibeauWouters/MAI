# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # print("Checking actions . . . ")
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information previous state
        previousGameScore = currentGameState.getScore()
        previousPos = currentGameState.getPacmanPosition()

        # Information about new state
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        successorGameScore = successorGameState.getScore()

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodPositions = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # If no food left: we win!!!
        if len(newFoodPositions) == 0:
            return 1000

        # Get approximate distance to food and ghosts
        food_distances = [manhattanDistance(newPos, foodPos) for foodPos in newFoodPositions]
        ghost_distances = [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPositions]

        # Base score: how the gamescore improves
        base_score = successorGameScore - previousGameScore

        # Get "best" values:
        closest_food_distance = min(food_distances)
        farthest_food_distance = max(food_distances)
        closest_ghost_distance = min(ghost_distances)
        total_food_distance = sum(food_distances)
        if closest_ghost_distance == 0:
            # Pacman dies
            return -1000

        # Check if we are in danger, i.e. a ghost closer than manhattan distance 3 and no pellet eaten:
        danger_perimeter = 3
        dangerous = False
        if closest_ghost_distance < danger_perimeter and 0 in newScaredTimes:
            dangerous = True

        # Make a score:
        if dangerous:
            # If we are at risk, play carefully - just get away from ghost
            score = - 1/closest_ghost_distance
        else:
            score = base_score + 1/closest_food_distance + 5*1/farthest_food_distance + 5*1/total_food_distance
            # Add penalty for standing still (in case we are safe)
            if newPos == previousPos:
                score += -1


        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getMinimaxValue(self, gameState, agent_index, current_depth):
        """Auxiliary function to compute the minimax value."""

        # BASE CASES
        # If terminal state, return the score
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        # If we exceeded max depth, estimate the utility with evaluationfunction
        elif current_depth == self.depth+1:
            return self.evaluationFunction(gameState)

        # RECURSIVE CASES
        # If not deep enough yet, then recursively get minimax values of next player
        else:
            new_actions = gameState.getLegalActions(agent_index)
            successors = [gameState.generateSuccessor(agent_index, action) for action in new_actions]

            next_agent_index = (agent_index + 1) % gameState.getNumAgents()
            # If again at zero: increase depth by one
            if next_agent_index == 0:
                next_depth = current_depth + 1
            else:
                next_depth = current_depth

            # Get values of the next one
            successor_values = []
            for successor in successors:
                new_value = self.getMinimaxValue(successor, next_agent_index, next_depth)
                successor_values.append(new_value)
            if agent_index == 0:
                return max(successor_values)
            else:
                return min(successor_values)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # We will store all minimax values in a list:
        minimax_values = []

        # Get minimax value for all legal actions in the game
        legal_actions = gameState.getLegalActions(self.index)
        for action in legal_actions:
            successor = gameState.generateSuccessor(self.index, action)
            next_index = (self.index + 1) % gameState.getNumAgents()
            minimax_values.append(self.getMinimaxValue(successor, next_index, 1))

        # If we are pacman (index = 0)
        if self.index == 0:
            # Do argmax
            best_minimax_value = max(minimax_values)
            best_index = minimax_values.index(best_minimax_value)
            best_action = legal_actions[best_index]
        # If we are a ghost:
        else:
            # Do argmin
            best_minimax_value = min(minimax_values)
            best_index = minimax_values.index(best_minimax_value)
            best_action = legal_actions[best_index]

        return best_action




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getMaxValue(self, gameState, agent_index, current_depth, alpha, beta):
        # Initialize v:
        v = -99999999

        # Get next agent
        next_agent_index = (agent_index + 1) % gameState.getNumAgents()

        # Update depth if necessary
        if next_agent_index == 0:
            next_depth = current_depth + 1
        else:
            next_depth = current_depth

        # Look at children, use alpha beta pruning
        new_actions = gameState.getLegalActions(agent_index)
        for action in new_actions:
            # Get child and recursively get its value
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = self.getMinimaxValue(successor, next_agent_index, next_depth, alpha, beta)
            # Update the value for v
            v = max(v, new_value)
            # Return (prune) if v is larger than smallest beta value (ghosts)
            if v > min(beta):
                break
            # Update Pacman's alpha value
            alpha = max(alpha, v)
        return v

    def getMinValue(self, gameState, agent_index, current_depth, alpha, beta):
        # Get max value:
        v = 99999999
        # Get next agent
        next_agent_index = (agent_index + 1) % gameState.getNumAgents()

        # Update depth if necessary
        if next_agent_index == 0:
            next_depth = current_depth + 1
        else:
            next_depth = current_depth

        # Look at children, use alpha beta pruning
        new_actions = gameState.getLegalActions(agent_index)
        for action in new_actions:
            # Get child and recursively determine its value
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = self.getMinimaxValue(successor, next_agent_index, next_depth, alpha, beta)
            # Update v
            v = min(v, new_value)
            # Return (prune) if v is smaller than alpha:
            if v < alpha:
                break
            # Update this agent's beta value
            beta[agent_index - 1] = min(beta[agent_index - 1], v)
        return v

    def getMinimaxValue(self, gameState, agent_index, current_depth, alpha, beta):

        # BASE CASES
        # If terminal state, return the score
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        # If we exceeded max depth, estimate the utility with evaluationfunction
        elif current_depth == self.depth+1:
            return self.evaluationFunction(gameState)

        # RECURSIVE CASES --- now with alpha beta pruning:
        # If not deep enough yet, then recursively get minimax values of children
        else:
            # Note: as we are using a list for the beta values, we have to make a copy, otherwise
            # we will overwrite the results in each recursive call.
            beta_copy = beta.copy()

            # In case we are Pacman (max player): call max value function
            if agent_index == 0:
                beta_copy = beta.copy()
                return self.getMaxValue(gameState, agent_index, current_depth, alpha, beta_copy)

            # In case we are NOT pacman (min player): call min value function
            else:

                return self.getMinValue(gameState, agent_index, current_depth, alpha, beta_copy)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # We will store all minimax values of children of root in a list:
        minimax_values = []
        # Get all possible actions, starting at the root
        legal_actions = gameState.getLegalActions(self.index)

        alpha = -99999999
        # As there can be more min players, use a list. Number of min players is number of agents - 1
        beta = [99999999] * (gameState.getNumAgents() - 1)

        # Note: for first player, need to explore all children, can only prune in the children
        next_index = (self.index + 1) % gameState.getNumAgents()
        for action in legal_actions:
            # Get child and compute its minimax value
            successor = gameState.generateSuccessor(self.index, action)
            new_value = self.getMinimaxValue(successor, next_index, 1, alpha, beta.copy())
            minimax_values.append(new_value)
            # As in the minimax function itself, update the alpha/beta passed onto the next branch
            if self.index == 0:
                alpha = max(alpha, new_value)
            else:
                beta[self.index - 1] = min(beta[self.index - 1], new_value)

        # From computed minimax values, get the best action
        # If we are pacman (index = 0)
        if self.index == 0:
            # Do argmax
            best_minimax_value = max(minimax_values)
            best_index = minimax_values.index(best_minimax_value)
            best_action = legal_actions[best_index]
        # If we are a ghost:
        else:
            # Do argmin
            best_minimax_value = min(minimax_values)
            best_index = minimax_values.index(best_minimax_value)
            best_action = legal_actions[best_index]

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def getMaxValue(self, gameState, agent_index, current_depth):
        # Initialize v:
        v = -99999999

        # Get next agent
        next_agent_index = (agent_index + 1) % gameState.getNumAgents()

        # Update depth if necessary
        if next_agent_index == 0:
            next_depth = current_depth + 1
        else:
            next_depth = current_depth

        # Look at children, use alpha beta pruning
        new_actions = gameState.getLegalActions(agent_index)
        for action in new_actions:
            # Get child and recursively get its value
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = self.getExpectimaxValue(successor, next_agent_index, next_depth)
            # Update the value for v
            v = max(v, new_value)
        return v

    def getExpValue(self, gameState, agent_index, current_depth):
        # Initialize v
        v = 0
        # Get next agent
        next_agent_index = (agent_index + 1) % gameState.getNumAgents()

        # Update depth if necessary
        if next_agent_index == 0:
            next_depth = current_depth + 1
        else:
            next_depth = current_depth

        # Look at children, use alpha beta pruning
        new_actions = gameState.getLegalActions(agent_index)
        p = 1/len(new_actions)

        for action in new_actions:
            # Get child and recursively determine its value
            successor = gameState.generateSuccessor(agent_index, action)
            new_value = self.getExpectimaxValue(successor, next_agent_index, next_depth)
            # Update v
            v += p*new_value
        return v

    def getExpectimaxValue(self, gameState, agent_index, current_depth):

        # BASE CASES
        # If terminal state, return the score
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()

        # If we exceeded max depth, estimate the utility with evaluationfunction
        elif current_depth == self.depth+1:
            return self.evaluationFunction(gameState)

        # RECURSIVE CASES --- now with alpha beta pruning:
        # If not deep enough yet, then recursively get minimax values of children
        else:

            # In case we are Pacman (max player): call max value function
            if agent_index == 0:
                return self.getMaxValue(gameState, agent_index, current_depth)

            # In case we are NOT pacman (min player): call min value function
            else:
                return self.getExpValue(gameState, agent_index, current_depth)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # We will store all minimax values of children of root in a list:
        expectimax_values = []
        # Get all possible actions, starting at the root
        legal_actions = gameState.getLegalActions(self.index)

        # Note: for first player, need to explore all children, can only prune in the children
        next_index = (self.index + 1) % gameState.getNumAgents()
        for action in legal_actions:
            # Get child and compute its minimax value
            successor = gameState.generateSuccessor(self.index, action)
            new_value = self.getExpectimaxValue(successor, next_index, 1)
            expectimax_values.append(new_value)

        # From computed minimax values, get the best action
        # If we are pacman (index = 0)
        if self.index == 0:
            # Do argmax
            best_minimax_value = max(expectimax_values)
            best_index = expectimax_values.index(best_minimax_value)
            best_action = legal_actions[best_index]
        # If we are a ghost:
        else:
            # Do argmin
            best_minimax_value = min(expectimax_values)
            best_index = expectimax_values.index(best_minimax_value)
            best_action = legal_actions[best_index]

        return best_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
