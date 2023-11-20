import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import math
import numpy as np
import scipy as sp
import copy
import itertools
import statistics
import time


# Reading the Product Space data #

g = nx.read_weighted_edgelist(
   r"C:\Users\Pau Vila\OneDrive - KU Leuven\Desktop\KU Leuven\Master Thesis\Reduced_example_adjacency_list.csv", delimiter=",",
   nodetype=int, create_using=nx.Graph(), )
weight = nx.get_edge_attributes(g, 'weight')

#print("Nodes set", g.nodes(g))
#print("Edge set:", g.edges(g))
#print("Weights", nx.get_edge_attributes(g, 'weight'))

active_nodes = []
all_nodes = set(g.nodes())
unactive_nodes = list(all_nodes - set(active_nodes))

#print("Active products", active_nodes)
#print("Unactive products", unactive_nodes)
#print("Total Number of Products", len(unactive_nodes) + len(active_nodes))


# Or creating a Scale-free network


#g = nx.barabasi_albert_graph(20, 2)
#mapping = {node: node + 1 for node in g.nodes()}
#g = nx.relabel_nodes(g, mapping)
#for u, v in g.edges():
 #   g[u][v]['weight'] = random.uniform(0, 1)

#pos = nx.random_layout(g)

#nx.draw_networkx_nodes(g, pos, node_size=1400, node_color='white', edgecolors='black')
#nx.draw_networkx_edges(g, pos, width=1, edge_color='black')

#plt.axis('off')
#plt.show()

#weight = nx.get_edge_attributes(g, 'weight')

#active_nodes = []
#all_nodes = set(g.nodes())
#unactive_nodes = list(all_nodes - set(active_nodes))


# Or a Random Graph


#n = 20
#p = 0.2
#g = nx.erdos_renyi_graph(n, p)

#mapping = {node: node + 1 for node in g.nodes()}
#g = nx.relabel_nodes(g, mapping)
#for u, v in g.edges():
 #   g[u][v]['weight'] = random.uniform(0.95, 1)

#pos = nx.random_layout(g)

#nx.draw_networkx_nodes(g, pos, node_size=1400, node_color='white', edgecolors='black')
#nx.draw_networkx_edges(g, pos, width=1, edge_color='black')

#plt.axis('off')
#plt.show()

#weight = nx.get_edge_attributes(g, 'weight')

#active_nodes = []
#all_nodes = set(g.nodes())
#unactive_nodes = list(all_nodes - set(active_nodes))


# Or a Small-world graph


#n = 20
#k=3
#p = 0.5
#g = nx.watts_strogatz_graph(n, k, p)

#mapping = {node: node + 1 for node in g.nodes()}
#g = nx.relabel_nodes(g, mapping)
#for u, v in g.edges():
 #   g[u][v]['weight'] = random.uniform(0.95, 1)

#pos = nx.random_layout(g)

#nx.draw_networkx_nodes(g, pos, node_size=1400, node_color='white', edgecolors='black')
#nx.draw_networkx_edges(g, pos, width=1, edge_color='black')

#plt.axis('off')
#plt.show()

#weight = nx.get_edge_attributes(g, 'weight')

#active_nodes = []
#all_nodes = set(g.nodes())
#unactive_nodes = list(all_nodes - set(active_nodes))



starting_time=time.time()

# Starting the MCTS #

class Node:

    def __init__(self, active_nodes, unactive_nodes, num_turns=0, hidden_parent=None, move=None):
        self.num_turns = num_turns
        self.active_nodes = active_nodes
        self.unactive_nodes = unactive_nodes
        self.number_visits = 0
        self.hidden_children = []
        self.total_reward = 0
        self.hidden_parent = hidden_parent
        self.move = move
        self.proportional_hidden_children = []


    def get_individual_probabilities(self, B=1, alpha=1.03):
        individual_probabilities = {}
        active_nodes = self.active_nodes
        unactive_nodes = self.unactive_nodes
        for node in unactive_nodes:
            sum_pi = 0
            sum_pi_a = 0
            degree = g.degree[node]
            for active_node in active_nodes:
                if g.has_edge(active_node, node):
                    sum_pi += g[active_node][node]['weight']
                    sum_pi_a += g[active_node][node]['weight']
            for unactive_node in unactive_nodes:
                if g.has_edge(unactive_node, node):
                    sum_pi += g[unactive_node][node]['weight']
            average_probability = sum_pi / degree
            bar_pi = average_probability
            p_i = bar_pi * B * ((sum_pi_a / sum_pi) ** alpha)
            individual_probabilities[node] = p_i
        return individual_probabilities

    def get_transition_probabilities(self):
        individual_probabilities = self.get_individual_probabilities()
        transition_probabilities = {}
        for child in self.hidden_children:
            if not child.originating_autonomous_nodes:
                transition_probabilities[child] = 1
            else:
                multiplying = 1
                for originating_node in child.originating_autonomous_nodes:
                     multiplying = multiplying * individual_probabilities[originating_node]
                     transition_probabilities[child] = multiplying
        return transition_probabilities

    def make_transition(self, node):
        transition_probabilities = self.get_transition_probabilities()
        sum_probabilities = sum(transition_probabilities.values())
        for hidden_child, prob in transition_probabilities.items():
            self.proportional_hidden_children.extend([hidden_child] * int(prob * 2))
        unique_children = len(set(self.proportional_hidden_children))
        transition = random.choice(self.proportional_hidden_children)
        return transition

    def node_is_terminal(self):
        if len(self.unactive_nodes) == 0:
            return True
        return False

    def average_reward(self):
        if self.number_visits == 0:
            return 0
        return self.total_reward / self.number_visits

    def add_hidden_child(self):
        for i in range(4):
            combinations = list(itertools.combinations(self.unactive_nodes, i))
            for combination in combinations:
               new_active_nodes = copy.deepcopy(self.active_nodes) + list(combination)
               new_unactive_nodes = copy.deepcopy(self.unactive_nodes)
               for node in combination:
                  new_unactive_nodes.remove(node)
               new_hidden_child = HiddenNode(new_active_nodes, new_unactive_nodes, node_parent=self)
               new_hidden_child.originating_autonomous_nodes = combination
               self.hidden_children.append(new_hidden_child)


class HiddenNode:
    def __init__(self, active_nodes, unactive_nodes, node_parent=Node, move=None):
        self.active_nodes = active_nodes
        self.unactive_nodes = unactive_nodes
        self.node_children = []
        self.node_parent = node_parent
        self.move = move
        self.originating_autonomous_nodes = []
        self.untried_actions = []
        self.untried_actions = self.unactive_nodes.copy()



    def possible_moves(self):
        return self.unactive_nodes

    def make_move(self, move):
        self.active_nodes.append(move)
        self.unactive_nodes = [node for node in self.unactive_nodes if node != move]

    def add_Node_child(self, move):
        new_active_nodes = copy.deepcopy(self.active_nodes)
        new_unactive_nodes = copy.deepcopy(self.unactive_nodes)
        new_active_nodes.append(move)
        new_unactive_nodes.remove(move)  # remove the move from the unactive_nodes list of the parent node
        self.untried_actions.remove(move)
        child = Node(active_nodes=new_active_nodes, unactive_nodes=new_unactive_nodes,
                     num_turns=self.node_parent.num_turns + 1, move=move)
        child.hidden_parent = self
        self.node_children.append(child)

    def hidden_node_is_terminal(self):
        if len(self.unactive_nodes) == 0:
            return True
        return False


class MCTSTreeSearch:
    def __init__(self, root_state, c=2):
        self.root = Node(active_nodes=root_state[0], unactive_nodes=root_state[1])
        self.root.number_visits = 1
        self.c = c
        self.n_iteration = 0

    def search(self):
        print("!!!!!!!!    NEW ITERATION    !!!!!!!")
        node = self.root
        if self.is_leaf(node):
            self.expand_normal_node(node)
        hidden_node = node.make_transition(node)
        while self.hidden_node_is_fully_expanded(hidden_node):
            node = self.select_child(hidden_node)
            if self.is_leaf(node):
                self.expand_normal_node(node)
            hidden_node = node.make_transition(node)
        self.expand_function(hidden_node)
        if len(hidden_node.node_children) > 0:
            node = self.select_child(hidden_node)
        print("<<<< SIMULATION BEGINS >>>>")
        reward, results = self.simulate_function(node)  # This maybe should be inside the loop
        reward_results = reward, results
        self.backpropagate(node, reward, root=self.root)
        self.n_iteration += 1
        node = self.root
        self.root.number_visits += 1
        return reward, results

    def expand_function(self, hidden_node):
        if len(hidden_node.untried_actions) > 0:
            move = random.choice(hidden_node.untried_actions)
            hidden_node.add_Node_child(move)

    def hidden_node_is_fully_expanded(self, hidden_node):
        if len(hidden_node.untried_actions) == 0:
           return True
        False

    def is_leaf(self, node):
        return len(node.hidden_children) == 0

    def hidden_node_is_leaf(self, hidden_node):
        return len(hidden_node.node_children) == 0

    def select_child(self, hidden_node):
        min_score = float("inf")
        min_child = None
        unvisited_children = [child for child in hidden_node.node_children if child.number_visits == 0]
        if unvisited_children:
            chosen_unexplored_child = random.choice(unvisited_children)
            return chosen_unexplored_child
        else:
            for child in hidden_node.node_children:
                score = child.total_reward / child.number_visits - self.c * math.sqrt(
                    math.log(hidden_node.node_parent.number_visits) / child.number_visits)
                if score < min_score:
                    min_score = score
                    min_children = []
                    min_children = [child]
                    if score == min_score:
                        min_children.append(child)
            if len(min_children) == 0:
            selected_child = random.choice(min_children)
            return selected_child


    def expand_normal_node(self, node):
        node.add_hidden_child()

    def select_random_child(self, hidden_node):
        children = hidden_node.node_children
        return random.choice(children)


    def simulate_function(self, node):
        random.seed()
        reward = node.num_turns
        results = []
        hidden_node = node.hidden_parent
        while len(node.unactive_nodes) > 0 and len(hidden_node.unactive_nodes) > 0:
            if len(node.hidden_children) == 0:
                self.expand_normal_node(node)
            hidden_node = node.make_transition(node)
            if len(hidden_node.unactive_nodes) == 0:
                break
            if len(hidden_node.node_children) == 0:
                move = random.choice(hidden_node.possible_moves())
                hidden_node.add_Node_child(move)
            node = self.select_random_child(hidden_node)  # THERE IS A PPROBLEM HERE!!!!! It will always simulate the same moves (once it has created the first child)
        reward = node.num_turns
        results.append((reward, self.n_iteration))  # This maybe should be out of the loop
        return reward, results

    def backpropagate(self, node, reward, root):
        root = self.root
        while node is not root:
            hidden_parent = node.hidden_parent
            parent = hidden_parent.node_parent
            node.total_reward += reward
            node.number_visits += 1
            node = parent


#number_iterations = 200

initial_network_state = (active_nodes, unactive_nodes)

number_trials = 1
number_iterations = 200

results_list = []
slopes = []
intercepts = []

for t in range(number_trials):
    mcts = MCTSTreeSearch(root_state=initial_network_state)

    trial_results = []
    for i in range(number_iterations):
        reward, results = mcts.search()
        trial_results.append((i, reward))

    results_list.append(trial_results)

    x, y = zip(*trial_results)
    x = np.array(x)
    y = np.array(y)
    slope, intercept = np.polyfit(x, y, 1)
    slopes.append(slope)
    intercepts.append(intercept)


fig, ax = plt.subplots()
for t in range(number_trials):
    trial_results = results_list[t]
    x, y = zip(*trial_results)
    ax.scatter(x, y, color='gray', alpha=0.0125)
    ax.plot(x, np.multiply(slopes[t], x) + intercepts[t], color='gray', alpha=0.7)


combined_slope = np.mean(slopes)
combined_intercept = np.mean(intercepts)
x_min = np.min([x for trial in results_list for x, _ in trial])
x_max = np.max([x for trial in results_list for x, _ in trial])
x_range = np.linspace(x_min, x_max, num=100)
ax.plot(x_range, combined_slope * x_range + combined_intercept, color='black',label='Combined Regression', linewidth=4)


#ax.set_title("Simulation Results ({} Trials)".format(number_trials))
ax.set_xlabel("Iteration",fontsize=14 ,fontweight='bold')
ax.set_ylabel("Time to activate the network",fontsize=14 ,fontweight='bold')
ax.grid(True, linewidth=0.5, color='gray', linestyle='--')

ax.legend(loc='upper right')


plt.show()

finish_time=time.time()

Total_time = finish_time - starting_time

print(f"Time to run the algorithm: {Total_time}")
