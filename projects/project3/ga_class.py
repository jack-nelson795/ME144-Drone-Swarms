"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems 
Project 3: Swarm
Spring 2026 Semester
GSI - Tommy Hosmer
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class GeneticAlgorithm(ABC):

    # These attributes are provided at runtime via the parameters dict.
    # Declare them explicitly to satisfy type checkers (Pylance).
    S: int
    P: int
    G: int
    K: int
    tolerance: float
    numLam: int

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

        # Assign commonly used GA hyperparameters before any use.
        # They will also be set again below via setattr for completeness.
        self.S = int(parameters.get("S", 0))
        self.P = int(parameters.get("P", 0))
        self.G = int(parameters.get("G", 0))
        self.K = int(parameters.get("K", 0))
        self.tolerance = float(parameters.get("tolerance", 0.0))
        self.numLam = int(parameters.get("numLam", 0))

        for key, value in parameters.items():
            setattr(self, key, value)

        self.strings = [np.array([])]*self.S
        self.costs_of_current_generation = [0]*self.S
        self.best_cost: List[float] = []
        self.average_cost: List[float] = []
        self.parents_average_cost: List[float] = []
        self.best_p_strings: List[np.ndarray] = []


    @abstractmethod
    def initialize_special_params(self):
        pass
    
    @abstractmethod
    def _generate_design_string(self):
        pass

    @abstractmethod
    def _evaluate_costs(self, start):
        pass


    def _sort_strings_and_costs(self):
        self.sorted_indices = np.argsort(self.costs_of_current_generation)
        self.strings = [self.strings[i] for i in self.sorted_indices]
        self.costs_of_current_generation = [self.costs_of_current_generation[i] for i in self.sorted_indices]

    def _mix_parents(self):
        rng = np.random.default_rng()  # create default generator

        # Generate K children from the current parent pool.
        # Each child is a convex combination of two parents with per-gene random mixing.
        for child_idx in range(self.K):
            p1, p2 = rng.choice(self.P, size=2, replace=False)
            alpha = rng.random(size=self.numLam)
            self.strings[self.P + child_idx] = alpha * self.strings[p1] + (1 - alpha) * self.strings[p2]

    def _add_immigrants(self):
        random_count = self.S - self.P - self.K
        for i in range(random_count):
            self.strings[i+self.P + self.K] = self._generate_design_string()

    def _store_data(self):
        self.average_cost.append(sum(self.costs_of_current_generation)/self.S)
        self.best_cost.append(self.costs_of_current_generation[0])
        self.parents_average_cost.append(sum(self.costs_of_current_generation[:self.P]) / self.P)
        self.best_p_strings = self.strings[:self.P]

    def GA(self):
        for i in range(self.S):
            self.strings[i] = self._generate_design_string()

        g = 1
        while g <= self.G:
            self._evaluate_costs(self.P)
            self._sort_strings_and_costs()
            self._store_data()
            if self.best_cost[-1] > self.tolerance: 
                break
            self._mix_parents()
            self._add_immigrants()
            g+=1
        
        data = {}
        data['best_cost'] = self.best_cost
        data['average_cost'] = self.average_cost
        data['parents_average_cost'] = self.parents_average_cost
        data['best_p_strings'] = self.best_p_strings
        return data
    
    @abstractmethod
    def specialGA(self):
        pass