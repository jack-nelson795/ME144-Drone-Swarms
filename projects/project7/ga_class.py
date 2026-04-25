"""
ME 144/244: Modeling, Simulation, and Digital Twins of Drone-Based Systems 
Project 7: Aerial Firefighting
Spring 2026 Semester

"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List

class GeneticAlgorithm(ABC):

    def __init__(self, parameters):
        self.parameters = parameters
        for key, value in parameters.items():
            setattr(self, key, value)

        # Make dynamically-loaded parameters explicit for static analyzers.
        self.K: int = int(parameters["K"])
        self.P: int = int(parameters["P"])
        self.S: int = int(parameters["S"])
        self.G: int = int(parameters["G"])
        self.NUMLAM: int = int(parameters["NUMLAM"])
        self.TOLERANCE: float = float(parameters["TOLERANCE"])

        # These are initialized by child classes in initialize_special_params().
        self.numLam: int = self.NUMLAM
        self.tolerance: float = self.TOLERANCE

        self.seed = getattr(self, 'RANDOM_SEED', 144)
        self.rng = np.random.default_rng(self.seed)

        self.strings: List[np.ndarray] = [np.array([], dtype=float) for _ in range(self.S)]
        self.costs_of_current_generation: List[float] = [0.0 for _ in range(self.S)]
        self.best_cost: List[float] = []
        self.average_cost: List[float] = []
        self.parents_average_cost: List[float] = []
        self.best_p_strings: List[np.ndarray] = []


    @abstractmethod
    def initialize_special_params(self) -> None:
        pass
    
    @abstractmethod
    def _generate_design_string(self) -> np.ndarray:
        pass

    @abstractmethod
    def _evaluate_costs(self, start: int) -> None:
        pass


    def _sort_strings_and_costs(self) -> None:
        self.sorted_indices = np.argsort(self.costs_of_current_generation)
        self.strings = [self.strings[i] for i in self.sorted_indices]
        self.costs_of_current_generation = [self.costs_of_current_generation[i] for i in self.sorted_indices]

    def _mix_parents(self) -> None:
        for i in range(0, self.P, 2):
            phi = self.rng.random(size=self.numLam)
            psi = self.rng.random(size=self.numLam)
            self.strings[self.P+i] = phi*self.strings[i] + (1-phi)*self.strings[i+1] 
            self.strings[self.P+i+1] = psi*self.strings[i] + (1-psi)*self.strings[i+1]

    def _add_immigrants(self) -> None:
        random_count = self.S - self.P - self.K
        for i in range(random_count):
            self.strings[i+self.P + self.K] = self._generate_design_string()

    def _store_data(self) -> None:
        self.average_cost.append(sum(self.costs_of_current_generation)/self.S)
        self.best_cost.append(self.costs_of_current_generation[0])
        self.parents_average_cost.append(sum(self.costs_of_current_generation[:self.P]) / self.P)
        self.best_p_strings = [np.array(string, dtype=float, copy=True) for string in self.strings[:self.P]]

    def GA(self, print_bool: bool = False) -> Dict[str, Any]:
        for i in range(self.S):
            self.strings[i] = self._generate_design_string()

        # Generation 1: evaluate every design string.
        self._evaluate_costs(0)
        self._sort_strings_and_costs()
        self._store_data()
        if self.best_cost[-1] < self.tolerance:
            data = {}
            data['best_cost'] = self.best_cost
            data['average_cost'] = self.average_cost
            data['parents_average_cost'] = self.parents_average_cost
            data['best_p_strings'] = self.best_p_strings
            return data

        # Later generations: preserve parent costs, evaluate only new strings.
        g = 2
        while g <= self.G:
            if print_bool:
                print(f'Generation : {g}')
                print(f'Best Cost  : {self.best_cost[-1]}')
            self._mix_parents()
            self._add_immigrants()
            self._evaluate_costs(self.P)
            self._sort_strings_and_costs()
            self._store_data()
            if self.best_cost[-1] < self.tolerance:
                break
            g+=1
        
        data = {}
        data['best_cost'] = self.best_cost
        data['average_cost'] = self.average_cost
        data['parents_average_cost'] = self.parents_average_cost
        data['best_p_strings'] = self.best_p_strings
        return data
    
    @abstractmethod
    def specialGA(self) -> Any:
        pass
