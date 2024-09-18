import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class QuantumEconConfig:
    start_year: int
    roadmap: Dict[int, int]
    hardware_slowdown: float
    quantum_improvement_rate: float
    physical_to_logical_ratio: float
    physical_to_logical_improvement: float
    hardware_slowdown: float
    extrapolation_type: str
    logical_to_problem_size: str = "2**q"
    penalty: str = "log(n, 2)"
    prob_size_intersect_range: Tuple[float, float] = (2, 1e10)


class QuantumEconAnalysis:
    def __init__(self, config: QuantumEconConfig):
        self.config = config
        self.start_year = config.start_year
        self.roadmap = config.roadmap
        self.hardware_slowdown = config.hardware_slowdown
        self.quantum_improvement_rate = config.quantum_improvement_rate
        self.physical_to_logical_ratio = config.physical_to_logical_ratio
        self.physical_to_logical_improvement = config.physical_to_logical_improvement
        self.extrapolation_type = config.extrapolation_type
        self.logical_to_problem_size = config.logical_to_problem_size
        self.penalty = config.penalty
        self.prob_size_intersect_range = config.prob_size_intersect_range

    def get_feasibility(self, year):
        return self.get_problem_size(year)

    def plot_feasible_problem(self, start_year=2024, stop_year=2027):
        years = np.arange(start_year, stop_year, step=0.01)
        plt.plot(
            years,
            [self.get_problem_size(year) for year in years],
            label="Problem Size",
        )
        plt.xlabel("Year")
        plt.yscale("log")
        plt.ylabel("Problem Size")
        plt.title("Quantum Econonmic Advantage")
        plt.grid(True)

    def plot_quantum_econ_advantage(
        self, classical_runtime, quantum_runtime, start_year=2024, stop_year=2027
    ):
        classical_runtime_func = self.create_runtime_function(classical_runtime)
        quantum_runtime_func = self.create_runtime_function(
            quantum_runtime, quantum=True
        )
        years = np.arange(start_year, stop_year, step=0.001)
        plt.plot(
            years,
            [
                self.find_n_star_by_year(
                    classical_runtime_func, quantum_runtime_func, year
                )
                for year in years
            ],
            label="Economic Advantage",
        )
        plt.xlabel("Year")
        plt.ylabel("Problem Size")
        plt.title("Problem Size Over Time")
        plt.yscale("log")
        plt.grid(True)

    def create_runtime_function(self, runtime_expr="n", quantum=False) -> callable:
        x, n = sp.symbols("x n")
        if quantum == True:
            expr = (
                sp.sympify(runtime_expr).subs(n, x)
                * self.hardware_slowdown
                * sp.sympify(self.penalty).subs(n, x)
            )
        else:
            expr = sp.sympify(runtime_expr).subs(n, x)
        # expr = sp.log(expr)
        return sp.lambdify(x, expr, "numpy")

    def binary_search_intersection(
        self, get_value1, get_value2, low=2024, high=2200, epsilon=0.001, log=True
    ):

        if log:
            log_low, log_high = np.log10(low), np.log10(high)

            while log_high - log_low > epsilon:
                log_mid = (log_low + log_high) / 2
                mid = 10**log_mid

                value1 = np.log10(get_value1(mid))
                value2 = np.log10(get_value2(mid))

                if value1 > value2:
                    log_high = log_mid
                else:
                    log_low = log_mid

            return 10 ** ((log_low + log_high) / 2)
        else:
            while high - low > epsilon:
                mid = (low + high) / 2
                if get_value1(mid) > get_value2(mid):
                    high = mid
                else:
                    low = mid
            return mid

    def econ_advantage_point(self, classical_runtime: str, quantum_runtime: str):
        classical_runtime_func = self.create_runtime_function(classical_runtime)
        quantum_runtime_func = self.create_runtime_function(
            quantum_runtime, quantum=True
        )
        estimated_year = self.find_n_star_star(
            classical_runtime_func, quantum_runtime_func
        )
        return estimated_year

    def find_prob_size_intersection(
        self, classical_runtime: str, quantum_runtime: str, problem_size=1e10
    ) -> float:
        classical_runtime_func = self.create_runtime_function(classical_runtime)
        quantum_runtime_func = self.create_runtime_function(
            quantum_runtime, quantum=True
        )

        def get_economic_advantage(year):
            return self.find_n_star_by_year(
                classical_runtime_func, quantum_runtime_func, year
            )

        def get_feasibility(year):
            return self.get_problem_size(year)

        economic_advantage_year = self.binary_search_intersection(
            lambda year: problem_size,
            get_economic_advantage,
            low=2024,
            high=2200,
            log=False,
        )
        print("economic_advantage_year", economic_advantage_year)
        feasibility_year = self.binary_search_intersection(
            get_feasibility,
            lambda year: problem_size,
            low=2024,
            high=2200,
            log=False,
        )
        # print("feasibility_year", feasibility_year)
        return max(economic_advantage_year, feasibility_year)

    # find n_star or the point where quantum computing surpasses classical computing for the given algorithm
    def find_n_star(
        self, classical_runtime_func: callable, quantum_runtime_func: callable
    ) -> float:
        # used to be binaryserach logartihmci intersection
        return self.binary_search_intersection(
            classical_runtime_func,
            quantum_runtime_func,
            low=self.prob_size_intersect_range[0],
            high=self.prob_size_intersect_range[1],
        )

    def find_n_star_star(
        self, classical_runtime_func: callable, quantum_runtime_func: callable
    ) -> float:
        def get_feasibility(year):
            return self.get_problem_size(year)

        def get_advantage(year):
            return self.find_n_star_by_year(
                classical_runtime_func, quantum_runtime_func, year
            )

        return self.binary_search_classical_intersection(
            get_feasibility, get_advantage, low=2024, high=2100, epsilon=0.01
        )

    def get_qubits(self, year) -> int:
        years = sorted(self.roadmap.keys())

        if year in self.roadmap:
            return self.roadmap[year]

        def extrapolate(x, x_values, y_values, exponential=False):
            if exponential:
                regression = np.polyfit(x_values, np.log10(y_values), 1)
                return 10 ** np.polyval(regression, x)
            else:
                regression = np.polyfit(x_values, y_values, 1)
                return np.polyval(regression, x)

        if year > years[-1]:
            return extrapolate(
                year,
                years[-2:],
                [self.roadmap[yr] for yr in years[-2:]],
                self.extrapolation_type == "exponential",
            )

        if year > years[0]:
            left_year, right_year = max(yr for yr in years if yr <= year), min(
                yr for yr in years if yr >= year
            )
            return extrapolate(
                year,
                [left_year, right_year],
                [self.roadmap[left_year], self.roadmap[right_year]],
                self.extrapolation_type == "exponential",
            )

        return 0

    def get_logical_qubits(self, year) -> int:
        physical_qubits = self.get_qubits(year)
        scaling_factor = self.physical_to_logical_ratio * (
            1 - self.physical_to_logical_improvement
        ) ** (year - 2024)
        return physical_qubits / scaling_factor

    def get_problem_size(self, year) -> int:
        logical_qubits = self.get_logical_qubits(year)
        problem_size_func = sp.sympify(self.logical_to_problem_size)
        return problem_size_func.subs({"q": logical_qubits})

    def find_n_star_by_year(self, classical_runtime_func, quantum_runtime_func, year):
        quantum_runtime_func_aux = lambda x: quantum_runtime_func(x) * (
            1 - self.quantum_improvement_rate
        ) ** (year - 2024)
        return self.find_n_star(classical_runtime_func, quantum_runtime_func_aux)
