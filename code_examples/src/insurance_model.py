"""P&C Insurance Product Bundling Optimization Model.

Formulates the insurance coverage bundling problem as a 0-1 Integer Linear Program (ILP),
analogous to the automotive option-packaging problem in arXiv:2509.08328 Section 4.

The optimization decides which insurance coverages to bundle into packages and at what
discount, maximizing total contribution margin subject to business constraints.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pulp


@dataclass
class InsuranceCoverage:
    """A single insurance coverage option (e.g., auto collision, home fire).

    Attributes:
        name: Human-readable name.
        family: Coverage family (e.g., "auto", "home", "specialty").
        price: Standalone price (annual premium).
        take_rate: Fraction of customers who buy this coverage standalone.
        contribution_margin_pct: Profit margin as fraction of price.
        is_mandatory_in_family: If True, exactly one from this family must be in a package.
    """
    name: str
    family: str
    price: float
    take_rate: float
    contribution_margin_pct: float = 0.3
    is_mandatory_in_family: bool = False


@dataclass
class CompatibilityRule:
    """A rule about which coverages can or cannot be bundled together.

    Attributes:
        coverage_i: First coverage name.
        coverage_j: Second coverage name.
        compatible: If False, these two cannot appear in the same package.
    """
    coverage_i: str
    coverage_j: str
    compatible: bool = True


@dataclass
class DependencyRule:
    """A dependency: coverage_j requires coverage_i in the same package.

    If coverage_j is in a package, coverage_i must also be in that package.
    """
    requires: str  # coverage_i (prerequisite)
    dependent: str  # coverage_j (depends on requires)


@dataclass
class BundlingProblem:
    """Complete specification of a P&C insurance bundling optimization problem.

    This is the analog of the automotive option-packaging problem from the paper.
    Decision: which coverages to include in each package (binary x_{im}).

    Attributes:
        coverages: List of available coverage options.
        num_packages: Number of packages to create (M).
        max_options_per_package: Maximum coverages per package (O).
        discount_factor: Bundle discount as fraction of sum of individual prices (d).
        price_elasticity: Elasticity parameter epsilon (negative, typically -1 to -3).
        compatibility_rules: List of compatibility restrictions.
        dependency_rules: List of dependency requirements.
        co_take_rates: Dict mapping frozenset of coverage names to their co-purchase rate.
        reference_price_factor: Factor for kappa (reference price for non-buyers), default 2.0.
    """
    coverages: list[InsuranceCoverage]
    num_packages: int = 2
    max_options_per_package: int = 4
    discount_factor: float = 0.15
    price_elasticity: float = -2.0
    compatibility_rules: list[CompatibilityRule] = field(default_factory=list)
    dependency_rules: list[DependencyRule] = field(default_factory=list)
    co_take_rates: dict = field(default_factory=dict)
    reference_price_factor: float = 2.0
    package_discounts: list[float] | None = None
    segment_affinity: np.ndarray | None = None
    price_sensitivity_beta: float = 0.0
    package_names: list[str] | None = None

    @property
    def N(self) -> int:
        """Number of individual coverage options."""
        return len(self.coverages)

    @property
    def M(self) -> int:
        """Number of packages to create."""
        return self.num_packages

    @property
    def families(self) -> dict[str, list[int]]:
        """Map family name -> list of coverage indices."""
        fam = {}
        for i, cov in enumerate(self.coverages):
            fam.setdefault(cov.family, []).append(i)
        return fam

    @property
    def mandatory_families(self) -> dict[str, list[int]]:
        """Families where at least one coverage is marked mandatory."""
        result = {}
        for fam_name, indices in self.families.items():
            if any(self.coverages[i].is_mandatory_in_family for i in indices):
                result[fam_name] = indices
        return result

    @property
    def optional_families(self) -> dict[str, list[int]]:
        """Families where no coverage is marked mandatory."""
        mandatory = set(self.mandatory_families.keys())
        return {k: v for k, v in self.families.items() if k not in mandatory}

    def get_discount(self, m: int) -> float:
        """Get discount factor for package m.

        Uses per-package discount if ``package_discounts`` is set,
        otherwise falls back to the uniform ``discount_factor``.
        """
        if self.package_discounts is not None:
            return self.package_discounts[m]
        return self.discount_factor

    def get_affinity(self, i: int, m: int) -> float:
        """Get segment affinity for coverage i in package m.

        Returns the alpha_{i,m} multiplier from the LTM affinity matrix.
        Falls back to 1.0 (neutral) when ``segment_affinity`` is not set.
        """
        if self.segment_affinity is not None:
            return float(self.segment_affinity[i, m])
        return 1.0

    def coverage_index(self, name: str) -> int:
        """Get index of a coverage by name."""
        for i, cov in enumerate(self.coverages):
            if cov.name == name:
                return i
        raise ValueError(f"Coverage '{name}' not found")


def compute_package_take_rate(
    problem: BundlingProblem,
    package_indices: list[int],
    package_number: int,
) -> float:
    """Compute the take rate for a package given its constituent coverages.

    Follows the demand model from paper Section 4.2 (Eqs. 22-26).
    Customers migrate from buying subsets to buying the full package
    based on price elasticity of demand.

    Args:
        problem: The bundling problem specification.
        package_indices: Indices of coverages in this package.
        package_number: Which package number (for existing package avoidance).

    Returns:
        Expected take rate for the package.
    """
    if not package_indices:
        return 0.0

    d = problem.discount_factor
    eps = problem.price_elasticity
    coverages = problem.coverages

    individual_prices = [coverages[i].price for i in package_indices]
    package_price = (1 - d) * sum(individual_prices)

    total_migration = 0.0

    # For each non-empty subset of the package coverages, compute migration
    n = len(package_indices)
    for mask in range(1, 2**n):
        subset_indices = [package_indices[j] for j in range(n) if mask & (1 << j)]
        subset_key = frozenset(coverages[i].name for i in subset_indices)

        # Get the take rate for this exact subset
        if mask == 2**n - 1:
            # Full package subset - use co-take-rate if available
            if subset_key in problem.co_take_rates:
                base_rate = problem.co_take_rates[subset_key]
            else:
                base_rate = np.prod([coverages[i].take_rate for i in subset_indices])
        elif len(subset_indices) == 1:
            base_rate = coverages[subset_indices[0]].take_rate
        else:
            if subset_key in problem.co_take_rates:
                base_rate = problem.co_take_rates[subset_key]
            else:
                base_rate = np.prod([coverages[i].take_rate for i in subset_indices])

        # Price ratio: package_price / reference_price_for_subset
        subset_price = sum(coverages[i].price for i in subset_indices)
        if subset_price > 0:
            price_ratio = package_price / subset_price
            migration_factor = min(1.0, price_ratio ** eps)
            total_migration += base_rate * migration_factor

    # Also add migration from non-buyers
    kappa = problem.reference_price_factor * max(individual_prices)
    non_buyer_rate = max(0, 1 - sum(c.take_rate for c in coverages))
    if kappa > 0 and non_buyer_rate > 0:
        price_ratio = package_price / kappa
        migration_factor = min(1.0, price_ratio ** eps)
        total_migration += non_buyer_rate * migration_factor

    return min(1.0, total_migration)


def build_ilp(problem: BundlingProblem) -> tuple[pulp.LpProblem, dict]:
    """Build the 0-1 ILP for the insurance bundling problem.

    Decision variables: x[i][m] = 1 if coverage i is in package m.

    Objective: maximize total contribution margin (Eq. 21 analog).
    Constraints: Eqs. 27-32 analog.

    Args:
        problem: Complete bundling problem specification.

    Returns:
        Tuple of (PuLP problem, variable dict x[i][m]).
    """
    N = problem.N
    M = problem.M
    beta = problem.price_sensitivity_beta
    coverages = problem.coverages

    model = pulp.LpProblem("InsuranceBundling", pulp.LpMaximize)

    # Decision variables: x[i][m] = 1 if coverage i is in package m
    x = {}
    for i in range(N):
        for m in range(M):
            x[i, m] = pulp.LpVariable(f"x_{i}_{m}", cat=pulp.LpBinary)

    # -- Objective Function (LTM formula, Eq. 21 analog) --
    # c_{i,m} = price_i * margin_pct_i * (1 - delta_m)
    #           * (take_rate_i * alpha_{i,m} * (1 + beta * delta_m))
    #
    # Backward compatibility:
    #   delta_m = discount_factor when package_discounts is None
    #   alpha_{i,m} = 1.0 when segment_affinity is None
    #   beta = 0.0 by default  =>  (1 + 0 * delta_m) = 1
    #   => margin_pct * (1 - d) * price * take_rate * 1 * 1 = original formula
    objective_terms = []

    for m in range(M):
        delta_m = problem.get_discount(m)
        for i in range(N):
            cov = coverages[i]
            alpha_im = problem.get_affinity(i, m)
            margin = (
                cov.price
                * cov.contribution_margin_pct
                * (1 - delta_m)
                * cov.take_rate
                * alpha_im
                * (1 + beta * delta_m)
            )
            objective_terms.append(margin * x[i, m])

    model += pulp.lpSum(objective_terms), "TotalContributionMargin"

    # -- Constraints --

    # Eq. 27: Exactly one from mandatory families per package
    for fam_name, indices in problem.mandatory_families.items():
        for m in range(M):
            model += (
                pulp.lpSum(x[i, m] for i in indices) == 1,
                f"Mandatory_{fam_name}_pkg{m}"
            )

    # Eq. 28: At most one from optional families per package
    for fam_name, indices in problem.optional_families.items():
        for m in range(M):
            model += (
                pulp.lpSum(x[i, m] for i in indices) <= 1,
                f"Optional_{fam_name}_pkg{m}"
            )

    # Eq. 29: Maximum options per package
    for m in range(M):
        model += (
            pulp.lpSum(x[i, m] for i in range(N)) <= problem.max_options_per_package,
            f"MaxOptions_pkg{m}"
        )

    # Eq. 30: Compatibility restrictions (symmetric: i and j cannot both be in package m)
    for rule in problem.compatibility_rules:
        if not rule.compatible:
            i = problem.coverage_index(rule.coverage_i)
            j = problem.coverage_index(rule.coverage_j)
            for m in range(M):
                model += (
                    x[i, m] + x[j, m] <= 1,
                    f"Compat_{i}_{j}_pkg{m}"
                )

    # Eq. 31: Dependency requirements (j requires i)
    for rule in problem.dependency_rules:
        i = problem.coverage_index(rule.requires)
        j = problem.coverage_index(rule.dependent)
        for m in range(M):
            model += (
                x[j, m] <= x[i, m],
                f"Dep_{j}_needs_{i}_pkg{m}"
            )

    return model, x


def solve_ilp(problem: BundlingProblem) -> dict:
    """Solve the insurance bundling ILP and return results.

    Args:
        problem: Complete bundling problem specification.

    Returns:
        Dict with 'status', 'objective', 'packages' (list of coverage names per package),
        'solution_vector' (flat binary array for quantum comparison).
    """
    model, x = build_ilp(problem)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    N = problem.N
    M = problem.M

    packages = []
    for m in range(M):
        pkg_coverages = []
        for i in range(N):
            if pulp.value(x[i, m]) > 0.5:
                pkg_coverages.append(problem.coverages[i].name)
        packages.append(pkg_coverages)

    # Build flat solution vector [x_0_0, x_1_0, ..., x_N-1_0, x_0_1, ..., x_N-1_M-1]
    solution_vector = np.zeros(N * M, dtype=int)
    for m in range(M):
        for i in range(N):
            if pulp.value(x[i, m]) > 0.5:
                solution_vector[m * N + i] = 1

    return {
        "status": pulp.LpStatus[model.status],
        "objective": pulp.value(model.objective),
        "packages": packages,
        "solution_vector": solution_vector,
        "num_variables": N * M,
    }


def get_ilp_matrices(problem: BundlingProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the ILP in standard form: max c^T x, s.t. Ax <= b, x in {0,1}^n.

    Converts the PuLP model to explicit numpy matrices for quantum algorithm input.

    Args:
        problem: Complete bundling problem specification.

    Returns:
        Tuple of (c, A, b) where c is cost vector, A is constraint matrix, b is RHS.
    """
    model, x_vars = build_ilp(problem)
    N = problem.N
    M = problem.M
    n = N * M  # total binary variables

    # Variable ordering: x[i,m] -> index m*N + i
    var_index = {}
    for m in range(M):
        for i in range(N):
            var_index[f"x_{i}_{m}"] = m * N + i

    # Extract cost vector from objective
    c = np.zeros(n)
    for var, coeff in model.objective.items():
        idx = var_index.get(var.name)
        if idx is not None:
            c[idx] = coeff

    # Extract constraints
    constraints = list(model.constraints.values())
    num_constraints = len(constraints)
    A = np.zeros((num_constraints, n))
    b = np.zeros(num_constraints)

    for row, constraint in enumerate(constraints):
        # PuLP stores as: sum(a_j * x_j) + constant <= 0 (for <=)
        # or sum(a_j * x_j) + constant >= 0 (for >=)
        # or sum(a_j * x_j) + constant == 0 (for ==)
        for var, coeff in constraint.items():
            idx = var_index.get(var.name)
            if idx is not None:
                A[row, idx] = coeff
        b[row] = -constraint.constant

    return c, A, b


def load_ltm_instance(
    data_dir: str | Path,
    beta: float = 1.2,
) -> BundlingProblem:
    """Load a BundlingProblem from LTM CSV files.

    Reads the six CSV files produced by the LTM data generation pipeline
    and constructs a fully-populated ``BundlingProblem`` with per-package
    discounts, segment affinity matrix, and price sensitivity factor.

    Args:
        data_dir: Path to directory containing the six ``instance_*.csv`` files.
        beta: Price sensitivity factor (default 1.2 from LTM spec).

    Returns:
        A ``BundlingProblem`` with all LTM fields populated.
    """
    data_dir = Path(data_dir)

    # -- 1. Read coverages --
    coverages: list[InsuranceCoverage] = []
    with open(data_dir / "instance_coverages.csv", newline="") as f:
        for row in csv.DictReader(f):
            coverages.append(InsuranceCoverage(
                name=row["name"],
                family=row["family"],
                price=float(row["price_usd"]),
                take_rate=float(row["take_rate"]),
                contribution_margin_pct=float(row["margin_pct"]),
                is_mandatory_in_family=row["mandatory"].strip() == "True",
            ))

    # -- 2. Read packages (per-package discount, max_options, names) --
    package_names: list[str] = []
    package_discounts: list[float] = []
    max_options_list: list[int] = []
    with open(data_dir / "instance_packages.csv", newline="") as f:
        for row in csv.DictReader(f):
            package_names.append(row["name"])
            package_discounts.append(float(row["discount"]))
            max_options_list.append(int(row["max_options"]))

    num_packages = len(package_names)

    # Use the minimum max_options across packages as the global cap
    # (the ILP constraint applies a single value; individual packages
    # all share the same max in the current formulation).
    max_options_per_package = min(max_options_list)

    # -- 3. Read dependencies --
    dependency_rules: list[DependencyRule] = []
    with open(data_dir / "instance_dependencies.csv", newline="") as f:
        for row in csv.DictReader(f):
            dependency_rules.append(DependencyRule(
                requires=row["required_coverage_name"],
                dependent=row["dependent_coverage_name"],
            ))

    # -- 4. Read incompatible pairs --
    compatibility_rules: list[CompatibilityRule] = []
    with open(data_dir / "instance_incompatible_pairs.csv", newline="") as f:
        for row in csv.DictReader(f):
            compatibility_rules.append(CompatibilityRule(
                coverage_i=row["coverage_a_name"],
                coverage_j=row["coverage_b_name"],
                compatible=False,
            ))

    # -- 5. Read segment affinity matrix --
    coverage_name_to_idx = {cov.name: i for i, cov in enumerate(coverages)}
    N = len(coverages)
    M = num_packages
    affinity = np.ones((N, M), dtype=float)
    with open(data_dir / "instance_segment_affinity.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cov_name = row["coverage"]
            i = coverage_name_to_idx[cov_name]
            for m, pkg_name in enumerate(package_names):
                affinity[i, m] = float(row[pkg_name])

    return BundlingProblem(
        coverages=coverages,
        num_packages=num_packages,
        max_options_per_package=max_options_per_package,
        discount_factor=sum(package_discounts) / len(package_discounts),
        compatibility_rules=compatibility_rules,
        dependency_rules=dependency_rules,
        package_discounts=package_discounts,
        segment_affinity=affinity,
        price_sensitivity_beta=beta,
        package_names=package_names,
    )


def subsample_problem(
    problem: BundlingProblem,
    n_coverages: int,
    n_packages: int,
) -> BundlingProblem:
    """Create a smaller BundlingProblem from the first coverages and packages.

    Slices the coverage list, package_discounts, segment_affinity, and
    package_names.  Filters dependency_rules and compatibility_rules to
    only reference coverages that remain in the subsampled set.

    Args:
        problem: The full-sized BundlingProblem.
        n_coverages: Number of coverages to keep (first n).
        n_packages: Number of packages to keep (first n).

    Returns:
        A new BundlingProblem with reduced dimensions.
    """
    kept_coverages = problem.coverages[:n_coverages]
    kept_names = {cov.name for cov in kept_coverages}

    # Filter dependency rules: both ends must be in the kept set
    dep_rules = [
        r for r in problem.dependency_rules
        if r.requires in kept_names and r.dependent in kept_names
    ]

    # Filter compatibility rules: both ends must be in the kept set
    compat_rules = [
        r for r in problem.compatibility_rules
        if r.coverage_i in kept_names and r.coverage_j in kept_names
    ]

    # Slice per-package data
    pkg_discounts = (
        problem.package_discounts[:n_packages]
        if problem.package_discounts is not None
        else None
    )
    pkg_names = (
        problem.package_names[:n_packages]
        if problem.package_names is not None
        else None
    )
    affinity = (
        problem.segment_affinity[:n_coverages, :n_packages]
        if problem.segment_affinity is not None
        else None
    )

    return BundlingProblem(
        coverages=kept_coverages,
        num_packages=n_packages,
        max_options_per_package=problem.max_options_per_package,
        discount_factor=problem.discount_factor,
        price_elasticity=problem.price_elasticity,
        compatibility_rules=compat_rules,
        dependency_rules=dep_rules,
        co_take_rates=problem.co_take_rates,
        reference_price_factor=problem.reference_price_factor,
        package_discounts=pkg_discounts,
        segment_affinity=affinity,
        price_sensitivity_beta=problem.price_sensitivity_beta,
        package_names=pkg_names,
    )
