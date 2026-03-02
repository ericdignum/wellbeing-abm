import numpy as np
import networkx as nx
from scipy import sparse
from scipy.stats import genextreme, t
from scipy.spatial.distance import cdist
from sda_functions import SDA, rewire_edges


class WellbeingSim:
    """
    Agent-based simulation model of well-being dynamics in a networked population.
    
    This model simulates how individual well-being evolves over time based on:
    - Personal capital accumulation and shocks
    - Social comparison with network neighbors
    - Adaptation/habituation to past conditions
    - Network structure and homophily
    - Feedback loops between well-being and capital growth
    
    The simulation tracks well-being, capital, and social networks across multiple
    time steps, with configurable parameters for network topology, comparison
    mechanisms, adaptation rates, and external shocks.
    """

    def __init__(self, params):
        """
        Initialise the well-being simulation with specified parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing simulation parameters with the following keys:
            
            Simulation settings:
            - N (int): Number of agents in the simulation
            - n_steps (int): Number of simulation time steps
            - seed (int, optional): Random seed for reproducibility
            
            Network configuration:
            - network_type (str): Network topology ('sda', 'random', 'preferential')
            - homophily (float): Homophily parameter for network formation
                                 Negative values create special network types:
                                 -1: random, -2/-3/-4: preferential with sorting
            - avg_degree (int): Average number of connections per node
            - p_rewire (float): Probability of rewiring an edge per time step
            - wb_in_dist (bool): Include well-being in distance calculations
            
            Social comparison:
            - soc_comparison (bool): Enable/disable social comparison
            - quantile (float): Quantile for comparison (0-1) if using quantile method
            - soc_comp_w (float or np.array): Weight of social comparison (0-1)
            
            Adaptation/habituation:
            - alpha (float or np.array): Adaptation rate to capital changes (0-1)
            - soc_to_wb (bool or float): Social capital benefit to well-being
            
            Capital dynamics:
            - capital_dist (str): Initial capital distribution 
                                  ('lognormal', 'beta', 'homogeneous')
            - mu (float): First parameter of capital distribution
            - sigma (float): Second parameter of capital distribution
            - growth_rate (float or np.array): Deterministic capital growth rate
            - wb_to_cap (float or np.array): Well-being feedback to capital growth
            - shock_sigma (float): Variance of stochastic capital shocks
            - perception (str): Well-being perception function ('concave' or 'sigmoid')
            
            Shock/intervention settings:
            - event_type (str): Type of events ('fixed', 'random', 'random+fixed')
            - p (float): Probability of receiving intervention shock
            - int_freq (int): Frequency of intervention events (time steps)
            - int_size (float): Size of intervention shock (absolute)
            - rel_size (float): Size of intervention shock (relative to capital)
        """
        # Store all parameters as instance attributes
        self.params = params
        for key, value in params.items():
            setattr(self, key, value)
        
        # Initialise random number generator with seed if provided
        try:
            self.rng = np.random.default_rng(seed=self.seed)
        except AttributeError:
            self.rng = np.random.default_rng()

        # Initialise agent capital levels
        self.capital = self.init_capital(
            dist=self.capital_dist, 
            mu=self.mu, 
            sigma=self.sigma
        )
        
        # Adaptation level starts equal to initial capital
        self.al_capital = self.capital.copy()
        
        # Social comparison values start equal to capital
        self.soc_comp_values = self.capital.copy()

        # Initialise well-being, network structure, and shock events
        self.init_wellbeing()
        self.init_network(network_type=self.network_type)
        self.init_events()

    def init_capital(self, dist='lognormal', mu=10, sigma=0.1):
        """
        Initialise the capital levels for all agents.
        
        Capital represents the economic or resource state of each agent and serves
        as a primary driver of well-being in the model.
        
        Parameters
        ----------
        dist : str, default='lognormal'
            Distribution type for initial capital values
        mu : float, default=10
            First parameter of the distribution (mean for lognormal, alpha for beta)
        sigma : float, default=0.1
            Second parameter of the distribution (std for lognormal, beta for beta)
        
        Returns
        -------
        capital : np.ndarray, shape (N, n_steps+1)
            Array to store capital values over time, initialised at step 0 and 1
        """
        capital = np.zeros((self.N, self.n_steps + 1))
        
        if dist == 'lognormal': 
            capital[:, 0] = self.rng.lognormal(mean=mu, sigma=sigma, size=self.N)
        elif dist == 'beta':
            capital[:, 0] = self.rng.beta(a=self.mu, b=self.sigma, size=self.N)
        elif dist == 'homogeneous':
            capital[:, 0] = 0.5
        else:
            raise ValueError(f'Distribution "{dist}" not implemented!')

        # Copy initial capital to step 1
        capital[:, 1] = capital[:, 0]
        return capital
    
    def update_capital(self, step, history, wellbeing, wb_to_cap, random_shocks):
        """
        Update agent capital levels for the current time step.
        
        Capital evolves based on:
        - Deterministic growth rate
        - Random shocks (noise)
        - Well-being feedback (higher well-being can boost capital growth)
        - Intervention shocks at specified intervals
        
        Parameters
        ----------
        step : int
            Current simulation time step
        history : np.ndarray
            Historical capital values
        well-being : np.ndarray
            Historical well-being values
        wb_to_cap : float or np.array
            Strength of well-being feedback to capital
        random_shocks : np.ndarray
            Pre-generated random shocks for all time steps
        
        Returns
        -------
        new_capital : np.ndarray, shape (N,)
            Updated capital values for current step
        """
        old_capital = history[:, step-1]
        
        # Calculate well-being change from previous step
        wb_diff = wellbeing[:, step-1] - wellbeing[:, step-2]
        
        # Growth rate includes base growth plus random shock
        growth_rate = 1 + random_shocks[:, step] + self.growth_rate
        
        # Well-being feedback: positive well-being change boosts capital growth
        # Sigmoid transformation keeps multiplier between 0 and 2
        multiplier = 2 / (1 + np.exp(-wb_to_cap * wb_diff))
        
        # Apply growth and well-being feedback
        new_capital = growth_rate * old_capital * multiplier
        
        # Apply intervention shock at specified frequency
        if self.step == self.int_freq:
            if self.N > 1:
                # Only apply to randomly selected subset of agents
                new_capital = np.where(
                    self.random_idx == 1, 
                    new_capital * (1 - self.rel_size), 
                    new_capital
                )
            else:
                # Apply to single agent
                new_capital *= (1 - self.rel_size)
        
        return new_capital
        
    def calc_dists(self, step):
        """
        Calculate Euclidean distances between all pairs of agents.
        
        Distance can be based on capital alone or include well-being depending
        on the wb_in_dist parameter. These distances are used for homophilic
        network formation and rewiring.
        
        Parameters
        ----------
        step : int
            Current time step for which to calculate distances
        
        Returns
        -------
        dists : np.ndarray, shape (N, N)
            Pairwise distance matrix between all agents
        """
        if self.wb_in_dist:
            # Use both capital and well-being as dimensions for distance
            points = np.vstack((
                self.capital[:, step], 
                self.wellbeing[:, step-1]
            )).T
        else:
            # Use only capital for distance calculation
            points = self.capital[:, step][:, np.newaxis]
        
        # Compute all pairwise Euclidean distances
        dists = cdist(points, points, metric='euclidean')
        self.dists = dists
        return dists

    def init_network(self, network_type='random'):
        """
        Initialise the social network connecting agents.
        
        The network determines which agents can influence each other through
        social comparison. Multiple network types are supported with different
        structural properties.
        
        Parameters
        ----------
        network_type : str, default='random'
            Type of network topology:
            - 'random': Random graph (h=-1)
            - 'preferential': Barabasi-Albert preferential attachment 
                (h=-2: Normal BA network (unrelated to capital), 
                 h=-3: Low capital agents have high degree,
                 h=-4: High capital agents have high degree)
            - 'sda': Social distance attachment model (homophily-based)
                0<h<0.2 gives issues with the optimisation, therefore use h=-1
                for fully random.
        """
        N = self.N
       
        if network_type == 'random':
            # Random graph with specified average degree
            network = nx.fast_gnp_random_graph(N, self.avg_degree/N, directed=False)
            
        elif network_type == 'preferential':
            # Scale-free network via preferential attachment
            network = nx.barabasi_albert_graph(n=self.N, m=self.avg_degree)

        elif network_type == 'sda':
            # Special cases using homophily parameter
            if self.homophily < 0:
                
                # Preferential attachment with capital-degree correlation
                if self.homophily <= -2:
                    network = nx.barabasi_albert_graph(
                        n=self.N, 
                        m=int(self.avg_degree / 2)
                    )
                    
                    # Sort agents by degree and assign capital accordingly
                    dgrs = network.degree()
                    dgrs = np.array([dgrs[i] for i in range(N)])
                    srtd_econ = np.sort(self.capital[:, 0])
                    
                    # Positive or negative correlation between degree and capital
                    if self.homophily == -3:
                        dgrs_idx = dgrs.argsort()  # Poor agents have low degree
                    elif self.homophily == -4:
                        dgrs_idx = dgrs.argsort()[::-1]  # Poor agents have high degree
                    
                    # Reassign capital based on degree ranking
                    self.capital[dgrs_idx, 0] = srtd_econ
                    self.capital[dgrs_idx, 1] = srtd_econ
                    self.al_capital[dgrs_idx, 0] = srtd_econ
                    self.al_capital[dgrs_idx, 1] = srtd_econ
                    
                    # Recalculate distances after reassignment
                    dists = self.calc_dists(step=1)
                else:
                    # Standard random network
                    network = nx.fast_gnp_random_graph(
                        N, 
                        self.avg_degree / self.N, 
                        directed=False
                    )
                
            else:
                # True SDA network with homophily
                dists = self.calc_dists(step=0)
                sda = SDA.from_dist_matrix(
                    D=dists, 
                    k=self.avg_degree, 
                    alpha=self.homophily, 
                    p_rewire=0.01, 
                    directed=False
                )
                network = sda.adjacency_matrix(sparse=False)
                network = nx.from_numpy_array(network)

        else:
            raise NotImplementedError(f"Network type '{network_type}' not implemented")

        # Store as sparse matrix for memory efficiency
        self.network = nx.to_scipy_sparse_array(network, dtype=bool)
        
        # Social capital initially zero (updated if soc_to_wb enabled)
        self.soc_cap = 0

    def update_network(self, step):
        """
        Update the social network through edge rewiring.
        
        Networks can be rewired randomly or according to homophily principles,
        where agents preferentially connect to similar others (based on capital
        and optionally well-being).
        
        Parameters
        ----------
        step : int
            Current simulation time step
        
        Returns
        -------
        new_network : np.ndarray, shape (N, N)
            Updated network adjacency matrix
        """
        # Start with previous network configuration
        new_network = self.network[:, :, step-1].copy()
        
        # Calculate current distances if needed for rewiring or social capital
        if self.soc_to_wb > 0 or self.do_network_update:
            dists = self.calc_dists(step=step)
            
        if self.do_network_update:
            
            if self.network_type == 'random':
                # Random rewiring: swap edges with probability p_rewire
                new_network = rewire_edges(
                    new_network, 
                    p=self.p_rewire, 
                    directed=False, 
                    copy=False
                )

            elif self.network_type == 'sda':
                
                if self.homophily < 0:
                    # Random rewiring for special network types
                    new_network = rewire_edges(
                        new_network, 
                        p=self.p_rewire, 
                        directed=False, 
                        copy=False
                    )
                else:
                    # Homophily-based rewiring: prefer similar agents
                    b = SDA.optim_b(
                        D=dists, 
                        k=self.avg_degree, 
                        alpha=self.homophily
                    )
                    P = SDA.prob_measure(
                        D=dists, 
                        b=b, 
                        alpha=self.homophily
                    )

                    # Probabilistic rewiring based on similarity
                    p_rewire = self.p_rewire
                    rand_nrs, rand_nrs_2 = self.rng.random(size=(2, self.N))
                    
                    for i, row in enumerate(new_network):
                        if rand_nrs[i] < p_rewire:
                            if row.sum() > 1:
                                # Select random neighbor and non-neighbor
                                rand_nbor = self.rng.choice(np.nonzero(row)[0])
                                rand_non_nbor = self.rng.choice(
                                    np.nonzero(1-row)[0]
                                )

                                # Rewire if non-neighbor is more similar
                                if P[i, rand_non_nbor] > 0.00001:
                                    condition = (P[i, rand_nbor] / 
                                               P[i, rand_non_nbor]) / 2
                                    if rand_nrs_2[i] < condition:
                                        # Remove edge to neighbor
                                        new_network[i, rand_nbor] = 0
                                        new_network[rand_nbor, i] = 0
                                        # Add edge to non-neighbor
                                        new_network[i, rand_non_nbor] = 1
                                        new_network[rand_non_nbor, i] = 1
                                    
        # Calculate social capital if enabled
        if self.soc_to_wb != 0:
            # Social capital based on clustering with similar neighbors
            similarities = 1 - dists / dists.max()
            nx_network = nx.from_numpy_array(
                new_network * similarities, 
                edge_attr='Similarity'
            )
            self.soc_cap = np.array(
                list(nx.clustering(nx_network, weight='Similarity').values())
            )
        else:
            self.soc_cap = 0

        return new_network

    def init_wellbeing(self):
        """
        Initialise well-being levels for all agents.
        
        Initial well-being is determined by personal traits that create
        heterogeneous setpoints. These setpoints represent baseline well-being
        levels that individuals tend to return to after positive or negative 
        events.
        """
        self.wellbeing = np.zeros((self.N, self.n_steps + 1))

        # Generate personal traits that determine well-being setpoint
        traits = np.random.normal(0, 1, size=self.N)
        self.traits = (traits - traits.min()) / (traits.max() - traits.min())
        
        # Non-linear transformation to create setpoint heterogeneity
        swb_setpoint = np.log(self.traits + 0.25)
        
        # Normalize setpoints to range [0.1, 0.9] to avoid boundary issues
        self.swb_setpoint = 0.8 * (
            (swb_setpoint - swb_setpoint.min()) / 
            (swb_setpoint.max() - swb_setpoint.min())
        ) + 0.1

        # Special case: homogeneous population has identical setpoints
        if self.capital_dist == 'homogeneous':
            self.wellbeing[:, 0] = 0.5
            self.swb_setpoint = 0.5
            self.mean_shocks = 0
        else:
            self.wellbeing[:, 0] = self.swb_setpoint

        # Copy initial well-being to step 1
        self.wellbeing[:, 1] = self.wellbeing[:, 0]

    def sparse_row_quantile(self, A, q, include_zeros=False):
        """
        Compute quantiles along rows of a sparse CSR matrix efficiently.
        
        This method calculates quantiles without densifying the sparse matrix,
        which is crucial for memory efficiency with large networks.
        
        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Input sparse matrix (typically neighbor capital values)
        q : float in [0, 1]
            Desired quantile (e.g., 0.5 for median, 0.75 for 75th percentile)
        include_zeros : bool, default=False
            Whether to include implicit zeros in quantile calculation
        
        Returns
        -------
        quantiles : np.ndarray, shape (n_rows,)
            Quantile value for each row
        """
        if not sparse.isspmatrix_csr(A):
            A = A.tocsr()
    
        n_rows, n_cols = A.shape
        out = np.zeros(n_rows, dtype=A.dtype)
    
        for i in range(n_rows):
            # Extract non-zero data for this row
            row_start, row_end = A.indptr[i], A.indptr[i + 1]
            row_data = A.data[row_start:row_end]
    
            if row_data.size == 0:
                out[i] = 0.0
                continue
    
            if not include_zeros:
                # Quantile over non-zero values only
                out[i] = np.quantile(row_data, q, method='nearest')
                continue
    
            # Include zeros in quantile calculation
            nz = row_data.size
            zeros = n_cols - nz
            frac_zero = zeros / n_cols
    
            if q <= frac_zero:
                # Quantile falls in zero region
                out[i] = 0.0
            else:
                # Adjust quantile for non-zero portion
                adjusted_q = (q - frac_zero) / (1 - frac_zero)
                out[i] = np.quantile(row_data, adjusted_q, method='nearest')
    
        return out
    
    def update_adapt_lvl(self, weight):
        """
        Update adaptation levels based on past capital and social comparison.
        
        Adaptation level represents the reference point against which current
        capital is evaluated. It combines:
        - Personal capital history (habituation)
        - Capital of comparison group (social comparison)
        
        Parameters
        ----------
        weight : float or np.array
            Adaptation rate (higher = faster adaptation)
        
        Returns
        -------
        new_adapt_lvl : np.ndarray, shape (N,)
            Updated adaptation levels
        """
        prev_step = self.step - 1
        prev_cap = self.capital[:, prev_step]
        prev_adapt_lvl = self.al_capital[:, prev_step]
        
        if self.soc_comparison:
            # Get capital of network neighbors (including self)
            network = self.network
            network.setdiag(1)
            nbors_cap = network.multiply(prev_cap)
            
            # Calculate comparison value (typically median or upper quantile)
            soc_comp_values = self.sparse_row_quantile(
                nbors_cap, 
                q=self.quantile, 
                include_zeros=False
            )
            
            # Combine own capital and social comparison
            comparison_part = (
                (1 - self.soc_comp_w) * prev_cap + 
                self.soc_comp_w * soc_comp_values
            )
            
            # Update adaptation level with weighted average
            new_adapt_lvl = (
                self.alpha * comparison_part + 
                (1 - self.alpha) * prev_adapt_lvl
            )
        else:
            # No social comparison: adapt only to own history
            soc_comp_values = 0
            new_adapt_lvl = (
                weight * prev_cap + 
                (1 - weight) * prev_adapt_lvl
            )
        
        # Store comparison values for analysis
        self.soc_comp_values[:, prev_step] = soc_comp_values
        return new_adapt_lvl

    def sqrt_concave(self, x, y0, r, prev_wellbeing):
        """
        Concave square-root function for well-being perception.
        
        This creates a concave relationship between capital deviations and
        well-being, capturing diminishing sensitivity to large changes.
        
        Parameters
        ----------
        x : np.ndarray
            Deviation from adaptation level
        y0 : np.ndarray
            Well-being setpoint (y-intercept at x=0)
        r : float
            X-intercept parameter (determines curvature)
        prev_wellbeing : np.ndarray
            Previous well-being values (fallback for invalid calculations)
        
        Returns
        -------
        concave : np.ndarray
            Well-being values based on concave transformation
        """
        # Calculate curvature parameter
        a = y0 / np.sqrt(-r)
        concave = a * np.sqrt(x - r)
        
        # Constrain well-being to valid range (0, 1)
        concave = np.where(concave <= 0, 0.001, concave)
        concave = np.where(concave >= 1, 0.999, concave)
        
        # Replace any NaN values with previous well-being
        concave = np.nan_to_num(concave, copy=True, nan=prev_wellbeing)
        return concave

    def update_wellbeing(self, step):
        """
        Update well-being levels based on capital relative to adaptation level.
        
        Well-being depends on the gap between current capital and the adaptation
        level (which combines past capital and social comparison). The relationship
        is asymmetric: losses hurt more than equivalent gains help (loss aversion).
        
        Parameters
        ----------
        step : int
            Current simulation time step
        
        Returns
        -------
        swb : np.ndarray, shape (N,)
            Updated wellbeing values for all agents
        """
        # Get current capital and previous wellbeing
        curr_capital = self.capital[:, step]
        prev_wellbeing = self.wellbeing[:, step-1]

        # Update adaptation level (combines habituation and comparison)
        al_capital = self.update_adapt_lvl(weight=self.alpha)
        self.al_capital[:, step] = al_capital
        
        # Calculate deviation from adaptation level
        diff_cap = curr_capital - al_capital

        # Parameters for wellbeing saturation function
        x0_point = self.swb_setpoint / (1 - self.swb_setpoint)
        lamb = 2.25  # Loss aversion parameter (losses weighted more)

        if self.perception == 'concave':
            # Concave square-root perception function
            r = -20_000
            swb = np.where(
                diff_cap <= 0, 
                self.sqrt_concave(
                    x=diff_cap, 
                    y0=self.swb_setpoint, 
                    r=r, 
                    prev_wellbeing=prev_wellbeing
                ),
                self.sqrt_concave(
                    x=diff_cap, 
                    y0=self.swb_setpoint, 
                    r=r * lamb, 
                    prev_wellbeing=prev_wellbeing
                )
            )
        else:
            # Sigmoid perception function (default)
            if self.N > 10:
                # Capital-dependent sensitivity (rich less sensitive to changes)
                min_slope = 0.0001
                max_slope = 0.00001
                min_val = curr_capital.min()
                max_val = curr_capital.max()
                slope_sig = min_slope + (
                    (curr_capital - min_val) / (max_val - min_val)
                ) * (max_slope - min_slope)
                self.slope_sig = -1 * slope_sig
            
            # Asymmetric sigmoid: steeper for losses than gains
            swb = np.where(
                diff_cap < 0, 
                x0_point / (x0_point + np.exp(self.slope_sig * lamb * diff_cap)),
                x0_point / (x0_point + np.exp(self.slope_sig * diff_cap))
            )
            
        return swb

    def init_events(self):
        """
        Initialise shock events that affect agent capital.
        
        Two types of events are supported:
        1. Fixed interventions: deterministic shocks at regular intervals to
           a random subset of agents
        2. Random shocks: stochastic fluctuations affecting all agents at all times
        
        Events can be combined (e.g., 'random+fixed') to model both regular
        background volatility and discrete intervention events.
        """
        event_type = self.event_type
        self.cap_event_sizes = np.zeros(self.capital.shape)

        if 'fixed' in event_type:
            # Fixed intervention events at regular intervals
            T = self.n_steps
            int_freq = self.int_freq
            int_size = self.int_size
            
            # Determine intervention time steps
            steps = np.array([
                step for step in range(50, T+1) 
                if step % int_freq == 0
            ]).astype(int)
            self.cap_event_steps = steps
            
            # Intervention amounts at each step
            amounts = np.array([
                int_size for step in range(50, T+1) 
                if step % int_freq == 0
            ])
            
            # Randomly select which agents receive intervention
            random_idx = self.rng.binomial(n=1, p=self.p, size=self.N)
            self.random_idx = random_idx
            
            # Create intervention matrix (agents x time steps)
            fixed_cap_events = random_idx[:, np.newaxis] * amounts[np.newaxis, :]

            # Add interventions to capital and track sizes
            self.capital[:, steps] += fixed_cap_events
            self.cap_event_sizes[:, steps] += fixed_cap_events
        
        if 'random' in event_type:
            # Generate random shocks from specified distribution
            if 'gev' in event_type:
                # Generalized Extreme Value distribution (for fat tails)
                random_cap_shocks = genextreme.rvs(
                    loc=0, 
                    c=0, 
                    scale=self.shock_sigma, 
                    size=self.capital.shape
                )
            elif 'student_t' in event_type:
                # Student's t-distribution (for heavy tails)
                random_cap_shocks = t.rvs(
                    df=self.shock_sigma, 
                    size=self.capital.shape
                )
            else:
                # Normal distribution (default)
                random_cap_shocks = self.rng.normal(
                    loc=0,
                    scale=self.shock_sigma, 
                    size=self.capital.shape
                )
            
            # Store random shocks for use during simulation
            self.random_cap_shocks = random_cap_shocks
        else:
            # No random shocks
            self.random_cap_shocks = np.zeros(self.capital.shape)
                
    def update(self, step):
        """
        Perform a single simulation step update.
        
        Updates occur in sequence:
        1. Capital (based on growth, shocks, and well-being feedback)
        2. Well-being (based on capital relative to adaptation level)
        
        Parameters
        ----------
        step : int
            Current simulation time step
        """
        # Update capital levels
        self.capital[:, step] += self.update_capital(
            step, 
            self.capital, 
            self.wellbeing, 
            self.wb_to_cap,
            self.random_cap_shocks
        )
        
        # Update well-being levels
        self.wellbeing[:, step] = self.update_wellbeing(step=step)

    def run_simulation(self):
        """
        Execute the full simulation from start to finish.
        
        Iterates through all time steps, updating capital and well-being at each
        step. The step counter is stored as an instance variable to enable
        time-dependent behavior (e.g., intervention timing).
        """
        for step in range(2, self.n_steps + 1):
            self.step = step
            self.update(step)
        

if __name__=='__main__':

    N = 50
    params = {
    # Network params
    'N': N, 'p': 0.5, 'network_type': 'sda', 
    'homophily': -1, 'avg_degree': 10, 'p_rewire': 0.1, 'wb_in_dist':False,

    # Simulation params
    'n_steps': 99,  'alpha': np.random.uniform(0, 0.5, size=N), 
    'soc_comp_w': np.random.uniform(0, 1, size=N), 'shock_sigma':0.005, 
    'soc_to_wb':False, 'slope_sig':-0.0001, 'perception':'sigmoid',

    # Income params
    'wb_to_cap':np.random.uniform(0, 2, size=N), 'mu': 10, 'sigma': 0.1, 
    'capital_dist': 'lognormal', 'soc_comparison':True, 
    'comparison_kind':'quantile', 'quantile':0.5, 'k':1, 'growth_rate':0.002,  
    
    # Event params
    'event_type':'fixed+random',
    'int_freq':50, 'int_size':-10_000, 'rel_size':0, # If using relative shocks
}
    model = WellbeingSim(params=params)
    model.run_simulation()
    print(model.wellbeing)