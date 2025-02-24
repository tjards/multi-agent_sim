# Stability of Semi-Trusted Heterogeneous Flocks

**Abstract:** *Flocking is a type of collective behaviour involving group objectives, often requiring the maintenance of structural qualities such as orientation and separation. A prevailing assumption of classical flocking methodologies is that of uniform values of inter-agent distances. Many practical applications demand greater flexibility, as the diversity of types and configurations grows with the popularity of multi-agent systems in society. Moreover, agents often operate with a mixture of shared and distinct goals without guarantees of trust or secure communication. Motivated by these challenges we update well‐established frameworks for cohesion, alignment, and navigation by relaxing this assumption of shared inter-agent distances and constraints. Through a new form of constrained collective potential function, we introduce a solution that permits negotiation of these parameters. In the spirit of the traditional flocking control canon, this negotiation is achieved purely through local observations and does not require any global information or inter-agent communication. The approach is robust to semi-trust scenarios, where neighbouring agents pursue conflicting goals. We validate the effectiveness of the approach through a series of simulations and a rigorous stability analysis.*

---

## 1. Introduction

Multi-agent robotics research is motivated by the simple idea that tasks can be completed better, faster, or with less effort when multiple agents work together. Swarm robotics describes such systems which are decentralized and asynchronous, composed of quasi-identical agents operating under simple rules and local information [1]. The collective behavior of the swarm is emergent, with properties not present in the individuals.

An example of swarm behavior is *flocking*, for which the initial properties were formulated in [2] and later integrated into the contemporary control theory canon in [3]. Flocking is a type of collective behavior that involves group objectives, such as target tracking or maintenance of structural qualities; this often takes the form of evenly-spaced agents converging around a common target, forming what is sometimes referred to as a *lattice* structure. The practical applications of flocking strategies are numerous, ranging from mobile communications, remote monitoring and data collection, distributed resource delivery, search and rescue, and military operations [4]. A recent review suggested a need for more comprehensive standardization of flocking methodologies in order to enable greater adoption in real-world applications [5]. This work is concerned with scenarios in which such agents must work together on a mixture of common and distinct goals with limited trust [6, 7].

### 1.1 Related Work

As the field draws from a broad range of related domains—including physics, biology, sociology, and computer science—it is rich with diverse methodologies, often achieving similar outcomes in different ways. Due largely to the work of [8, 9, 10], gradient- and consensus-based methods have drawn considerable attention in flocking research. In such approaches, interactions between agents are typically represented as a graph, while performance and stability are analyzed in terms of control-Lyapunov functions. While significant work has investigated the case of time-varying topologies [11], a common assumption is that the graph is undirected and connected at all times [3]. This early work generally relied on two assumptions: that agents share common navigation objectives and separation parameters. Later work relaxed the former condition using *pinning* control, demonstrating that only a subset of the agents need be informed of navigation goals to remain stable [12, 13].

### 1.2 Agent Dynamics

Let us consider agent *i* ∈ ℵ at time *t* with position **x**₍ᵢ₎(*t*) ∈ ℝ², velocity **v**₍ᵢ₎(*t*) ∈ ℝ², and inputs **u**₍ᵢ₎(*t*) ∈ ℝ² governed by the following *double integrator* dynamics:

$$
\begin{aligned}
\dot{\mathbf{x}}_i(t) &= \mathbf{v}_i(t), \\
\dot{\mathbf{v}}_i(t) &= \mathbf{u}_i(t),
\end{aligned}
$$

where **x**₍ᵢ₎(*t*) represents Cartesian coordinates and **u**₍ᵢ₎(*t*) = ¨**x**₍ᵢ₎(*t*) is the acceleration vector. Such dynamics are applicable to a wide range of systems.

### 1.3 Connectivity

Consider a set of agents represented as an undirected connected graph 𝒢 = (ℵ, ℰ), where ℰ ⊆ {(*i, j*): *i, j* ∈ ℵ, *j* ≠ *i*}. The neighbourhood of agent *i* is defined at any instant as:

$$
\mathcal{V}_i = \{ j \in \mathcal{V} : \| \mathbf{x}_i(t) - \mathbf{x}_j(t)\| < r_i \},
$$

with *r*₍ᵢ₎ being the fixed range at which agent *i* can sense or communicate with its neighbours. The **adjacency matrix** *A* is defined such that

$$
A_{ij} = 
\begin{cases} 
1, & \text{if an edge exists between } i \text{ and } j, \\
0, & \text{otherwise}.
\end{cases}
$$

A *component* of the graph is a subset of vertices with no neighbours outside itself. Later, we introduce *pinning control*, where certain agents are selected to drive the motion of the overall network. Components are important in this context, as each requires at least one pin to drive its vertices toward a target. Pin vertices are typically selected based on measures of centrality.

## 2. Homogeneous Lattice Flocking

This section draws on previous work to formulate homogeneous lattice flocking. The basic elements—cohesion, alignment, and navigation—are presented as energy potential functions. (For convenience, we temporarily omit the time variable *t*.)

### 2.1 Cohesion

Numerous solutions exist in the literature to achieve cohesion in a lattice (as defined by $$\|\mathbf{x}_i - \mathbf{x}_j\| = d).$$

Examples include variations of the Lennard-Jones potential [10, 14], the formulation of Olfati-Saber [3], Morse potential [15], Soft Repulsion [16], Mixed Exponential and Polynomial approaches [17], Riesz potential [18], Gaussian potential [19], and the Gromacs Soft-Core method [20]. In all cases, each agent *i* minimizes an energy potential function \\(V^c_i(\mathbf{z}^c_i)\\) via gradient descent:

$$
\dot{\mathbf{x}}_i = -\nabla_{\mathbf{x}_i}V^c_i(\mathbf{z}^c_i),
$$

where \\(\mathbf{z}^c_i\\) (typically \\((\mathbf{x}_i, \mathbf{x}_j)\\)) and a desired separation \\(d_{ij}\\) are used along with a tunable parameter \\(k_c > 0\\). For example, a modified Lennard-Jones potential is given by

$$
V^c_i(\mathbf{x}_i, \mathbf{x}_j) = \left( \frac{k_c d_{ij}^6}{2\|\mathbf{x}_i - \mathbf{x}_j\|} \right)^{12} - \left( \frac{k_c}{\|\mathbf{x}_i - \mathbf{x}_j\|} \right)^6,
$$

with its gradient

$$
\nabla_{\mathbf{x}_i}V^c_i(\mathbf{x}_i,\mathbf{x}_j) = 6k_c\left(\frac{-d_{ij}^6}{\|\mathbf{x}_i - \mathbf{x}_j\|^{13}} + \frac{1}{\|\mathbf{x}_i - \mathbf{x}_j\|^7} \right) \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|}.
$$

This controller drives the system to the equilibrium where \\(\|\mathbf{x}_i - \mathbf{x}_j\| = d_{ij}\\). (Later, we relax the assumption that \\(d_{ij}\\) is fixed.)

### 2.2 Alignment

Alignment is the tendency of agents to adopt the average heading of their neighbours. The control policy for alignment is given by

$$
\dot{\mathbf{v}}_i = -\nabla_{\mathbf{v}_i}V^a_i(\mathbf{z}^a_i),
$$

with \\(\mathbf{z}^a_i\\) typically including \\((\mathbf{x}_i, \mathbf{x}_j, \mathbf{v}_i, \mathbf{v}_j)\\). For example, one formulation is

$$
V^a_i(\mathbf{x}_i, \mathbf{x}_j, \mathbf{v}_i, \mathbf{v}_j) = \frac{1}{2} k_a\sum_{j \in \mathcal{V}_i} \rho(\|\mathbf{x}_i - \mathbf{x}_j\|) \|\mathbf{v}_i - \mathbf{v}_j\|^2,
$$

where \\(k_a > 0\\) and \\(\rho(\cdot)\\) is a smooth weighting function. One can compute the gradient accordingly [21].

### 2.3 Navigation

A central finding in [3] is the need to incorporate group objectives to build and maintain global cohesion when the flock is not initially connected. When lattice formation is pursued without a group objective, fragmentation may occur. Previous work has addressed this by defining a group objective via a shared navigation term. In our approach, we reduce the number of agents required to execute the navigation task by incorporating pinning control. For each connected component of the graph, the agent with the maximum degree centrality is selected as a pin (see, e.g., [22, 23]). In addition, pinning of leaf nodes [24] can improve overall convergence properties.

### 2.4 Summary

The conditions under which the combined implementations of cohesion, alignment, and navigation are stable—assuming shared separation parameters—are well established in the literature [3, 14]. Without loss of generality, for the homogeneous case where \\(d_{ij} = d, \; \forall (i,j) \in \mathcal{V}\\), there exists a stable collective potential function \\(V^f_i(\mathbf{z}^f_i)\\) (with \\(\mathbf{z}^f_i = (\mathbf{x}_i, \mathbf{x}_j, \mathbf{v}_i, \mathbf{v}_j)\\)) whose gradient descent control law drives the system to an equilibrium configuration with the desired separations.

## 3. Heterogeneous Lattice Flocking

In practice, agents may be configured with different initial desired separations \\(d_{ij}(0)\\) (with \\(d_{ij}(0) \not\equiv d_{ji}(0)\\)) and distinct separation constraints. In this setting, each agent must negotiate its own \\(d_{ij}(t)\\) through local observation without direct communication. A low-pass filter is introduced to estimate a neighbour’s separation parameter:

$$
\dot{\hat{d}}_{ji}(t) = \frac{1}{\tau} \Bigl( \|\mathbf{x}_i(t) - \mathbf{x}_j(t)\| - \hat{d}_{ji}(t) \Bigr), \quad \hat{d}_{ji}(0) = d_{ij}(0),
$$

where \\(\tau > 0\\) is the filter’s time constant.

A custom smooth *bump* function is defined to enforce each agent’s separation constraints \\(d_{i,\min} < d_{ij}(t) < d_{i,\max}\\):

$$
V_i^b(\hat{d}_{ji}) =
\begin{cases}
\exp\!\left[ -\left( \frac{\lambda^2}{1 - \lambda^2} \right)^p \right], & \text{if } d_{i,\min} < \hat{d}_{ji} < d_{i,\max}, \\
0, & \text{otherwise},
\end{cases}
$$

with

$$
\lambda = \frac{2\left(\hat{d}_{ji} - \frac{d_{i,\min} + d_{i,\max}}{2}\right)}{d_{i,\max} - d_{i,\min}},
$$

and \\(p > 1\\) controlling the flatness of the bump. This function is then incorporated into a constrained collective potential function that drives each agent’s desired separation \\(d_{ij}(t)\\) toward its estimated neighbour’s value \\(\hat{d}_{ji}(t)\\) when within acceptable bounds.


## 4. Stability Analysis

A Lyapunov candidate function is constructed that combines the potentials for cohesion, alignment, navigation, and the lattice scaling induced by the negotiation of \\(d_{ij}(t)\\):

$$
V_L(t) = \sum_{i\in \mathcal{V}} \sum_{j\in \mathcal{V}_i} \Bigl( V^c_i(\mathbf{x}_i(t), \mathbf{x}_j(t)) + V^a_i(\mathbf{x}_i(t), \mathbf{x}_j(t), \mathbf{v}_i(t), \mathbf{v}_j(t)) + V^n_i(\mathbf{x}_i(t), \mathbf{v}_i(t)) + V^d_{ij}(d_{ij}(t), \hat{d}_{ji}(t)) \Bigr).
$$

Under the assumption that the homogeneous case is stable [3, 14] and that the perturbations introduced by the negotiation of \\(d_{ij}(t)\\) are bounded, one can show that the overall Lyapunov candidate is bounded-input bounded-output (BIBO) stable. Moreover, by designing the control law for \\(d_{ij}(t)\\) based on the gradient of \\(V^d_{ij}(d_{ij}, \hat{d}_{ji})\\), the lattice scaling component is shown to be asymptotically stable in the Lyapunov sense.

## 5. Experimental Results

The proposed flocking techniques were implemented in simulations. In the first simulation, 30 agents were assembled into a homogeneous flock with a fixed desired separation \\(d_{ij} = 10\\) (see Table 1). The agents initially distributed randomly converged to a stable configuration (Figures shown in the original document).

In a second simulation, 7 agents were initiated with conflicting separation parameters and different constraints. Despite the lack of direct communication, each agent successfully negotiated its own \\(d_{ij}(t)\\) via local observations, leading to a stable flock configuration with acceptable negotiated separations.

## Conclusion

A broad range of classical flocking methodologies provide stable, emergent lattice formations under the assumption of uniform inter-agent distances. Many real‐world applications demand greater flexibility as multi-agent systems become increasingly diverse. Motivated by these challenges, our work relaxes the assumption of a shared inter-agent separation and individual constraints by updating the established frameworks for cohesion, alignment, and navigation. Central to our approach is a constrained potential function that negotiates inter-agent parameters locally without the need for global communication. Simulation results and rigorous stability analysis validate the effectiveness of the proposed approach.

## References

[1] Beni, G. (2005). *From Swarm Intelligence to Swarm Robotics*. In *Swarm Robotics* (pp. 1–9). Springer Berlin Heidelberg.

[2] Reynolds, C. W. (1987). Flocks, herds, and schools: A distributed behavioral model. *ACM Siggraph Computer Graphics*, 21(4), 25–34.

[3] Olfati-Saber, R. (2006). Flocking for multi-agent dynamic systems: algorithms and theory. *IEEE Transactions on Automatic Control*, 51(3), 401–420.

[4] Xiao, Z., & Wang, X. (2024). Evolutionary Niche Artificial Fish Swarm Algorithm for Dynamic Subgroup Size Adjustment in Robot Swarms. *IEEE Transactions on Cognitive and Developmental Systems*, 16(4), 1274–1290.

[5] Nedjah, N., & Silva Junior, L. (2019). Review of methodologies and tasks in swarm robotics towards standardization. *Swarm and Evolutionary Computation*, 50, 100565.

[6] Kio, O. G., Yuan, M., Allison, R. S., & Shan, J. (2024). Performance-based Data-driven Assessment of Trust. In *2024 IEEE 4th International Conference on Human-Machine Systems (ICHMS)* (pp. 1–6).

[7] Akbari, B., Yuan, M., Wang, H., Zhu, H., & Shan, J. (2024). A Factor Graph Model of Trust for a Collaborative Multi-Agent System. *arXiv preprint arXiv:2402.07049*.

[8] Olfati-Saber, R., & Murray, R. M. (2004). Consensus problems in networks of agents with switching topology and time-delays. *IEEE Transactions on Automatic Control*, 49(9), 1520–1533.

[9] Olfati-Saber, R., & Murray, R. M. (2003). Consensus protocols for networks of dynamic agents. In *Proceedings of the American Control Conference* (pp. 951–956).

[10] Wen, G., Duan, Z., Su, H., Chen, G., & Yu, W. (2012). A Connectivity-preserving flocking algorithm for multi-agent dynamical systems with bounded potential function. *IET Control Theory and Applications*, 6(6), 813–821.

[11] Ren, W., & Beard, R. W. (2005). Consensus seeking in multiagent systems under dynamically changing interaction topologies. *IEEE Transactions on Automatic Control*, 50(5), 655–661.

[12] Su, H., Wang, X., & Lin, Z. (2009). Flocking of Multi-Agents With a Virtual Leader. *IEEE Transactions on Automatic Control*, 54(2), 293–307.

[13] Gao, J., Xu, X., Ding, N., & Li, E. (2017). Flocking motion of multi-agent system by dynamic pinning control. *IET Control Theory & Applications*, 11(5), 714–722.

[14] Moscato, P., & Haque, M. N. (2024). New alternatives to the Lennard-Jones potential. *Scientific Reports*, 14, 11169.

[15] Morse, P. M. (1929). Diatomic molecules according to the wave mechanics. II. Vibrational levels. *Physical Review*, 34(1), 57.

[16] Rogers, T., et al. (2014). Soft repulsive interactions for dense formations. *Journal of Physical Chemistry*, 118, 10–19.

[17] Schweighofer, M., et al. (2008). Exponential-attractive potentials for clustering. In *Swarm Intelligence Conference*.

[18] Landkof, N. S. (1972). *Foundations of Modern Potential Theory*. Springer.

[19] Franz, S. (2007). Gaussian potential functions in agent-based modeling. *Journal of Applied Physics*, 15, 1–20.

[20] Gapsys, V., Seeliger, D., & de Groot, B. L. (2012). New Soft-Core Potential Function for Molecular Dynamics Based Alchemical Free Energy Calculations. *Journal of Chemical Theory and Computation*, May, 1–5.

[21] Saber, R. O., & Murray, R. M. (2003). Flocking with obstacle avoidance: cooperation with limited communication in mobile networks. In *42nd IEEE International Conference on Decision and Control* (Vol. 2, pp. 2022–2028).

[22] Nozari, E., Pasqualetti, F., & Cortés, J. (2019). Heterogeneity of central nodes explains the benefits of time-varying control scheduling in complex dynamical networks. *Journal of Complex Networks*, 7(5), 659–701.

[23] Pasqualetti, F., Zampieri, S., & Bullo, F. (2014). Controllability metrics, limitations and algorithms for complex networks. In *2014 American Control Conference* (pp. 3287–3292).

[24] Liu, H., Xu, X., Lu, J.-A., Chen, G., & Zeng, Z. (2021). Optimizing Pinning Control of Complex Dynamical Networks Based on Spectral Properties of Grounded Laplacian Matrices. *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 51(2), 786–796.

