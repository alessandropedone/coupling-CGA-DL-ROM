# Multi-physics model

This note documents the mathematical model for an idealized elctromechanical MEMS implemented in the coupled solver in the ``multi_physics`` package.

> For more details on the implementation and specific features, please refer to the package documentation.

## Overview

The physics of this system is made of two parts:
- __Electrostatics__: the electrostatic potential $\phi$ solves the Laplace equation, with Dirichlet boundary conditions on the electrodes.
- __Mechanics__ (of the upper electrode): Euler-Bernoulli beam deformation modeled via a 4-mode reduced-order model.

> The choice of 4 as the number of modes is motivated by the fact that actually only the first mode is actually relevant for these kind of applications. In particular, if you are interested, you can take a look at the test which compares the simulations with 1, 2, 3 and 4 modes.

The coupling is due to the __electrostatic force__ which can deform the upper electrode. This force can be computed using Maxwell-stress traction, which depends on the electric field, and we project it onto the modal basis to obtain generalized modal forces.

*Reduced-order modeling* comes into play with two main approaches:
- __Classical (mechanical)__: we project the mechanical part onto the first 4 modes;
- __DL (electrostatic)__: we speedup the computation of the force replacing the full FEM computation of $\phi$ with a geometry-aware DL surrogate model that predicts directly the force.

So, this justifies the use of the following terms in the sequel:
- __Classical ROM__: solver that uses the modal projection and the full FEM approximation of the electrostatic part;
- __DL-ROM__: solver that uses the modal projection and computes the electrostatic force using a DeepONet.

> DL-ROM uses __both__ reduced-order modeling approaches.

> The DL-ROM allows us to overcome the main bottleneck of the problem: remeshing at each time step, since the force is just given directly by the neural network. See below and the report for more details.

Now we discuss in depth the classical ROM, and at the end we make a brief comment on the differences with the DL-ROM.

## Geometry and kinematics


The problem is time-dependent, so at each time step, we will need to update the geometry of the domain.

### Reference 2D geometry 

Two parallel electrodes are embedded in a circular outer boundary. We recall here the tags related to Gmsh `.geo` template of the reference geometry.
Typical physical tags in the Gmsh `.geo` template:
- `10`: `force_segment` — portion of moving/upper electrode boundary where forces are evaluated, i.e. the lower edge;
- `11`: `upper_plate` — remaining moving/upper  electrode boundary;
- `12`: `lower_plate` — fixed/lower electrode boundary;
- `20`: `boundary` — outer circular boundary;

> The electrodes are treated as **perfect conductors** and they do not belong to the electrostatic computational domain.

> The `.geo` template is written in microns. The solver converts mesh coordinates to meters (SI) immediately after reading.

### Modal parametrization
The moving electrode boundary is parametrized by the first 4 mode shapes $\{v\}_{n=1}^4$, more specifically we have that the displacement (from the rest configuration) $w$ is given by

$$
w(x,t) = \sum_{n=1}^{4} \hat v_n(x) \, q_n(t),
$$

where $q_n(t)$ are the modal coordinates/coefficients.

The mode shapes $\hat w_n$ depend on the choice of boundary condition. Here we discuss the cantilever case and the clamped-clamped case.

__Cantilever.__ The mode shapes are

$$
v_n (x)= \cosh\beta_nx - \cos\beta_nx + C_n\ (\sin\beta_nx-\sinh\beta_nx)
$$

with $\xi\in[0,L]$ and 

$$
C_n=\frac{\cos\beta_nL + \cosh\beta_nL}{\sin\beta_nL+\sinh\beta_nL}.
$$

The wavenumbers are $\beta_i = \lambda_i/L$, where $\lambda_n$ are the roots of 

$$
\cosh(\beta_n L)\cos(\beta_nL)+1 = 0.
$$

The first roots are

$$
\lambda_1\approx 1.8751,\quad
\lambda_2\approx 4.6941,\quad
\lambda_3\approx 7.8548,\quad
\lambda_4\approx 10.9955.
$$

__Clamped-Clamped.__ The mode shapes are

$$
v_n (x)= \sinh\beta_nx - \sin\beta_nx + C_n (\cosh\beta_nx-\cos\beta_nx).
$$

with $\xi\in[0,L]$ and

$$
C_n = \frac{\cos\beta_nL - \cosh\beta_nL}{\sin\beta_nL+\sinh\beta_nL}.
$$

The wavenumbers are $\beta_n=\lambda_n/L$, where
$\lambda_n$ are the roots of

$$
\cosh(\lambda)\cos(\lambda)-1=0.
$$

The first roots are

$$
\lambda_1\approx 4.7300,\quad
\lambda_2\approx 7.8532,\quad
\lambda_3\approx 10.9956,\quad
\lambda_4\approx 14.1372.
$$



## Electrostatics

Let's formulate more precisely the electrostatic problem:

$$ \begin{cases}
-\nabla\cdot\left(\varepsilon\,\nabla \phi\right) = 0 &\text{in } \Omega(t)\\ 
\phi = V_u\ &\text{on }\Gamma_u(t)\\
\phi = V_\ell(t)\ &\text{on }\Gamma_\ell\\
\varepsilon\nabla\phi\cdot n = 0 \quad \text{or} \quad  \phi = V_o&\text{on } \Gamma_o\\
\end{cases}
$$

where $\varepsilon=\varepsilon_0\varepsilon_r$ is the permittivity, $\Gamma_u(t)$ is the moving/upper conductor boundary, $\Gamma_\ell$ is the fixed/lower conductor boundary, and $\Gamma_o$ is the outer boundary. 

> Recall that $\varepsilon_0 = 8.8541878128\times 10^{-12}\,\mathrm{F/m}$.

> $\varepsilon_r$ is user-specified (default $1$).

> The electric field is just $\mathbf{E} = -\nabla \phi$.


__Implementation notes:__
1. In the solver implementation the following typical driving voltage is used:

$$ 
V_\ell(t)=V_{\mathrm{dc}} + V_{\mathrm{ac}}\sin(2\pi f t).
$$

3. The conforming finite element space $\mathbb{P}^1$ of continuous piecewise linear functions is used for the discrete weak formulation of the problem for $\phi$.
4. The electric field $\mathbf{E}$ is obtained by computing exactly the gradient of $\phi$, which will belong automatically to the discontinuous finite element space $\mathbb{P}^0$ of piecewise constant functions.



## Electrostatic traction via Maxwell stress

In a linear dielectrics, the Maxwell stress tensor is given by

$$
\mathbf{T} = \varepsilon\left(\mathbf{E}\otimes\mathbf{E} - \frac{1}{2}|\mathbf{E}|^2\mathbf{I}\right).
$$

The traction on the domain can be computed as

$$
\mathbf{t} = \mathbf{T}\,\boldsymbol{n},
$$

where $n$ is the outward unit normal with respect to $\Omega(t)$, so by action-reaction the force on the moving electrode is

$$
\mathbf{t}_u = -\mathbf{T}\,\boldsymbol{n}.
$$

> This sign convention is essential, since the electrode is represented as a hole in the domain mesh.

With this traction, it's possible to compute the physical force (Newtons) acting on the electrode in this way:

$$
\mathbf{F} = b\int_{\Gamma_u}\mathbf{t}_u\,ds,
$$

where $b$ is the out-of-plane thickness (user-specified, with a default value of $10\,\mu m$).

## Generalized modal forces

Instead of computing the actual force, we want to project the integration of the traction onto the first 4 modes.

As explained in the report, the modal shape vector is taken as transverse-only:

$$
\hat{\boldsymbol{w}}_n(x) =
\begin{bmatrix}
0\\
v_n(x)
\end{bmatrix}.
$$

Then the corresponding generalized modal force is computed by virtual work:

$$
F_n(t) = b\int_{\Gamma_{10}(t)} \mathbf{t}_u(s,t)\cdot \hat{\boldsymbol{w}}_n(s)\,ds,
$$

where $\Gamma_{10}(t) \subseteq \Gamma_u(t)$ is the boundary segment marked by physical tag 10 (`force_segment`), that is the lower edge.


## Modal mechanical model

The moving beam is represented by modal coordinates $\{q_n(t)\}_{n=1}^4$. Skipping the datails (see report), we know that, since the mode shapes are an orthogonal base of eigenfunctions of the operator $\partial_x^4$, the damped equation for the an Euler-Bernoulli beam becomes the following diagonal system of ODEs:

$$
m_n\ddot q_n(t) + c_n \dot q_n(t) + k_n q_n(t) = F_n(t),
$$

where:
- $m_n$ is the modal mass [kg],
- $c_n$ is the modal damping [kg/s],
- $k_n$ is the modal stiffness [N/m],
- $F_n$ is the generalized force [N].

It is standard to express the parameters in terms of the natural frequencies $\omega_n$ and damping ratios $\zeta_n$

$$
k_n = m_n\omega_n^2,\qquad c_n = 2\zeta_n\omega_n m_n,
$$

we therefore adopt this representation.


## 5. Time integration

For time integration, a Newmark scheme with $\beta=\frac{1}{4}$, $\gamma=\frac{1}{2}$ is applied independently to each modal DOF.

Given $q^k, \dot q^k, \ddot q^k$, define predictors:

$$
q_{\mathrm{pred}} = q^k + \Delta t\,\dot q^k + \frac{\Delta t^2}{2}(1-2\beta)\ddot q^k,
$$

$$
\dot q_{\mathrm{pred}} = \dot q^k + \Delta t(1-\gamma)\ddot q^k.
$$

Then solve for $\ddot q^{k+1}$:

$$
\ddot q^{k+1} =
\frac{
F^{k+1} - c\,\dot q_{\mathrm{pred}} - k\,q_{\mathrm{pred}}
}{
m + \gamma\Delta t\,c + \beta\Delta t^2\,k
}.
$$

Finally, correct:

$$
q^{k+1} = q_{\mathrm{pred}} + \beta\Delta t^2 \ddot q^{k+1},\qquad
\dot q^{k+1} = \dot q_{\mathrm{pred}} + \gamma\Delta t \ddot q^{k+1}.
$$

In the coupled implementation, $F^{k+1}$ is approximated from the mesh generated by the current modal coordinates (explicit/partitioned coupling). More accurate schemes would use a fixed-point iteration per time step, but this is deliberately avoided here.


## Coupling algorithm

For time steps $k=0,\dots,N-1$:

1. **Geometry update**: construct moving electrode boundary from $\{q_n^k\}_{n=1}^4$ and write `.geo`.
2. **Remesh**: run Gmsh to generate air mesh $\Omega(t_k)$.
3. **Electrostatics solve**: compute $\phi_h^k$.
4. **Maxwell stress**: compute traction $\mathbf{t}_c^k$ on $\Gamma_{10}(t_k)$.
5. **Modal projection**: compute generalized forces $\{F_n^k\}_{n=1}^4$.
6. **Time integration**: update $q_n^{k+1}, \dot q_n^{k+1}, \ddot q_n^{k+1}$ with Newmark for every $n$.
7. **Output**: write $\phi_h^k$ to a ParaView time series, store modal histories and execution times.

> Modal coordinates $q_n$ are stored internally in meters and converted to microns only when substituting modifying geometries.

## Diagnostic quantities

At each step $t_k$, the following quantities are computed, displayed and saved:

$$
\begin{gather*}
\min_{\Omega(t_k)} \phi_h^k \\[6pt]
\max_{\Omega(t_k)} \phi_h^k \\[6pt]
\max_{\Omega(t_k)} |\mathbf{E}| \\[6pt]
W = \frac12 \varepsilon \int_{\Omega} |\nabla \phi|^2 \, dx &\text{(electrostatic energy)}\\[6pt]
C \approx \frac{2W}{(V_\ell - V_u)^2} &\text{(capacitance-like estimate)}
\end{gather*}
$$

## DL-ROM

### Normal derivative prediction

The main difference between the two ROMs lies in the fact that the DL-ROM has access to a neural network that allows us to predict 

$$
\frac{\partial\phi}{\partial \boldsymbol{n}}
$$ 

on the lower edge of the upper electrode. In particular, the neural network we propose takes as input the geometric parameters: $q_n$, distance between the electrodes and over-etch of the upper electrode.

> All inputs of the network are in converted microns.

Actually you don't need to specify $V_l$ (sinusoidal in time). In fact, if you set $V_u=0$ and $V_l=1$ and train the network, you can derive the normal derivative for a generic couple of voltages using a simple tranformation:

$$
\frac{\partial\phi}{\partial \boldsymbol{n}} = (V_l - V_u) \, \mathcal{D}(\mu)
$$

where $D$ is the DeepONet and $\mu$ is the vector containing the geometric parameters.


### Generalized forces

This is useful since the traction can be rewritten in this way:

$$
\mathbf{t}_u = -\frac{1}{2}\varepsilon\left(\frac{\partial\phi}{\partial \boldsymbol{n}}\right)^2 \boldsymbol{n}
$$

and $\boldsymbol{n}$ can be computed without the mesh in the entire domain. Indeed, it's sufficient to discretize lower edge and integrate on it to get the generalized force:

$$
F_n(t) = b\int_{\Gamma_{10}(t)} -\frac{1}{2}\varepsilon\left(\frac{\partial\phi}{\partial \boldsymbol{n}}(s)\right)^2 n_2(s) \, \hat w_n(s)\,ds.
$$

### Potential surrogate

There's also the possibility to use a surrogate for the potential $\phi$. In this repository we only illustrate this possibility and it's accuracy, while this is not a source of computational speedup, since in any case you need a mesh to visualize the solution. See the two test cases about visualization of the solution.


## Modelling assumptions

- Quasi-static electrostatics (no displacement currents in time).
- Perfectly conducting electrodes with prescribed potentials.
- Domain treated as linear dielectric (constant $\varepsilon_r$).
- Modal mechanics assumes a linear basis and diagonal modal dynamics (no geometric nonlinearity, no contact/pull-in).
- Coupling is partitioned with remeshing; stability near pull-in may require smaller $\Delta t$ and/or iterative coupling.