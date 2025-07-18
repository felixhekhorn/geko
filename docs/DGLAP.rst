Solving DGLAP
=============

We follow here the notation and conventions from the `eko documentation`_.

.. _eko documentation: https://eko.readthedocs.io/

Photon |PDF| follow a modified |DGLAP| equation compared to the proton case, which contains an additional inhomogeneous term:

.. math::
    \frac{d}{d\ln(\mu_F^2)} \tilde{\mathbf{f}}(\mu_F^2) = -\gamma(a_s(\mu_F^2)) \cdot \tilde{\mathbf{f}}(\mu_F^2) - \tilde{\mathbf k}(a_s(\mu_F^2))

where :math:`\tilde{\mathbf k}` are the photon-parton splitting functions (in Mellin space), which are computable in |pQCD|:

.. math::
    \mathbf k(a_s) = a_{em}\sum_{j=0}\left(a_s\right)^{j} \mathbf k^{(j)}

We apply the the usual transformation of variables and get

.. math::
    \frac{d}{da_s} \tilde{\mathbf{f}}^\gamma(a_s) = -\frac{\gamma(a_s)}{\beta(a_s)} \cdot \tilde{\mathbf{f}}^\gamma(a_s) - \frac{\tilde{\mathbf k}(a_s)}{\beta(a_s)}

which we then need to solve.

We can write the solution as a sum of a homogeneous (hadronic) component :math:`\mathbf f^\gamma_{hom}`
and an inhomogeneous (point-like) component :math:`\mathbf f^{\gamma}_{inhom}`:

.. math::
    \mathbf f^\gamma = \mathbf f^\gamma_{hom} + \mathbf f^\gamma_{inhom}

While the homogeneous term is just given by the standard solution (using the (standard) |EKO|)

.. math::
    \tilde{\mathbf f}^\gamma_{hom}(a_s) = \tilde {\mathbf E}(a_s \leftarrow a_s^0) \tilde{\mathbf f}^\gamma(a_s^0)

we find for the inhomogeneous term

.. math::
    \tilde{\mathbf f}^\gamma_{inhom}(a_s) = \int\limits_{a_s^0}^{a_s}\! da_s'\, \tilde{\mathbf E}(a_s \leftarrow a_s') \frac{-\tilde{\mathbf k}(a_s')}{\beta(a_s')}

which is thus the central equation we need to solve in γEKO.
Note that both the homogeneous and inhomogeneous solution have to be solved simultaneously (or better consistently).

Leading order
-------------

At |LO| all ingredients are known analytically and we can give a closed form solution. For the non-singlet case we find

.. math::
    \tilde f_{inhom,ns}^{\gamma,(0)}(a_s) = \frac {a_{em}k_{ns}^{(0)}}{(\gamma_{ns}^{(0)}+\beta_0)} \left(\frac{\exp\left(\gamma_{ns}^{(0)}\ln(a_s/a_s^0) / \beta_0\right)}{a_s^0} -\frac 1 {a_s} \right)

and, analogously, in the singlet case

.. math::
    \tilde {\mathbf f}_{inhom,S}^{\gamma,(0)}(a_s) = \sum_{\lambda\in\{+,-\}} \frac {a_{em}}{(\gamma_{S,\lambda}^{(0)}+\beta_0)} \left(\frac{\exp\left(\gamma_{S,\lambda}^{(0)}\ln(a_s/a_s^0) / \beta_0\right)}{a_s^0} -\frac 1 {a_s} \right) \mathbf e_{\lambda}^{(0)} \left(\begin{matrix}
        k_q^{(0)}\\
        0
    \end{matrix}\right)

where we need to sum over the two eigenvalues of the singlet anomalous dimension matrix.

Iterative solution
------------------

Beyond |LO| |EKO| s are in general not known as closed form expression, but a numerical approximation strategy has to be implemented
(see `eko documentation`_ for a detailed discussion).
Currently γEKO only supports the :code:`iterate-exact` solution, which relies on an iterative approach to solve the |RGE|.
We exploit the strategy for computing the (standard) |EKO| :math:`\tilde {\mathbf E}` to solve *simultaneously* our master equation here.
The central observation is that we can use the same decomposition of :math:`\tilde {\mathbf E}` into (infinitesimally) smaller pieces
also for solving our master equation, by applying the trapezoidal rule to the integral.

In practice it works like this:

- assume we split the integral along the points :math:`\{a_s^k, k = 0\ldots M\}` with :math:`a_s^M = a_s` the upper boundary of the integral
- define the interval ranges :math:`\{\Delta a_s^k = a_s^{k+1} - a_s^k, k = 0\ldots M-1\}`
- start the iteration by :math:`\tilde{\mathbf g}^0 = \frac{\Delta a_s^0}{2} \tilde {\mathbf E}(a_s^1 \leftarrow a_s^0) \frac{-\tilde{\mathbf k}(a_s^0)}{\beta(a_s^0)}`
- iterate :math:`M-2` times:
- close the iteration by
- identify the final result: :math:`\tilde{\mathbf f}^\gamma_{inhom}(a_s) = \tilde{\mathbf g}^M`
