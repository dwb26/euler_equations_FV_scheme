import numpy as np

class HLLE(object):

	def __init__(self, L, nx, k):
		self.L = L
		self.nx = nx
		self.dx = L / (nx - 1)
		self.k = k

	def f(self, q):
		"""
		The flux function for the equations.
		"""
		rho = q[0]
		u = q[1] / rho
		return np.array([rho * u, rho * u ** 2 + rho / self.k])

	def jac_eig(self, Q):
		"""
		Evaluates the Jacobian of the flux function f at the cell value Q and returns the eigenvalues of the Jacobian. These eigenvalues are then used to estimate the smallest and largest wave speeds.
		"""
		# Evaluate the Jacobian
		rho = Q[0]
		u = Q[1] / rho
		Df = np.array([[0, 1], [-u ** 2 + 1 / self.k, 2 * u]])

		# Compute and return the eigenvalues
		w, v = np.linalg.eig(Df)
		return w

	def roe_eig(self, Q_l, Q_r):
		"""
		Computes the Roe average matrix over the interface between Q_l and Q_r and returns the eigenvalues.
		"""
		# Calculate the Roe average of u
		rho_l = Q_l[0]
		u_l = Q_l[1] / rho_l
		rho_r = Q_r[0]
		u_r = Q_r[1] / rho_r
		u_hat = (np.sqrt(rho_l) * u_l + np.sqrt(rho_r) * u_r) / (np.sqrt(rho_l) + np.sqrt(rho_r))

		# Construct the Roe matrix
		A_hat = np.array([[0, 1], [-u_hat ** 2 + 1 / self.k, 2 * u_hat]])

		# Compute and return the eigenvalues
		w, v = np.linalg.eig(A_hat)
		return w

	def estimate_low_speed(self, w_jac, w_roe):
		"""
		Estimate the low speed using the eigenvalues of the Jacobian on the left cell and the eigenvalues of the Roe matrix over the interface.
		"""
		a = np.min([w_jac[0], w_roe[0]])
		b = np.min([w_jac[1], w_roe[1]])
		s1 = np.min((a, b))
		return s1

	def estimate_high_speed(self, w_jac, w_roe):
		"""
		Estimate the high speed using the eigenvalues of the Jacobian on the right cell and the eigenvalues of the Roe matrix over the interface.
		"""
		a = np.max((w_jac[0], w_roe[0]))
		b = np.max((w_jac[1], w_roe[1]))
		s2 = np.max((a, b))
		return s2

	def estimate_fluctuations(self, Q_l, Q_r, s1, s2):
		"""
		Compute the fluctuation estimates.
		"""
		# Estimate the value of the star region using the HLLE speeds
		Q_hat = (self.f(Q_r) - self.f(Q_l) - s2 * Q_r + s1 * Q_l) / (s1 - s2)

		# Compute the wave estimates
		W1 = Q_hat - Q_l
		W2 = Q_r - Q_hat

		# Compute the fluctuation estimates
		apdq = np.max((0, s1)) * W1 + np.max((0, s2)) * W2
		amdq = np.min((0, s1)) * W1 + np.min((0, s2)) * W2
		return apdq, amdq

	def solve(self, Q_i, apdq, amdq, dt):
		"""
		Compute the forward time cell value using the incoming fluctuations at the cell interfaces and the computed time increment.
		"""
		return Q_i - dt / self.dx * (apdq + amdq)


































