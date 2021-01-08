from solvers import HLLE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ------------------------------------------------------------------------------------------------------------------- #
#
# Parameters
#
# ------------------------------------------------------------------------------------------------------------------- #
X_l = 0; X_r = 1
L = X_r - X_l
# L = 5000									# Length of pipe (metres)
nx = 101									# Number of spatial cells
nt = 250									# Number of time solutions
ode_nt = 50									# Number of time solutions within the ODE solver
dx = L / (nx - 1)							# Space increment
k = 1e-09									# Compressibility parameter
cfl = 0.8									# Courant-Friedrichs-Levy number
m = 2										# Dimension of solution space
rho_ref = 1000								# Reference density
p_ref = 1e+05								# Reference pressure
gravity = 9.8								# Gravity
mid = int((nx + 1) / 2)						# Middle spatial cell
xs = np.linspace(0, L, nx)					# Spatial mesh for plotting
Q = np.empty((nt, m, nx + 2))				# Array of each time solution
speeds = np.empty(nx + 1)					# Array to contain the speeds to compute the time increment
fluctuations = np.empty((nx + 1, m, m))		# Array to contain the right and left fluctuations on each cell interface
dts = np.empty(nt - 1)						# Contains each computed time increment
rho = np.empty((nt, nx))					# Density solution array
u = np.empty((nt, nx))						# Velocity solution array
p = np.empty((nt, nx))						# Pressure solution array


# ------------------------------------------------------------------------------------------------------------------- #
#
# Initial conditions
#
# ------------------------------------------------------------------------------------------------------------------- #
p[0] = 2e+05 * np.ones(nx)
rho[0] = rho_ref + k * (p[0] - p_ref)
u[0] = np.zeros(nx)
Q[0, 0, 1 : nx + 1] = rho[0]
Q[0, 1, 1 : nx + 1] = rho[0] * u[0]
hlle = HLLE(L, nx, k)


# ------------------------------------------------------------------------------------------------------------------- #
#
# Boundary conditions
#
# ------------------------------------------------------------------------------------------------------------------- #
# uL = -5e+03
uL = 5e+03
pR = 2e+05
rhoR = rho_ref + k * (pR - p_ref)


# ------------------------------------------------------------------------------------------------------------------- #
#
# Auxiliary functions
#
# ------------------------------------------------------------------------------------------------------------------- #
def psi(t, q, j):
	"""
	Source term function.
	"""
	if j <= mid:
		return np.array([0, gravity * q[0]])
	return -np.array([0, gravity * q[0]])

def runge_kutta(Q, t, dt, j):
	K1 = dt * psi(t, Q, j)
	K2 = dt * psi(t + 0.5 * dt, Q + 0.5 * K1, j)
	K3 = dt * psi(t + 0.5 * dt, Q + 0.5 * K2, j)
	K4 = dt * psi(t + dt, Q + K3, j)
	return Q + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)


# ------------------------------------------------------------------------------------------------------------------- #
#
# Time iterations
#
# ------------------------------------------------------------------------------------------------------------------- #
for n in range(nt - 1):

	# Compute the inlet and outlet boundary conditions
	Q[n, 1, 1] = rho[n, 0] * uL
	Q[n, 0, nx] = rhoR
	Q[n, 1, nx] = rhoR * u[n, -1]
	
	# Assign the ghost cells
	Q[n, :, 0] = Q[n, :, 1]
	Q[n, :, -1] = Q[n, :, -2]

	for j in range(1, nx + 2):

		# Assign the left and right data for the Riemann problem
		Q_l, Q_r = Q[n, :, j - 1], Q[n, :, j]

		# Compute the eigenvalues of the left and right Jacobian matrices and the Roe matrix centered at the interface
		w_jac_l = hlle.jac_eig(Q_l)
		w_jac_r = hlle.jac_eig(Q_r)
		w_roe = hlle.roe_eig(Q_l, Q_r)

		# Estimate the low and high speeds based on these eigenvalues
		s1 = hlle.estimate_low_speed(w_jac_l, w_roe)
		s2 = hlle.estimate_high_speed(w_jac_r, w_roe)

		# Store the relevant cfl speed data
		speeds[j - 1] = np.max(np.abs((s1, s2)))

		# Estimate the fluctuations based on the Riemann data and the speeds
		apdq, amdq = hlle.estimate_fluctuations(Q_l, Q_r, s1, s2)
		fluctuations[j - 1, 0], fluctuations[j - 1, 1] = apdq, amdq

	# Compute the time increment based on the cfl condition
	dts[n] = cfl * dx / np.max(speeds)
	ode_dt = dts[n] / (nt - 1)

	for j in range(1, nx + 1):

		# Assign the right-going fluctuations on the left interface of cell j
		apdq = fluctuations[j - 1, 0]

		# Assign the left-going fluctuations on the right interface of cell j
		amdq = fluctuations[j, 1]

		# Apply the conservative formula to find the homogenous problem solution
		Q_star = hlle.solve(Q[n, :, j], apdq, amdq, dts[n])
		# Q[n + 1, :, j] = hlle.solve(Q[n, :, j], apdq, amdq, dts[n])
	
		# Solve the ODE problem with the PDE solution as the initial condition
		for m in range(ode_nt):
			Q_star = runge_kutta(Q_star, m * ode_dt, ode_dt, j)
		Q[n + 1, :, j] = Q_star

	# Compute the primitive variable data
	rho[n + 1] = Q[n + 1, 0, 1 : nx + 1]
	u[n + 1] = Q[n + 1, 1, 1 : nx + 1] / rho[n + 1]
	p[n + 1] = (rho[n + 1] - rho_ref) / k + p_ref

	rho[n + 1, -1] = rho_ref + k * (pR - p_ref)
	u[n + 1, 0] = uL
	p[n + 1, -1] = pR


# ------------------------------------------------------------------------------------------------------------------- #
#
# Plotting
#
# ------------------------------------------------------------------------------------------------------------------- #
plot = True
# plot = False

if plot:
	fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

	rho_data, = axs[0].plot(xs, rho[0])
	u_data, = axs[1].plot(xs, u[0])
	p_data, = axs[2].plot(xs, p[0])

	axs[0].set(ylim=(np.min(rho) - 0.1, np.max(rho) + 0.1))
	axs[1].set(ylim=(np.min(u) - 10, np.max(u) + 10))
	axs[2].set(ylim=(np.min(p) - 0.1, np.max(p) + 0.1))

	axs[0].set_title(r"$\rho$")
	axs[1].set_title(r"$u$")
	axs[2].set_title(r"$p$")
	plt.subplots_adjust(hspace=0.4)

	def update(n):
		rho_data.set_data(xs, rho[n])
		u_data.set_data(xs, u[n])
		p_data.set_data(xs, p[n])
		plt.suptitle(n * dts[n - 1])
		return rho_data, u_data, p_data,

	ani = animation.FuncAnimation(fig, func=update, frames=range(1, nt, 2))
	plt.show()
































