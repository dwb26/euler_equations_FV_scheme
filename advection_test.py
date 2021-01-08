import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ------------------------------------------------------------------------------------------------------------------- #
#
# Parameters
#
# ------------------------------------------------------------------------------------------------------------------- #
X_l = -25
X_r = 25
L = X_r - X_l
nx = 751
nt = 200
dx = L / (nx - 1)
xs = np.linspace(X_l, X_r, nx)
Q = np.empty((nt, nx + 2))
Q_true = np.empty((nt, nx))
a = 10
lmbda = -1
cfl = 0.8
dt = cfl * dx / a
fluxes = np.empty(nx + 1)


# ------------------------------------------------------------------------------------------------------------------- #
#
# Initial conditions
#
# ------------------------------------------------------------------------------------------------------------------- #
Q[0, 1 : nx + 1] = np.exp(-xs ** 2 / 0.5)
Q_true[0] = np.exp(-xs ** 2 / 0.5)


# ------------------------------------------------------------------------------------------------------------------- #
#
# Auxiliary functions
#
# ------------------------------------------------------------------------------------------------------------------- #
def source(t, q):
	return lmbda * q

def runge_kutta(Q, t, dt):
	K1 = dt * source(t, Q)
	K2 = dt * source(t + 0.5 * dt, Q + 0.5 * K1)
	K3 = dt * source(t + 0.5 * dt, Q + 0.5 * K2)
	K4 = dt * source(t + dt, Q + K3)
	return Q + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)

def forward_euler(Q, t, dt):
	if (1 + dt * lmbda) > 1:
		raise ValueError("Not stable {}".format(1 + dt * lmbda))
	return Q + dt * source(t, Q)

def f(q):
	return a * q

def godunov(Q_l, Q_r, a):
	if a > 0:
		return f(Q_l)
	return f(Q_r)

def true_solution(xs, t):
	return np.exp(-(xs - a * t) ** 2 / 0.5) * np.exp(lmbda * t)


# ------------------------------------------------------------------------------------------------------------------- #
#
# Time iterations
#
# ------------------------------------------------------------------------------------------------------------------- #
for n in range(nt - 1):

	# Set the values of the ghost cells
	Q[n, 0] = Q[n, 1]
	Q[n, -1] = Q[n, -2]

	# Solve the homogeneous problem
	for j in range(1, nx + 2):
		Q_l = Q[n, j - 1]; Q_r = Q[n, j]
		fluxes[j - 1] = godunov(Q_l, Q_r, a)
	for j in range(1, nx + 1):
		Q_star = Q[n, j] - dt / dx * (fluxes[j] - fluxes[j - 1])

		# Use the solution as the initial conditions for the ODE solver
		dt_c = dt / (nt - 1)
		for m in range(nt):
			Q_star = runge_kutta(Q_star, m * dt_c, dt_c)
		Q[n + 1, j] = Q_star
		
	Q_true[n + 1] = true_solution(xs, (n + 1) * dt)

Q = Q[:, 1 : nx + 1]


# ------------------------------------------------------------------------------------------------------------------- #
#
# Plotting
#
# ------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot(111)

Q_data, = ax.plot(xs, Q[0])
Q_true_data, = ax.plot(xs, Q_true[0])

def update(k):
	Q_data.set_data(xs, Q[k])
	Q_true_data.set_data(xs, Q_true[k])
	plt.suptitle(k * dt)
	return Q_data, Q_true_data,

ani = animation.FuncAnimation(fig, func=update, frames=range(1, nt, 5))
plt.show()




































