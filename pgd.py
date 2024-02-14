import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PGD:

    def __init__(self, f, subgrad_f, x0=1.0, x_star=0.0, R=2.5, T=15):
        self.f = f
        self.subgrad_f = subgrad_f

        # default domain is -R/2 <= x <= R/2
        self.R = R
        assert np.abs(x0) <= R / 2, "x0 is not in the domain of f(x)"

        self.T = T
        self.points = np.array([[x0, f(x0)]])
        self.eta = []
        self.optimum = np.array([[x0, f(x0)]])

        # about errors
        self.x_star = x_star
        self.upper_bounds = [self.f(x0) - self.f(x_star)]
        self.errors = [self.f(x0) - self.f(x_star)]

    def project(self, x):
        return np.clip(x, -self.R / 2, self.R / 2)

    def perform_pgd(self):
        # prime
        x = self.points[-1][0]

        for t in range(1, self.T + 1):
            eta_t = self.compute_eta(t)
            self.eta.append(eta_t)
            x = self.project(x - eta_t * self.subgrad_f(x))
            self.points = np.append(self.points, [[x, self.f(x)]], axis=0)

            new_optimum = self.compute_optimum(self.points)
            self.optimum = np.append(
                self.optimum, [[new_optimum, self.f(new_optimum)]], axis=0
            )

            self.upper_bounds.append(self.compute_upper_bound(t))
            self.errors.append(self.f(new_optimum) - self.f(self.x_star))

    def animate(self):
        self.perform_pgd()

        x_grid = np.linspace(-self.R / 2, self.R / 2, 100)
        f_grid = np.vectorize(self.f)(x_grid)

        # prime the figure
        fig, (ax1, ax2) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(10, 5)
        )
        ax1.plot(x_grid, f_grid)
        ax1.axis("equal")
        scat1 = ax1.scatter(
            [self.points[0, 0]], [self.points[0, 1]], c="g", label="points"
        )  # alpha=0.5
        scat2 = ax1.scatter(
            [self.optimum[0, 0]],
            [self.optimum[0, 1]],
            marker="x",
            c="r",
            label="optima",
        )  # alpha=0.5,
        ax1.legend()

        bar1 = ax2.bar(
            0,
            self.upper_bounds[0],
            1,
            color="tab:blue",
            fill=False,
            label="upper bound",
        )
        bar2 = ax2.bar(0, self.errors[0], 1, color="g", label="optimum error")
        ax2.set_ylim(0, 1.1 * max(self.points[:, 1]))
        ax2.legend()

        # remove spines
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        fig.suptitle("Projected Gradient Descent")

        # animate
        def animate(i):
            scat1.set_offsets(self.points[:i])
            scat2.set_offsets(self.optimum[:i])

            for rect1, rect2 in zip(bar1, bar2):
                rect1.set_height(self.upper_bounds[i])
                rect2.set_height(self.errors[i])
                fig.canvas.draw()

        ani = animation.FuncAnimation(fig, animate, frames=self.T)
        return ani

    def compute_upper_bound(self, t):
        raise NotImplementedError

    def compute_eta(self, t):
        raise NotImplementedError

    def compute_optimum(self, points):
        raise NotImplementedError
