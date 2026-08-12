import matplotlib.pyplot as plt
from simulations.plot_utils import compute_flow_field, plot_flow_field, plot_speed
from simulations.ring_attractor import RingAttractorSimulation


def main():
    model = RingAttractorSimulation()
    fig, ax = plt.subplots(figsize=(6, 6))
    grid, arrows = plot_flow_field(ax, model, -1.25, 1.25, 0.05)

    grid, flows, axes = compute_flow_field(model, -1.25, 1.25, 1 / 100)
    plot_speed(ax, grid, flows, vmin=0, vmax=0.075, alpha=0.5)

    plt.show()


if __name__ == "__main__":
    main()
