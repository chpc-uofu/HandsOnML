import numpy as np

def make_spirals(n_samples=4000, noise=0.25, turns=3, seed=0):

    rng = np.random.default_rng(seed)
    n = n_samples // 2
    theta = np.linspace(0, turns*2*np.pi, n)
    r = theta
    x1 = r * np.cos(theta) + rng.normal(0, noise, size=n)
    y1 = r * np.sin(theta) + rng.normal(0, noise, size=n)
    X1 = np.column_stack([x1, y1])
    y1_label = np.zeros(n, dtype=int)

    # Second spiral: phase shifted by π
    theta2 = theta + np.pi
    x2 = r * np.cos(theta2) + rng.normal(0, noise, size=n)
    y2 = r * np.sin(theta2) + rng.normal(0, noise, size=n)
    X2 = np.column_stack([x2, y2])
    y2_label = np.ones(n, dtype=int)

    X = np.vstack([X1, X2])
    y = np.hstack([y1_label, y2_label])
    return X, y
