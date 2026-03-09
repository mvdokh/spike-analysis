import matplotlib.pyplot as plt
import numpy as np

# Set the bifurcation point
l0 = 0

# Create the parameter range 'l'
l_before = np.linspace(-4, l0, 100)
l_after = np.linspace(l0, 4, 100)
l_full = np.concatenate([l_before, l_after])

# --- Plotting ---
plt.figure(figsize=(10, 6))

# 1. Before l0 (l < l0): One stable attractor
# We'll represent this attractor at X = 0
X_stable_before = np.zeros_like(l_before)
plt.plot(l_before, X_stable_before, 'b-', label='Stable Attractor', linewidth=2.5)

# 2. After l0 (l >= l0): Two stable attractors and one unstable
# The original attractor at X=0 becomes unstable
X_unstable = np.zeros_like(l_after)
plt.plot(l_after, X_unstable, 'r--', label='Unstable Point', linewidth=2.5)

# The two new stable attractors appear (e.g., X = +/- sqrt(l - l0))
X_stable_plus = np.sqrt(l_after - l0)
X_stable_minus = -np.sqrt(l_after - l0)
plt.plot(l_after, X_stable_plus, 'b-', linewidth=2.5)
plt.plot(l_after, X_stable_minus, 'b-', linewidth=2.5)

# --- Styling ---
# Add the critical point line
plt.axvline(x=l0, color='black', linestyle=':', linewidth=2, label=f'Bifurcation Point (l0 = {l0})')

# Labels and Title
plt.title('Conceptual Diagram of a Bifurcation', fontsize=16)
plt.xlabel('Parameter (l)', fontsize=14)
plt.ylabel('System State (X)', fontsize=14)

# Add text annotations
plt.text(-2.5, 2.0, 'One Attractor\n(System has one stable state)', 
         horizontalalignment='center', fontsize=12, color='blue')
plt.text(2.5, 2.0, 'Two Attractors\n(System has two stable states)', 
         horizontalalignment='center', fontsize=12, color='blue')

# Add arrows to show the "flow"
plt.annotate('', xy=(-1, 0.1), xytext=(-1, 1.5),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=8))
plt.annotate('', xy=(-1, -0.1), xytext=(-1, -1.5),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=8))
plt.annotate('', xy=(3, 1.73), xytext=(3, 2.5),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=8))
plt.annotate('', xy=(3, -1.73), xytext=(3, -2.5),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=8))


# Grid and layout
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='lower left')
plt.axhline(0, color='black', alpha=0.5, lw=0.5) # Add x-axis line
plt.tight_layout()

# Save the figure
plt.savefig('bifurcation_diagram.png')