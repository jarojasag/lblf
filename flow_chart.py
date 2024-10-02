import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Define box positions with increased spacing
boxes = {
    'General Population': (0.05, 0.75),
    'Elites': (0.05, 0.45),
    'Radicalization\nDynamics': (0.35, 0.65),
    'Relative Income/\nWages': (0.35, 0.45),
    'Elite\nCompetition': (0.35, 0.25),
    'PSI (Political Stress\nIndex)': (0.65, 0.55),
    'Sociopolitical\nInstability': (0.65, 0.35),
    'Moderates': (0.65, 0.15)
}

# Draw boxes sized to just cover the text
for box, (x, y) in boxes.items():
    rect = patches.FancyBboxPatch((x, y), 0.02, 0.002, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightgrey')
    ax.add_patch(rect)
    ax.text(x, y, box, ha='center', va='center', fontsize=10)

# Define arrows
arrows = [
    ('General Population', 'Radicalization\nDynamics'),
    ('Elites', 'Radicalization\nDynamics'),
    ('Radicalization\nDynamics', 'PSI (Political Stress\nIndex)'),
    ('Relative Income/\nWages', 'Radicalization\nDynamics'),
    ('Relative Income/\nWages', 'PSI (Political Stress\nIndex)'),
    ('Elite\nCompetition', 'PSI (Political Stress\nIndex)'),
    ('PSI (Political Stress\nIndex)', 'Sociopolitical\nInstability'),
    ('Moderates', 'Sociopolitical\nInstability')
]

# Draw arrows
for start, end in arrows:
    start_pos = boxes[start]
    end_pos = boxes[end]
    ax.annotate('', xy=(end_pos[0] + 0.125, end_pos[1] + 0.05), xytext=(start_pos[0] + 0.125, start_pos[1] + 0.05),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=2))

# Draw resource distribution arrows
ax.annotate('', xy=(0.175, 0.575), xytext=(0.175, 0.625), arrowprops=dict(facecolor='black', arrowstyle='<->', lw=2))
ax.text(0.025, 0.6, 'Resource\nDistribution', ha='center', va='center', rotation=90, fontsize=10)

# Set plot limits and remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.title("System Dynamics Representation of the MPF Model")
plt.show()
