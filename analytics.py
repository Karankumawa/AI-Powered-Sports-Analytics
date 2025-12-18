import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import csv

class AnalyticsEngine:
    def __init__(self):
        pass

    def generate_heatmap(self, tracks):
        """
        Generates a heatmap figure.
        Args:
            tracks: List of [x, y] coordinates.
        Returns:
            fig: Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        if not tracks:
            ax.text(0.5, 0.5, "No Data", ha='center')
            return fig

        x = [t[0] for t in tracks]
        y = [t[1] for t in tracks]

        # Pitch dimensions 105x68
        h = ax.hist2d(x, y, bins=[50, 30], range=[[0, 105], [0, 68]], cmap='hot', alpha=0.75)
        fig.colorbar(h[3], ax=ax, label='Frequency')
        ax.set_title('Player Heatmap')
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Width (m)')
        ax.invert_yaxis()
        return fig

    def plot_convex_hull(self, team_tracks):
        """
        Returns convex hull figure.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_title('Team Convex Hull')
        ax.invert_yaxis()
        
        points = np.array(team_tracks)
        if len(points) < 3:
            ax.text(0.5, 0.5, "Not enough points (<3)", ha='center')
            return fig
            
        # Plot points
        ax.plot(points[:,0], points[:,1], 'o', color='blue')
        
        try:
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], 'r-', lw=2)
            ax.fill(points[hull.vertices,0], points[hull.vertices,1], 'b', alpha=0.1)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center')
            
        return fig

    def get_csv_data(self, all_tracks):
        """
        Prepares CSV string data.
        """
        output = [["frame", "id", "x", "y"]]
        for record in all_tracks:
            output.append(record)
        return output
