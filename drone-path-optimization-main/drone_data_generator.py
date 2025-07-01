import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os

def generate_drone_training_data(num_samples=500, filename='drone_training_data.csv'):
    """
    Generate synthetic drone flight data for training the reinforcement learning model.
    
    Parameters:
    - num_samples: Number of training samples to generate
    - filename: Name of the output CSV file
    
    Returns:
    - DataFrame with the generated data
    """
    # Initialize lists to store data
    data = []
    
    # Define space boundaries (assuming a 100x100x100 space)
    space_boundaries = [100, 100, 100]
    
    # Generate random source and destination points
    for i in range(num_samples):
        # Source coordinates (x, y, z)
        source_x = random.uniform(0, space_boundaries[0])
        source_y = random.uniform(0, space_boundaries[1])
        source_z = random.uniform(0, space_boundaries[2])
        
        # Destination coordinates (x, y, z)
        dest_x = random.uniform(0, space_boundaries[0])
        dest_y = random.uniform(0, space_boundaries[1])
        dest_z = random.uniform(0, space_boundaries[2])
        
        # Battery level (between 60% and 100%)
        battery_level = random.uniform(60, 100)
        
        # Drone speed capability (in m/s, between 2 and 10)
        drone_speed = random.uniform(2, 10)
        
        # Calculate direct distance between source and destination
        distance = np.sqrt((dest_x - source_x)**2 + (dest_y - source_y)**2 + (dest_z - source_z)**2)
        
        # Add occasional obstacles
        obstacles = []
        if random.random() < 0.3:  # 30% chance to have obstacles
            num_obstacles = random.randint(1, 3)
            for j in range(num_obstacles):
                # Random obstacle position between source and destination
                t = random.uniform(0.2, 0.8)  # Position along path (0.2-0.8 to avoid being too close to start/end)
                obs_x = source_x + t * (dest_x - source_x) + random.uniform(-15, 15)
                obs_y = source_y + t * (dest_y - source_y) + random.uniform(-15, 15)
                obs_z = source_z + t * (dest_z - source_z) + random.uniform(-15, 15)
                
                # Random radius between 2 and 8
                radius = random.uniform(2, 8)
                
                # Add to obstacles list
                obstacles.append(f"{obs_x:.2f},{obs_y:.2f},{obs_z:.2f},{radius:.2f}")
        
        obstacles_str = ";".join(obstacles)
        
        # Calculate optimal path (straight line if no obstacles)
        if not obstacles:
            path_length = distance
        else:
            # Simplified model: path is longer if there are obstacles
            path_length = distance * random.uniform(1.1, 1.5)
        
        # Calculate estimated battery consumption (simplified model)
        # More battery used for longer paths and higher speeds
        battery_consumption = (path_length / 100) * (drone_speed / 5) * random.uniform(10, 20)
        
        # Calculate estimated flight time
        flight_time = path_length / drone_speed
        
        # Add to data list
        data.append({
            'source_x': source_x,
            'source_y': source_y,
            'source_z': source_z,
            'dest_x': dest_x,
            'dest_y': dest_y,
            'dest_z': dest_z,
            'battery_level': battery_level,
            'drone_speed': drone_speed,
            'distance': distance,
            'path_length': path_length,
            'obstacles': obstacles_str,
            'battery_consumption': battery_consumption,
            'flight_time': flight_time
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"Generated {num_samples} drone flight training samples and saved to {filename}")
    
    return df

def visualize_drone_data(df, num_samples=5):
    """
    Visualize some random samples from the generated drone data
    
    Parameters:
    - df: DataFrame with drone data
    - num_samples: Number of random samples to visualize
    """
    # Select random samples
    sample_indices = random.sample(range(len(df)), min(num_samples, len(df)))
    
    for idx in sample_indices:
        row = df.iloc[idx]
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Source and destination points
        source = [row['source_x'], row['source_y'], row['source_z']]
        dest = [row['dest_x'], row['dest_y'], row['dest_z']]
        
        # Plot source and destination
        ax.scatter(source[0], source[1], source[2], c='g', marker='o', s=100, label='Source')
        ax.scatter(dest[0], dest[1], dest[2], c='r', marker='x', s=100, label='Destination')
        
        # Plot direct path
        ax.plot([source[0], dest[0]], [source[1], dest[1]], [source[2], dest[2]], 'b--', label='Direct Path')
        
        # Plot obstacles if present
        if row['obstacles']:
            obstacles = row['obstacles'].split(';')
            for obs_str in obstacles:
                if obs_str:
                    x, y, z, r = map(float, obs_str.split(','))
                    ax.scatter(x, y, z, c='k', marker='s', s=r*20, label='Obstacle')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Drone Flight Data - Sample {idx}\n'
                    f'Battery: {row["battery_level"]:.1f}%, Speed: {row["drone_speed"]:.1f} m/s\n'
                    f'Distance: {row["distance"]:.1f} m, Est. Flight Time: {row["flight_time"]:.1f} s')
        
        # Set legend (only show one instance of each label)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Generate data
    df = generate_drone_training_data(num_samples=500)
    
    # Visualize some random samples
    visualize_drone_data(df, num_samples=3)
