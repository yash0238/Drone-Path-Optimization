import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import random
from collections import deque
import os
import time

# Try to use GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {len(gpus)} GPU(s) available")
    else:
        print("No GPU available, using CPU")
except Exception as e:
    print(f"GPU initialization error: {e}")
    print("Using CPU for training")

class DroneEnvironment(gym.Env):
    """
    Custom Environment for DJI Tello drone path planning using reinforcement learning
    """
    
    def __init__(self, start_position, target_position, initial_battery=100, max_speed=8.0, 
                 battery_penalty_factor=0.1, time_penalty_factor=0.01, collision_penalty=100):
        super(DroneEnvironment, self).__init__()
        
        # Define action and observation space
        # Actions: dx, dy, dz (changes in position) and speed
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0]),  # min dx, dy, dz, speed
            high=np.array([1, 1, 1, 1]),    # max dx, dy, dz, speed (normalized)
            dtype=np.float32
        )
        
        # Observations: current position (x,y,z), target position (x,y,z), battery level, speed
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([100, 100, 100, 100, 100, 100, 100, 1]),
            dtype=np.float32
        )
        
        # Environment parameters
        self.start_position = np.array(start_position, dtype=np.float32)
        self.target_position = np.array(target_position, dtype=np.float32)
        self.current_position = self.start_position.copy()
        self.initial_battery = initial_battery
        self.battery_level = initial_battery
        self.max_speed = max_speed  # Maximum speed in m/s
        self.current_speed = 0.0
        
        # Penalty factors
        self.battery_penalty_factor = battery_penalty_factor
        self.time_penalty_factor = time_penalty_factor
        self.collision_penalty = collision_penalty
        
        # Environment boundaries (assuming a 100x100x100 space)
        self.boundaries = np.array([100, 100, 100])
        
        # Obstacles (can be defined based on your environment)
        self.obstacles = []
        
        # Path history for visualization
        self.path_history = [self.current_position.copy()]
        
        # Steps counter
        self.steps = 0
        self.max_steps = 1000
        
    def _get_observation(self):
        # Normalize current position, target position, battery, and speed
        normalized_current = self.current_position / self.boundaries
        normalized_target = self.target_position / self.boundaries
        normalized_battery = self.battery_level / self.initial_battery
        normalized_speed = self.current_speed / self.max_speed
        
        return np.concatenate([
            normalized_current, normalized_target, 
            [normalized_battery, normalized_speed]
        ])
    
    def _calculate_reward(self):
        # Calculate distance to target
        distance = np.linalg.norm(self.current_position - self.target_position)
        
        # Base reward: negative distance (closer = better)
        reward = -distance * 0.1
        
        # Check if target reached (within a threshold)
        if distance < 2.0:  # 2 meters threshold
            reward += 100  # Bonus for reaching target
        
        # Penalty for battery usage
        reward -= (self.initial_battery - self.battery_level) * self.battery_penalty_factor
        
        # Penalty for time (steps taken)
        reward -= self.steps * self.time_penalty_factor
        
        # Check for collisions with boundaries
        if (self.current_position < 0).any() or (self.current_position > self.boundaries).any():
            reward -= self.collision_penalty
            return reward, True  # Collision happened, terminate episode
        
        # Check for collisions with obstacles
        for obstacle in self.obstacles:
            # Simple spherical obstacle check
            center, radius = obstacle
            if np.linalg.norm(self.current_position - center) < radius:
                reward -= self.collision_penalty
                return reward, True  # Collision happened, terminate episode
        
        # Check if battery depleted
        if self.battery_level <= 0:
            reward -= 50  # Penalty for running out of battery
            return reward, True  # Battery depleted, terminate episode
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            return reward, True  # Max steps reached, terminate episode
        
        return reward, False
    
    def step(self, action):
        self.steps += 1
        
        # Extract and denormalize actions
        dx, dy, dz, speed_factor = action
        
        # Update speed (normalized action to real speed)
        self.current_speed = speed_factor * self.max_speed
        
        # Calculate movement based on direction and speed
        movement = np.array([dx, dy, dz], dtype=np.float32)
        movement = movement / np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else movement
        movement *= self.current_speed
        
        # Update position
        self.current_position += movement
        
        # Add to path history
        self.path_history.append(self.current_position.copy())
        
        # Update battery level based on movement and speed
        # Simple model: battery decreases more with higher speed and distance
        battery_consumption = 0.1 * np.linalg.norm(movement) * (0.5 + 0.5 * speed_factor)
        self.battery_level -= battery_consumption
        
        # Calculate reward and check termination
        reward, done = self._calculate_reward()
        
        return self._get_observation(), reward, done, {}
    
    def reset(self):
        self.current_position = self.start_position.copy()
        self.battery_level = self.initial_battery
        self.current_speed = 0.0
        self.steps = 0
        self.path_history = [self.current_position.copy()]
        return self._get_observation()
    
    def render(self, mode='human'):
        if mode == 'human':
            # 3D visualization of the drone path
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot path history
            path = np.array(self.path_history)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', label='Drone Path')
            
            # Plot start and target
            ax.scatter(self.start_position[0], self.start_position[1], self.start_position[2], 
                      c='g', marker='o', s=100, label='Start')
            ax.scatter(self.target_position[0], self.target_position[1], self.target_position[2], 
                      c='r', marker='x', s=100, label='Target')
            
            # Plot current position
            ax.scatter(self.current_position[0], self.current_position[1], self.current_position[2], 
                      c='b', marker='^', s=100, label='Current')
            
            # Plot obstacles
            for center, radius in self.obstacles:
                # This is a simplified representation of obstacles
                ax.scatter(center[0], center[1], center[2], c='k', marker='s', s=radius*10, label='Obstacle')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Drone Flight Path\nBattery: {self.battery_level:.1f}%, Speed: {self.current_speed:.1f} m/s')
            
            # Show legend with unique labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            plt.show()


class OUActionNoise:
    """Ornstein-Uhlenbeck process noise for exploration"""
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        
    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    
    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for continuous control
    """
    def __init__(self, state_dim, action_dim, action_high):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99  # discount factor
        self.tau = 0.005   # target network update rate
        
        # Actor model (policy)
        self.actor = self._build_actor()
        self.target_actor = self._build_actor()
        self.target_actor.set_weights(self.actor.get_weights())
        
        # Critic model (value)
        self.critic = self._build_critic()
        self.target_critic = self._build_critic()
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Learning rate scheduler
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.95
        )
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Noise process for exploration
        self.noise = OUActionNoise(mean=np.zeros(action_dim), std_deviation=0.2 * action_high)
        
        # Training metrics
        self.actor_loss_history = []
        self.critic_loss_history = []
        
    def _build_actor(self):
        """Actor (Policy) Model"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(512, activation="relu")(inputs)
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(self.action_dim, activation="tanh")(x)
        # Scale outputs to action space
        outputs = outputs * self.action_high
        model = Model(inputs, outputs)
        return model
    
    def _build_critic(self):
        """Critic (Value) Model"""
        # State input
        state_input = Input(shape=(self.state_dim,))
        state_out = Dense(512, activation="relu")(state_input)
        state_out = Dense(256, activation="relu")(state_out)
        
        # Action input
        action_input = Input(shape=(self.action_dim,))
        action_out = Dense(256, activation="relu")(action_input)
        
        # Combine state and action pathways
        concat = Concatenate()([state_out, action_out])
        x = Dense(256, activation="relu")(concat)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(1)(x)  # Q-value
        
        model = Model([state_input, action_input], outputs)
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy"""
        state = np.reshape(state, [1, self.state_dim])
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        action = self.actor(state_tensor).numpy()[0]
        if add_noise:
            action += self.noise()
        # Clip action to be within bounds
        return np.clip(action, -self.action_high, self.action_high)
    
    def train(self):
        """Updates actor and critic networks from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            state, action, reward, next_state, done = self.memory[i]
            states.append(np.reshape(state, self.state_dim))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.reshape(next_state, self.state_dim))
            dones.append(float(done))
            
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Train critic
        with tf.GradientTape() as tape:
            # Get target actions and Q values
            target_actions = self.target_actor(next_states_tensor)
            target_q_values = self.target_critic([next_states_tensor, target_actions])
            
            # Calculate critic target
            critic_target = rewards_tensor + self.gamma * target_q_values * (1 - dones_tensor)
            
            # Get current Q values
            current_q_values = self.critic([states_tensor, actions_tensor])
            
            # Calculate critic loss
            critic_loss = tf.reduce_mean(tf.square(critic_target - current_q_values))
        
        # Get critic gradients and update
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Record critic loss
        self.critic_loss_history.append(critic_loss.numpy())
        
        # Train actor
        with tf.GradientTape() as tape:
            # Get actions from actor
            actor_actions = self.actor(states_tensor)
            
            # Calculate actor loss (negative of critic value)
            actor_loss = -tf.reduce_mean(self.critic([states_tensor, actor_actions]))
        
        # Get actor gradients and update
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Record actor loss
        self.actor_loss_history.append(actor_loss.numpy())
        
        # Update target networks
        self._update_target_networks()
    
    def _update_target_networks(self):
        """Soft update target networks"""
        # Actor update
        for target_var, var in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
        
        # Critic update
        for target_var, var in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
    
    def save(self, filepath):
        """Save actor and critic models"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        actor_path = filepath + "_actor.weights.h5"
        critic_path = filepath + "_critic.weights.h5"
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
        print(f"Model saved to {filepath}")
        
    def load(self, filepath):
        """Load actor and critic models"""
        try:
            actor_path = filepath + "_actor.weights.h5"
            critic_path = filepath + "_critic.weights.h5"
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            self.target_actor.load_weights(actor_path)
            self.target_critic.load_weights(critic_path)
            print(f"Model loaded from {filepath}")
            return True
        except:
            print(f"Failed to load model from {filepath}")
            return False


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


def process_drone_data(csv_file):
    """
    Process drone flight data from CSV file
    Expected CSV format:
    - Contains source and destination coordinates (x, y, z)
    - Battery level
    - Drone speed
    - May contain obstacle information
    
    Returns:
    - List of environment instances for training
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Initialize environments list
        environments = []
        
        # Process each row in the CSV
        for idx, row in df.iterrows():
            # Extract source and destination coordinates
            try:
                source = [row['source_x'], row['source_y'], row['source_z']]
                destination = [row['dest_x'], row['dest_y'], row['dest_z']]
                battery = row.get('battery_level', 100)  # Default to 100 if not present
                speed = row.get('drone_speed', 5.0)      # Default to 5.0 if not present
                
                # Create environment
                env = DroneEnvironment(
                    start_position=source,
                    target_position=destination,
                    initial_battery=battery,
                    max_speed=speed
                )
                
                # Add obstacles if present in the data
                if 'obstacles' in row and isinstance(row['obstacles'], str) and row['obstacles']:
                    # Parse obstacles from string format
                    # Assuming format like "x1,y1,z1,r1;x2,y2,z2,r2;..."
                    obstacle_strs = row['obstacles'].split(';')
                    for obs_str in obstacle_strs:
                        if obs_str:
                            x, y, z, r = map(float, obs_str.split(','))
                            env.obstacles.append(([x, y, z], r))
                
                environments.append(env)
                
            except KeyError as e:
                print(f"Missing required column in row {idx}: {e}")
                continue
                
        return environments
    
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return []


def train_drone_agent(csv_file, episodes=1000, max_steps=500, save_path="models/drone_ddpg"):
    """
    Train a DDPG agent for drone path planning using data from CSV
    
    Parameters:
    - csv_file: Path to CSV file with drone data
    - episodes: Number of training episodes
    - max_steps: Maximum steps per episode
    - save_path: Path to save trained model
    
    Returns:
    - Trained agent
    """
    print("Loading and processing drone data...")
    environments = process_drone_data(csv_file)
    
    if not environments:
        print("No valid environments created from CSV. Check data format.")
        return None
    
    print(f"Created {len(environments)} training environments from CSV data")
    
    # Use first environment for agent initialization
    env = environments[0]
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    
    print(f"Initializing DDPG agent with state_dim={state_dim}, action_dim={action_dim}")
    agent = DDPGAgent(state_dim, action_dim, action_high)
    
    # Training metrics
    rewards_history = []
    avg_rewards_history = []
    success_rate_history = []
    steps_per_episode = []
    
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(episodes):
        # Randomly select an environment for this episode
        env_idx = np.random.randint(0, len(environments))
        env = environments[env_idx]
        
        # Reset environment
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            # Update current state and accumulate reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Save training metrics
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
        avg_rewards_history.append(avg_reward)
        steps_per_episode.append(step + 1)
        
        # Calculate success rate (reached destination within threshold)
        distance_to_target = np.linalg.norm(env.current_position - env.target_position)
        success = distance_to_target < 2.0
        success_rate = np.mean([1 if success else 0] + [s for s in success_rate_history[-99:]] if success_rate_history else [])
        success_rate_history.append(success)
        
        # Print progress
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode: {episode}/{episodes}, Reward: {episode_reward:.2f}, " + 
                  f"Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}, " + 
                  f"Steps: {step+1}, Time: {elapsed_time:.1f}s")
            
            # Save model periodically
            if episode % 100 == 0 and episode > 0:
                agent.save(f"{save_path}_episode_{episode}")
                
                # Visualize path occasionally
                if episode % 200 == 0:
                    env.render()
    
    # Save final trained model
    agent.save(save_path)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    
    # Plot training progress
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.plot(avg_rewards_history)
    plt.title('Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Episode Reward', 'Average Reward'])
    
    plt.subplot(2, 2, 2)
    plt.plot([sum(success_rate_history[:i+1])/(i+1) for i in range(len(success_rate_history))])
    plt.title('Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    plt.subplot(2, 2, 3)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(2, 2, 4)
    if len(agent.actor_loss_history) > 10:  # Only plot if we have data
        plt.plot(agent.actor_loss_history[::10])  # Plot every 10th value to reduce noise
        plt.plot(agent.critic_loss_history[::10])
        plt.title('Loss')
        plt.xlabel('Training Step (x10)')
        plt.ylabel('Loss')
        plt.legend(['Actor Loss', 'Critic Loss'])
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_training_progress.png")
    plt.show()
    
    return agent


def evaluate_agent(agent, env, num_episodes=5):
    """
    Evaluate a trained agent in the given environment
    
    Parameters:
    - agent: Trained DDPG agent
    - env: Environment to evaluate in
    - num_episodes: Number of evaluation episodes
    
    Returns:
    - Dictionary with evaluation metrics
    """
    success_count = 0
    path_lengths = []
    battery_usages = []
    steps_list = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        initial_battery = env.battery_level
        
        while not done and steps < 500:
            action = agent.act(state, add_noise=False)  # No exploration during evaluation
            next_state, reward, done, _ = env.step(action)
            state = next_state
            steps += 1
        
        # Check if target reached
        distance_to_target = np.linalg.norm(env.current_position - env.target_position)
        success = distance_to_target < 2.0
        if success:
            success_count += 1
        
        # Calculate path length
        path = np.array(env.path_history)
        path_length = 0
        for i in range(len(path)-1):
            path_length += np.linalg.norm(path[i+1] - path[i])
        
        path_lengths.append(path_length)
        battery_usages.append(initial_battery - env.battery_level)
        steps_list.append(steps)
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Success: {success}")
        print(f"  Path Length: {path_length:.2f} units")
        print(f"  Battery Used: {initial_battery - env.battery_level:.2f}%")
        print(f"  Steps: {steps}")
        
        # Visualize the path
        env.render()
    
    # Calculate metrics
    success_rate = success_count / num_episodes
    avg_path_length = np.mean(path_lengths)
    avg_battery_usage = np.mean(battery_usages)
    avg_steps = np.mean(steps_list)
    
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Path Length: {avg_path_length:.2f} units")
    print(f"Average Battery Usage: {avg_battery_usage:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    
    return {
        'success_rate': success_rate,
        'avg_path_length': avg_path_length,
        'avg_battery_usage': avg_battery_usage,
        'avg_steps': avg_steps
    }


class DigitalTwin:
    """
    Digital Twin for a drone that interfaces with the reinforcement learning agent
    """
    def __init__(self, agent, env=None):
        self.agent = agent
        
        if env is None:
            # Create a default environment
            self.env = DroneEnvironment(
                start_position=[0, 0, 0],
                target_position=[50, 50, 50],
                initial_battery=100,
                max_speed=5.0
            )
        else:
            self.env = env
            
        self.current_state = None
        self.target_trajectory = []
        
    def initialize(self, start_position, target_position, battery_level=100, max_speed=5.0):
        """
        Initialize the digital twin with real drone data
        
        Parameters:
        - start_position: [x, y, z] coordinates of the starting position
        - target_position: [x, y, z] coordinates of the target position
        - battery_level: Current battery level (%)
        - max_speed: Maximum drone speed (m/s)
        """
        # Reset environment with new parameters
        self.env.start_position = np.array(start_position, dtype=np.float32)
        self.env.target_position = np.array(target_position, dtype=np.float32)
        self.env.initial_battery = battery_level
        self.env.max_speed = max_speed
        
        # Reset environment state
        self.current_state = self.env.reset()
        
        # Reset path history
        self.env.path_history = [self.env.current_position.copy()]
        
        print(f"Digital Twin initialized:")
        print(f"  Start: {start_position}")
        print(f"  Target: {target_position}")
        print(f"  Battery: {battery_level}%")
        print(f"  Max Speed: {max_speed} m/s")
        
        # Calculate and store optimal trajectory
        self.plan_optimal_trajectory()
        
        return self.current_state
    
    def plan_optimal_trajectory(self):
        """Calculate the optimal trajectory using the trained agent"""
        # Create a copy of the current state for planning
        self.target_trajectory = [self.env.current_position.copy()]
        
        # Create a temporary environment for planning
        plan_env = DroneEnvironment(
            start_position=self.env.current_position.copy(),
            target_position=self.env.target_position.copy(),
            initial_battery=self.env.battery_level,
            max_speed=self.env.max_speed
        )
        plan_env.obstacles = self.env.obstacles.copy()
        
        # Plan trajectory
        state = plan_env.reset()
        done = False
        max_steps = 500
        step = 0
        
        while not done and step < max_steps:
            action = self.agent.act(state, add_noise=False)  # No exploration noise
            next_state, _, done, _ = plan_env.step(action)
            state = next_state
            self.target_trajectory.append(plan_env.current_position.copy())
            step += 1
            
            # Check if target reached
            distance = np.linalg.norm(plan_env.current_position - plan_env.target_position)
            if distance < 2.0:  # Same threshold as in reward function
                break
        
        print(f"Planned optimal trajectory with {len(self.target_trajectory)} waypoints")
        return self.target_trajectory
    
    def get_next_action(self):
        """
        Get the next optimal action from the agent based on current state
        
        Returns:
        - action: [dx, dy, dz, speed_factor]
        """
        if self.current_state is None:
            raise ValueError("Digital twin not initialized. Call initialize() first.")
        
        action = self.agent.act(self.current_state, add_noise=False)
        return action
    
    def update_state(self, new_position, new_battery, new_speed):
        """
        Update the digital twin state based on real drone feedback
        
        Parameters:
        - new_position: [x, y, z] coordinates of current position
        - new_battery: Current battery level (%)
        - new_speed: Current speed (m/s)
        
        Returns:
        - Updated state
        """
        # Update environment state
        self.env.current_position = np.array(new_position, dtype=np.float32)
        self.env.battery_level = new_battery
        self.env.current_speed = new_speed
        
        # Add to path history
        self.env.path_history.append(self.env.current_position.copy())
        
        # Get new observation
        self.current_state = self.env._get_observation()
        
        # Check if we need to replan (e.g., significant deviation from planned trajectory)
        if len(self.target_trajectory) > 0:
            current_position = np.array(new_position)
            expected_position = self.target_trajectory[min(len(self.env.path_history)-1, len(self.target_trajectory)-1)]
            
            deviation = np.linalg.norm(current_position - expected_position)
            if deviation > 5.0:  # Threshold for replanning
                print(f"Significant deviation detected ({deviation:.2f} units). Replanning trajectory...")
                self.plan_optimal_trajectory()
        
        return self.current_state
    
    def visualize_state(self, show_trajectory=True):
        """
        Visualize the current state of the digital twin
        
        Parameters:
        - show_trajectory: Whether to show the planned trajectory
        """
        # 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual path
        path = np.array(self.env.path_history)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', linewidth=2, label='Actual Path')
        
        # Plot planned trajectory if requested
        if show_trajectory and len(self.target_trajectory) > 0:
            trajectory = np.array(self.target_trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'g--', linewidth=2, label='Planned Trajectory')
        
        # Plot start and target
        ax.scatter(self.env.start_position[0], self.env.start_position[1], self.env.start_position[2], 
                  c='g', marker='o', s=100, label='Start')
        ax.scatter(self.env.target_position[0], self.env.target_position[1], self.env.target_position[2], 
                  c='r', marker='x', s=100, label='Target')
        
        # Plot current position
        ax.scatter(self.env.current_position[0], self.env.current_position[1], self.env.current_position[2], 
                  c='b', marker='^', s=100, label='Current')
        
        # Plot obstacles
        for center, radius in self.env.obstacles:
            ax.scatter(center[0], center[1], center[2], c='k', marker='s', s=radius*10, label='Obstacle')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Digital Twin State\nBattery: {self.env.battery_level:.1f}%, Speed: {self.env.current_speed:.1f} m/s')
        
        # Show legend with unique labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.show()
    
    def get_estimated_arrival_time(self):
        """
        Estimate time to reach the target based on current state and planned trajectory
        
        Returns:
        - Estimated time in seconds
        """
        if len(self.target_trajectory) == 0:
            return float('inf')
        
        # Calculate remaining distance based on planned trajectory
        remaining_distance = 0
        for i in range(len(self.env.path_history)-1, len(self.target_trajectory)-1):
            if i+1 < len(self.target_trajectory):
                remaining_distance += np.linalg.norm(
                    np.array(self.target_trajectory[i+1]) - np.array(self.target_trajectory[i])
                )
        
        # Estimate time based on current speed
        if self.env.current_speed > 0:
            return remaining_distance / self.env.current_speed
        else:
            return float('inf')
    
    def get_battery_estimate(self):
        """
        Estimate battery level at destination based on current state and planned trajectory
        
        Returns:
        - Estimated battery level (%)
        """
        if len(self.target_trajectory) == 0:
            return 0
        
        # Calculate remaining distance based on planned trajectory
        remaining_distance = 0
        for i in range(len(self.env.path_history)-1, len(self.target_trajectory)-1):
            if i+1 < len(self.target_trajectory):
                remaining_distance += np.linalg.norm(
                    np.array(self.target_trajectory[i+1]) - np.array(self.target_trajectory[i])
                )
        
        # Estimate battery usage (simplified model)
        # This matches the consumption model in DroneEnvironment.step()
        speed_factor = self.env.current_speed / self.env.max_speed
        estimated_consumption = 0.1 * remaining_distance * (0.5 + 0.5 * speed_factor)
        
        return max(0, self.env.battery_level - estimated_consumption)


def simulate_drone_flight(digital_twin, steps=100, noise_level=0.1):
    """
    Simulate a drone flight using the digital twin for path planning
    
    Parameters:
    - digital_twin: Initialized digital twin
    - steps: Maximum steps to simulate
    - noise_level: Level of noise to add to actions (to simulate real-world uncertainties)
    
    Returns:
    - Simulation results
    """
    print("Starting drone flight simulation...")
    
    current_position = digital_twin.env.current_position.copy()
    current_battery = digital_twin.env.battery_level
    current_speed = digital_twin.env.current_speed
    
    for step in range(steps):
        # Get next action from the digital twin
        action = digital_twin.get_next_action()
        
        # Add noise to simulate real-world uncertainties
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=len(action))
            action = np.clip(action + noise, -1, 1)
        
        # Extract action components
        dx, dy, dz, speed_factor = action
        
        # Update speed
        current_speed = speed_factor * digital_twin.env.max_speed
        
        # Calculate movement
        movement = np.array([dx, dy, dz], dtype=np.float32)
        movement = movement / np.linalg.norm(movement) if np.linalg.norm(movement) > 0 else movement
        movement *= current_speed
        
        # Update position (simulating the real drone movement)
        current_position += movement
        
        # Update battery (simulating real battery consumption)
        battery_consumption = 0.1 * np.linalg.norm(movement) * (0.5 + 0.5 * speed_factor)
        current_battery -= battery_consumption
        
        # Update digital twin with new state (as if receiving telemetry from real drone)
        digital_twin.update_state(current_position.copy(), current_battery, current_speed)
        
        # Check if target reached
        distance_to_target = np.linalg.norm(current_position - digital_twin.env.target_position)
        if distance_to_target < 2.0:
            print(f"Target reached in {step+1} steps!")
            break
        
        # Check if battery depleted
        if current_battery <= 0:
            print(f"Flight terminated: Battery depleted after {step+1} steps.")
            break
        
        # Visualization every 10 steps
        if step % 10 == 0:
            print(f"Step {step+1}/{steps}:")
            print(f"  Position: [{current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}]")
            print(f"  Battery: {current_battery:.2f}%")
            print(f"  Speed: {current_speed:.2f} m/s")
            print(f"  Distance to target: {distance_to_target:.2f} units")
            print(f"  Estimated arrival time: {digital_twin.get_estimated_arrival_time():.2f} seconds")
            print(f"  Estimated final battery: {digital_twin.get_battery_estimate():.2f}%")
            
            # Visualize current state
            if step % 20 == 0:
                digital_twin.visualize_state()
    
    # Final visualization
    print("\nFlight completed:")
    print(f"  Final position: [{current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}]")
    print(f"  Final battery: {current_battery:.2f}%")
    print(f"  Total steps: {step+1}")
    print(f"  Final distance to target: {distance_to_target:.2f} units")
    
    digital_twin.visualize_state()
    
    return {
        'success': distance_to_target < 2.0,
        'steps': step+1,
        'final_distance': distance_to_target,
        'final_battery': current_battery,
        'path': digital_twin.env.path_history
    }


def visualize_comparison():
    """Create a comparison visualization between traditional and RL approaches"""
    plt.figure(figsize=(12, 5))
    
    # Traditional approach (left)
    plt.subplot(1, 2, 1)
    plt.title("Traditional Approach")
    
    # Start and end points
    plt.scatter(0, 0, color='green', s=100, marker='o', label='Start')
    plt.scatter(10, 10, color='red', s=100, marker='o', label='End')
    
    # Right-angle path
    plt.plot([0, 0, 10], [0, 10, 10], 'b-', marker='s', markersize=8)
    plt.xlim(-2, 12)
    plt.ylim(-2, 12)
    plt.text(1, 4, "• Multiple discrete commands", fontsize=9)
    plt.text(1, 3, "• Right-angle movements", fontsize=9)
    plt.text(1, 2, "• Manual path planning", fontsize=9)
    
    # RL approach (right)
    plt.subplot(1, 2, 2)
    plt.title("RL Approach (Theoretical)")
    
    # Start and end points
    plt.scatter(0, 0, color='green', s=100, marker='o', label='Start')
    plt.scatter(10, 10, color='red', s=100, marker='o', label='End')
    
    # Smooth path
    x = np.linspace(0, 10, 100)
    y = 10 * (1 - np.cos(x * np.pi / 20))
    plt.plot(x, y, 'c-', marker='s', markersize=8, markevery=25)
    plt.xlim(-2, 12)
    plt.ylim(-2, 12)
    plt.text(1, 4, "• Single destination command", fontsize=9)
    plt.text(1, 3, "• Smooth trajectory", fontsize=9)
    plt.text(1, 2, "• Automatic path planning", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("rl_vs_traditional.png")
    plt.show()


def main():
    """Main function to run the drone reinforcement learning system"""
    print("Drone Reinforcement Learning Path Planning System")
    print("=" * 50)
    
    # Check if CSV exists, if not generate it
    csv_file = "drone_training_data.csv"
    if not os.path.exists(csv_file):
        print("\nGenerating training data...")
        generate_drone_training_data(num_samples=500, filename=csv_file)
    
    # Check if trained model exists
    model_path = "models/drone_ddpg"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_exists = os.path.exists(f"{model_path}_actor.weights.h5") and os.path.exists(f"{model_path}_critic.weights.h5")
    
    if model_exists:
        print("\nLoading pre-trained model...")
        
        # Create a temporary environment to initialize agent
        temp_env = DroneEnvironment(
            start_position=[0, 0, 0],
            target_position=[50, 50, 50]
        )
        
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]
        action_high = temp_env.action_space.high
        
        agent = DDPGAgent(state_dim, action_dim, action_high)
        load_success = agent.load(model_path)
        
        if not load_success:
            print("Failed to load model. Training a new one...")
            agent = train_drone_agent(csv_file, episodes=500, save_path=model_path)
    else:
        print("\nNo pre-trained model found. Training a new one...")
        agent = train_drone_agent(csv_file, episodes=500, save_path=model_path)
    
    # Create test environment for evaluation
    test_env = DroneEnvironment(
        start_position=[10, 10, 10],
        target_position=[80, 80, 80],
        initial_battery=90,
        max_speed=5.0
    )
    
    # Add some obstacles
    test_env.obstacles = [
        ([45, 45, 45], 10),  # A large obstacle in the middle
        ([30, 60, 40], 5)    # Another obstacle
    ]
    
    print("\nEvaluating agent on test environment...")
    evaluation_results = evaluate_agent(agent, test_env, num_episodes=3)
    
    # Create digital twin
    print("\nInitializing Digital Twin for drone...")
    digital_twin = DigitalTwin(agent, test_env)
    
    # Simulate flight
    print("\nSimulating drone flight with digital twin guidance...")
    simulation_results = simulate_drone_flight(digital_twin, steps=200, noise_level=0.05)
    
    # Visualize comparison between traditional and RL approaches
    print("\nGenerating comparison visualization...")
    visualize_comparison()
    
    print("\nDrone flight demonstration completed!")


if __name__ == "__main__":
    main()
