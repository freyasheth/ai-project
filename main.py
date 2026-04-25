import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
N = 10
START = (0, 0)
GOAL = (N - 1, N - 1)
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_DICT = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
REWARD_GOAL = 100
REWARD_INVALID = -100
REWARD_STEP = -1
ALPHA = 0.15
GAMMA = 0.95
EPSILON_INIT = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
EPISODES = 3000
MAX_STEPS = 200
EVAL_EPISODES = 50
ADAPT_EPISODES = 300
ADAPT_EVAL_INTERVAL = 25
CHECK_INTERVAL = 100
RECENT_PATHS_WINDOW = 5
BOREDOM_THRESHOLD = 0.65
MAX_FLOW_OBSTACLES = 3
SEEDS = list(range(20))
MAP_IDS = ['simple', 'medium', 'hard']
HAZARD_TYPES = ['middle_one', 'middle_two', 'turn_one', 'corridor_one', 'random_one']

# Three 10x10 evacuation maps with START=(0,0), GOAL=(9,9)
MAP_OBSTACLES = {
    'simple': {
        # A simple building with one long corridor and occasional side rooms
        (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
        (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4),
        (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6),
        (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8),
    },
    'medium': {
        # Two routes around a central barrier
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
        (6, 2), (6, 3), (6, 4),
        (8, 3), (8, 4), (8, 5), (8, 6),
    },
    'hard': {
        # A harder building layout with multiple blocked corridors and at least three routes
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (3, 1), (4, 1), (5, 1),
        (2, 3), (3, 3), (4, 3),
        (5, 3), (6, 3),
        (3, 6), (4, 6), (5, 6), (6, 6),
        (7, 2), (7, 3), (7, 4), (7, 5),
        (8, 1), (8, 2), (8, 3), (8, 5), (8, 6),
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def state_to_idx(state):
    return state[0] * N + state[1]


def is_valid(pos, obstacles):
    x, y = pos
    return 0 <= x < N and 0 <= y < N and pos not in obstacles


def step(state, action, obstacles):
    dx, dy = ACTION_DICT[action]
    new_state = (state[0] + dx, state[1] + dy)
    if not is_valid(new_state, obstacles):
        return state, REWARD_INVALID, False
    if new_state == GOAL:
        return new_state, REWARD_GOAL, True
    return new_state, REWARD_STEP, False


def bfs_path_exists(obstacles):
    if START in obstacles or GOAL in obstacles:
        return False, []
    visited = {START}
    queue = deque([(START, [START])])
    while queue:
        state, path = queue.popleft()
        if state == GOAL:
            return True, path
        for action in ACTIONS:
            dx, dy = ACTION_DICT[action]
            neighbor = (state[0] + dx, state[1] + dy)
            if neighbor not in visited and is_valid(neighbor, obstacles):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return False, []


def choose_action(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    state_idx = state_to_idx(state)
    values = Q[state_idx]
    max_value = np.max(values)
    best_actions = [i for i, v in enumerate(values) if v == max_value]
    return ACTIONS[random.choice(best_actions)]


def q_update(Q, state, action, reward, next_state):
    s_idx = state_to_idx(state)
    ns_idx = state_to_idx(next_state)
    a_idx = ACTIONS.index(action)
    Q[s_idx, a_idx] += ALPHA * (reward + GAMMA * np.max(Q[ns_idx]) - Q[s_idx, a_idx])


def greedy_path(Q, obstacles):
    state = START
    path = [state]
    visited = set()
    steps = 0
    while state != GOAL and steps < MAX_STEPS:
        visited.add(state)
        state_idx = state_to_idx(state)
        values = Q[state_idx]
        max_value = np.max(values)
        best_actions = [i for i, v in enumerate(values) if v == max_value]
        action = ACTIONS[random.choice(best_actions)]
        next_state, _, done = step(state, action, obstacles)
        if next_state == state:
            return False, path, 'invalid_move'
        path.append(next_state)
        state = next_state
        steps += 1
        if state in visited:
            return False, path, 'loop'
        if done:
            return True, path, 'success'
    if state != GOAL:
        return False, path, 'max_steps'
    return True, path, 'success'


def evaluate_agent(Q, obstacles, episodes=EVAL_EPISODES):
    successes = 0
    total_steps = 0
    total_reward = 0
    failure_counts = {'invalid_move': 0, 'loop': 0, 'max_steps': 0}
    representative_path = []
    representative_reason = 'success'
    for i in range(episodes):
        success, path, reason = greedy_path(Q, obstacles)
        if i == 0:
            representative_path = path
            representative_reason = reason
        if success:
            successes += 1
            total_steps += len(path) - 1
            total_reward += REWARD_GOAL + (len(path) - 2) * REWARD_STEP
        else:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
            if reason == 'invalid_move':
                total_reward += REWARD_INVALID
            else:
                total_reward += REWARD_STEP * len(path)
    success_rate = successes / episodes * 100
    avg_steps = total_steps / successes if successes > 0 else float('nan')
    avg_reward = total_reward / episodes
    if successes == 0:
        representative_reason = max(failure_counts, key=failure_counts.get)
    return success_rate, avg_steps, avg_reward, representative_path, success_rate > 0, representative_reason


def path_to_state_action_pairs(path):
    pairs = []
    for i in range(len(path) - 1):
        state = path[i]
        next_state = path[i + 1]
        for action, delta in ACTION_DICT.items():
            if (state[0] + delta[0], state[1] + delta[1]) == next_state:
                pairs.append((state, action))
                break
    return pairs


def boredom_score(recent_paths):
    if not recent_paths:
        return 0.0
    all_pairs = []
    for path in recent_paths:
        all_pairs.extend(path_to_state_action_pairs(path))
    unique_pairs = set(all_pairs)
    return 1.0 - (len(unique_pairs) / len(all_pairs)) if all_pairs else 0.0


def train_q_learning(obstacles, seed, episodes=EPISODES, Q_init=None, epsilon_start=EPSILON_INIT):
    set_seed(seed)
    Q = Q_init.copy() if Q_init is not None else np.zeros((N * N, len(ACTIONS)))
    epsilon = epsilon_start
    results = []
    unique_paths = set()
    for episode in range(episodes):
        state = START
        total_reward = 0
        steps = 0
        success = False
        while steps < MAX_STEPS:
            action = choose_action(Q, state, epsilon)
            next_state, reward, done = step(state, action, obstacles)
            q_update(Q, state, action, reward, next_state)
            total_reward += reward
            steps += 1
            if done:
                success = True
                break
            state = next_state
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        results.append({'episode': episode, 'success': success, 'steps': steps, 'reward': total_reward})
        if episode % CHECK_INTERVAL == 0:
            path_success, greedy_p, _ = greedy_path(Q, obstacles)
            if path_success:
                unique_paths.add(tuple(greedy_p))
    path_diversity = len(unique_paths)
    return Q, results, path_diversity


def select_flow_obstacle(obstacles, path):
    if len(path) < 6:
        return None
    candidate_indices = range(len(path) // 4, len(path) * 3 // 4)
    for idx in candidate_indices:
        pos = path[idx]
        if pos not in obstacles and pos != START and pos != GOAL:
            new_obstacles = obstacles | {pos}
            if bfs_path_exists(new_obstacles)[0]:
                return pos
    return None


def train_flow_guided_q_learning(obstacles, seed):
    set_seed(seed)
    Q = np.zeros((N * N, len(ACTIONS)))
    epsilon = EPSILON_INIT
    current_obstacles = set(obstacles)
    results = []
    recent_paths = []
    flow_obstacles = []
    challenge_log = []
    unique_paths = set()
    for episode in range(EPISODES):
        state = START
        total_reward = 0
        steps = 0
        success = False
        while steps < MAX_STEPS:
            action = choose_action(Q, state, epsilon)
            next_state, reward, done = step(state, action, current_obstacles)
            q_update(Q, state, action, reward, next_state)
            total_reward += reward
            steps += 1
            if done:
                success = True
                break
            state = next_state
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        results.append({'episode': episode, 'success': success, 'steps': steps, 'reward': total_reward})
        if episode % CHECK_INTERVAL == 0:
            path_success, greedy_p, _ = greedy_path(Q, current_obstacles)
            if path_success:
                unique_paths.add(tuple(greedy_p))
                recent_paths.append(greedy_p)
                if len(recent_paths) > RECENT_PATHS_WINDOW:
                    recent_paths.pop(0)
                boredom = boredom_score(recent_paths)
                if boredom > BOREDOM_THRESHOLD and len(flow_obstacles) < MAX_FLOW_OBSTACLES:
                    new_obstacle = select_flow_obstacle(current_obstacles, greedy_p)
                    if new_obstacle:
                        current_obstacles.add(new_obstacle)
                        flow_obstacles.append(new_obstacle)
                        challenge_log.append({'episode': episode, 'boredom': boredom, 'obstacle': new_obstacle})
                        epsilon = max(epsilon, 0.35)
    path_diversity = len(unique_paths)
    return Q, results, current_obstacles, flow_obstacles, challenge_log, path_diversity


def adapt_agent(Q_init, obstacles, seed, episodes=ADAPT_EPISODES, interval=ADAPT_EVAL_INTERVAL):
    Q = Q_init.copy()
    set_seed(seed)
    epsilon = 0.35
    episodes_to_recover = episodes
    recovery_found = False
    recovery_curve = []
    batch = 0
    for block in range(interval, episodes + 1, interval):
        for _ in range(interval):
            state = START
            steps = 0
            while steps < MAX_STEPS:
                action = choose_action(Q, state, epsilon)
                next_state, reward, done = step(state, action, obstacles)
                q_update(Q, state, action, reward, next_state)
                steps += 1
                if done:
                    break
                state = next_state
        batch += interval
        success_rate, _, _, _, _, _ = evaluate_agent(Q, obstacles, episodes=EVAL_EPISODES)
        recovery_curve.append({'episode': batch, 'success': success_rate})
        if not recovery_found and success_rate >= 90.0:
            episodes_to_recover = batch
            recovery_found = True
    return Q, episodes_to_recover, recovery_curve


def generate_hazard_cases(obstacles, baseline_path):
    hazards = {}
    available_cells = [pos for pos in baseline_path[1:-1] if pos != START and pos != GOAL and pos not in obstacles]
    if not available_cells:
        return hazards
    # middle_one
    mid_idx = len(available_cells) // 2
    hazards['middle_one'] = available_cells[mid_idx]
    # middle_two
    if len(available_cells) > 3:
        first = available_cells[max(0, len(available_cells) // 3)]
        second = available_cells[min(len(available_cells) - 1, 2 * len(available_cells) // 3)]
        hazards['middle_two'] = [first, second] if first != second else [first]
    else:
        hazards['middle_two'] = [available_cells[0], available_cells[-1]] if available_cells[0] != available_cells[-1] else [available_cells[0]]
    # turn_one
    for idx in range(1, len(baseline_path) - 1):
        prev_dir = (baseline_path[idx][0] - baseline_path[idx - 1][0], baseline_path[idx][1] - baseline_path[idx - 1][1])
        next_dir = (baseline_path[idx + 1][0] - baseline_path[idx][0], baseline_path[idx + 1][1] - baseline_path[idx][1])
        if prev_dir != next_dir and baseline_path[idx] not in obstacles and baseline_path[idx] != START and baseline_path[idx] != GOAL:
            hazards['turn_one'] = baseline_path[idx]
            break
    # corridor_one
    mid_line = N // 2
    for pos in baseline_path[1:-1]:
        if abs(pos[1] - mid_line) <= 1 and pos not in obstacles:
            hazards['corridor_one'] = pos
            break
    # random_one
    all_valid = [(x, y) for x in range(N) for y in range(N) if (x, y) not in obstacles and (x, y) != START and (x, y) != GOAL]
    random.shuffle(all_valid)
    for pos in all_valid:
        if pos not in baseline_path[1:-1]:
            hazards['random_one'] = pos
            break
    if 'random_one' not in hazards and available_cells:
        hazards['random_one'] = random.choice(available_cells)
    # Validate hazard cases with BFS
    validated = {}
    for hazard_type, hazard in hazards.items():
        if isinstance(hazard, list):
            candidate_obstacles = obstacles.union(hazard)
        else:
            candidate_obstacles = obstacles | {hazard}
        valid, _ = bfs_path_exists(candidate_obstacles)
        if valid:
            validated[hazard_type] = hazard
    return validated


def plot_grid(obstacles, path=None, obstacle=None, extra_obstacles=None, title='', filename=''):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.grid(True)
    for obs in obstacles:
        ax.add_patch(plt.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1, color='black'))
    if extra_obstacles:
        for obs in extra_obstacles:
            ax.add_patch(plt.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1, color='orange', alpha=0.7))
    if obstacle:
        if isinstance(obstacle, list):
            for obs in obstacle:
                ax.add_patch(plt.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1, color='red', alpha=0.6))
        else:
            ax.add_patch(plt.Rectangle((obstacle[1] - 0.5, obstacle[0] - 0.5), 1, 1, color='red', alpha=0.6))
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, '-o', color='dodgerblue', linewidth=2, markersize=6)
    ax.plot(START[1], START[0], 'gs', markersize=12, label='Start')
    ax.plot(GOAL[1], GOAL[0], 'rs', markersize=12, label='Goal')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize='small')
    if filename:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def moving_average(values, window=50):
    return pd.Series(values).rolling(window=window, min_periods=1).mean().tolist()


def run_single_map_seed(map_id, seed):
    print(f'Running map={map_id} seed={seed}')
    obstacles = MAP_OBSTACLES[map_id]
    assert START not in obstacles and GOAL not in obstacles
    path_exists, reference_path = bfs_path_exists(obstacles)
    assert path_exists, f'No valid path for {map_id}'
    Q_base, train_base, path_div_base = train_q_learning(obstacles, seed)
    success_before, avg_steps_before, avg_reward_before, _, _, _ = evaluate_agent(Q_base, obstacles)
    Q_flow, train_flow, flow_obstacles, flow_added, challenge_log, path_div_flow = train_flow_guided_q_learning(obstacles, seed)
    fg_success_before, fg_avg_steps_before, fg_avg_reward_before, _, _, _ = evaluate_agent(Q_flow, obstacles)
    baseline_greedy_success, baseline_greedy_path, _ = greedy_path(Q_base, obstacles)
    if not baseline_greedy_success:
        baseline_greedy_path = reference_path
    hazards = generate_hazard_cases(obstacles, baseline_greedy_path)
    results = []
    representative_curves = None
    for hazard_type in HAZARD_TYPES:
        if hazard_type not in hazards:
            continue
        hazard = hazards[hazard_type]
        changed_obstacles = obstacles.union(hazard if isinstance(hazard, list) else {hazard})
        base_zero_rate, base_zero_steps, base_zero_reward, _, _, base_zero_reason = evaluate_agent(Q_base, changed_obstacles)
        flow_zero_rate, flow_zero_steps, flow_zero_reward, _, _, flow_zero_reason = evaluate_agent(Q_flow, changed_obstacles)
        Q_base_adapted, base_recover_episodes, base_recovery_curve = adapt_agent(Q_base, changed_obstacles, seed)
        Q_flow_adapted, flow_recover_episodes, flow_recovery_curve = adapt_agent(Q_flow, changed_obstacles, seed)
        base_adapt_rate, base_adapt_steps, base_adapt_reward, _, _, _ = evaluate_agent(Q_base_adapted, changed_obstacles)
        flow_adapt_rate, flow_adapt_steps, flow_adapt_reward, _, _, _ = evaluate_agent(Q_flow_adapted, changed_obstacles)
        base_zero_adapt = base_zero_rate / success_before if success_before > 0 else 0.0
        base_adapt_adapt = base_adapt_rate / success_before if success_before > 0 else 0.0
        flow_zero_adapt = flow_zero_rate / fg_success_before if fg_success_before > 0 else 0.0
        flow_adapt_adapt = flow_adapt_rate / fg_success_before if fg_success_before > 0 else 0.0
        results.append({
            'map_id': map_id,
            'seed': seed,
            'method': 'Baseline Q-learning',
            'hazard_type': hazard_type,
            'success_before': success_before,
            'zero_shot_success_after': base_zero_rate,
            'adapted_success_after': base_adapt_rate,
            'zero_shot_adaptability': base_zero_adapt,
            'adapted_adaptability': base_adapt_adapt,
            'avg_steps_before': avg_steps_before,
            'zero_shot_avg_steps_after': base_zero_steps,
            'adapted_avg_steps_after': base_adapt_steps,
            'avg_reward_before': avg_reward_before,
            'zero_shot_avg_reward_after': base_zero_reward,
            'adapted_avg_reward_after': base_adapt_reward,
            'episodes_to_recover': base_recover_episodes,
            'path_diversity': path_div_base,
            'number_of_flow_obstacles_added': 0,
            'zero_shot_failure_reason': base_zero_reason,
        })
        results.append({
            'map_id': map_id,
            'seed': seed,
            'method': 'Flow-guided Q-learning',
            'hazard_type': hazard_type,
            'success_before': fg_success_before,
            'zero_shot_success_after': flow_zero_rate,
            'adapted_success_after': flow_adapt_rate,
            'zero_shot_adaptability': flow_zero_adapt,
            'adapted_adaptability': flow_adapt_adapt,
            'avg_steps_before': fg_avg_steps_before,
            'zero_shot_avg_steps_after': flow_zero_steps,
            'adapted_avg_steps_after': flow_adapt_steps,
            'avg_reward_before': fg_avg_reward_before,
            'zero_shot_avg_reward_after': flow_zero_reward,
            'adapted_avg_reward_after': flow_adapt_reward,
            'episodes_to_recover': flow_recover_episodes,
            'path_diversity': path_div_flow,
            'number_of_flow_obstacles_added': len(flow_added),
            'zero_shot_failure_reason': flow_zero_reason,
        })
        if representative_curves is None:
            representative_curves = {
                'map_id': map_id,
                'seed': seed,
                'hazard_type': hazard_type,
                'baseline_curve': base_recovery_curve,
                'flow_curve': flow_recovery_curve,
                'changed_obstacles': changed_obstacles,
                'baseline_path': baseline_greedy_path,
                'flow_path': None,
                'hazard': hazard,
                'flow_obstacles': flow_added,
            }
    return results, challenge_log, representative_curves


def run_multiple_maps_seeds():
    all_results = []
    all_challenge_logs = []
    representative_case = None
    for map_id in MAP_IDS:
        for seed in SEEDS:
            results, challenge_log, rep = run_single_map_seed(map_id, seed)
            all_results.extend(results)
            all_challenge_logs.extend([{'map_id': map_id, 'seed': seed, **entry} for entry in challenge_log])
            if representative_case is None and rep is not None:
                representative_case = rep
    return all_results, all_challenge_logs, representative_case


def save_outputs(all_results, all_challenge_logs):
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(all_results)
    challenge_df = pd.DataFrame(all_challenge_logs)
    results_df.to_csv('all_results.csv', index=False)
    challenge_df.to_csv('challenge_log.csv', index=False)
    numeric_cols = [
        'zero_shot_success_after',
        'adapted_success_after',
        'zero_shot_adaptability',
        'adapted_adaptability',
        'episodes_to_recover',
        'path_diversity',
    ]
    summary_by_method = results_df.groupby('method')[numeric_cols].agg(['mean', 'std'])
    summary_by_method.to_csv('summary_by_method.csv')
    summary_by_map = results_df.groupby('map_id')[numeric_cols].agg(['mean', 'std'])
    summary_by_map.to_csv('summary_by_map.csv')
    summary_by_hazard = results_df.groupby('hazard_type')[numeric_cols].agg(['mean', 'std'])
    summary_by_hazard.to_csv('summary_by_hazard.csv')


def plot_comparison(results_df, curves_df, challenge_df, representative_case):
    # Map layouts
    for map_id, obstacles in MAP_OBSTACLES.items():
        plot_grid(obstacles, title=f'{map_id.capitalize()} map layout', filename=f'{map_id}_map.png')
    # Representative baseline path before obstacle
    rep_map = representative_case['map_id']
    rep_seed = representative_case['seed']
    rep_hazard = representative_case['hazard']
    rep_obstacles = MAP_OBSTACLES[rep_map]
    Q_base, _, _ = train_q_learning(rep_obstacles, rep_seed)
    _, base_path, _ = greedy_path(Q_base, rep_obstacles)
    plot_grid(rep_obstacles, path=base_path, title='Representative Baseline Path Before Obstacle', filename='baseline_path_before.png')
    # Flow-guided path after obstacle
    changed_obstacles = representative_case['changed_obstacles']
    Q_flow, _, _, _, _, _ = train_flow_guided_q_learning(rep_obstacles, rep_seed)
    _, flow_path, _ = greedy_path(Q_flow, changed_obstacles)
    plot_grid(changed_obstacles, path=flow_path, obstacle=rep_hazard, extra_obstacles=representative_case['flow_obstacles'], title='Representative Flow-guided Path After Obstacle', filename='flow_path_after.png')
    # Bar plots by method
    mean_by_method = results_df.groupby('method')[['zero_shot_success_after', 'episodes_to_recover', 'path_diversity']].mean()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(mean_by_method.index, mean_by_method['zero_shot_success_after'], color=['tab:blue', 'tab:orange'])
    axes[0].set_title('Zero-shot success by method')
    axes[0].set_ylabel('Success rate (%)')
    axes[1].bar(mean_by_method.index, mean_by_method['episodes_to_recover'], color=['tab:blue', 'tab:orange'])
    axes[1].set_title('Episodes to recover by method')
    axes[1].set_ylabel('Episodes')
    axes[2].bar(mean_by_method.index, mean_by_method['path_diversity'], color=['tab:blue', 'tab:orange'])
    axes[2].set_title('Path diversity by method')
    axes[2].set_ylabel('Unique greedy paths')
    plt.tight_layout()
    plt.savefig('comparison_bars.png')
    plt.close()
    # Recovery curve during adaptation
    if representative_case is not None:
        plt.figure(figsize=(10, 6))
        base_curve = representative_case['baseline_curve']
        flow_curve = representative_case['flow_curve']
        plt.plot([entry['episode'] for entry in base_curve], [entry['success'] for entry in base_curve], label='Baseline adaptation')
        plt.plot([entry['episode'] for entry in flow_curve], [entry['success'] for entry in flow_curve], label='Flow-guided adaptation')
        plt.xlabel('Adaptation episodes')
        plt.ylabel('Evaluation success rate (%)')
        plt.title('Recovery curve during adaptation')
        plt.legend()
        plt.savefig('adaptation_recovery_curve.png')
        plt.close()
    # Training success curve for first map/seed
    first_map = MAP_IDS[0]
    Q_base, train_base, _ = train_q_learning(MAP_OBSTACLES[first_map], SEEDS[0])
    Q_flow, train_flow, _, _, _, _ = train_flow_guided_q_learning(MAP_OBSTACLES[first_map], SEEDS[0])
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average([int(r['success']) for r in train_base]), label='Baseline')
    plt.plot(moving_average([int(r['success']) for r in train_flow]), label='Flow-guided')
    plt.xlabel('Episode')
    plt.ylabel('Training success (moving average)')
    plt.title('Training success curve')
    plt.legend()
    plt.savefig('training_success_curve.png')
    plt.close()
    # Challenge timeline
    if not challenge_df.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(challenge_df['episode'], challenge_df['boredom'], alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Boredom score')
        plt.title('Flow-guided challenge update timeline')
        plt.savefig('challenge_timeline.png')
        plt.close()


def print_summary(results_df):
    mean_results = results_df.groupby('method')[['zero_shot_success_after', 'adapted_success_after', 'zero_shot_adaptability', 'adapted_adaptability', 'episodes_to_recover', 'path_diversity']].mean()
    print('Overall summary table:')
    print(mean_results.round(2))
    best_idx = results_df['zero_shot_adaptability'].idxmax()
    worst_idx = results_df['zero_shot_adaptability'].idxmin()
    best_case = results_df.loc[best_idx]
    worst_case = results_df.loc[worst_idx]
    print('\nBest representative case:')
    print(f"{best_case['method']} on {best_case['map_id']} with {best_case['hazard_type']} (zero-shot adaptability={best_case['zero_shot_adaptability']:.2f})")
    print('\nWorst representative case:')
    print(f"{worst_case['method']} on {worst_case['map_id']} with {worst_case['hazard_type']} (zero-shot adaptability={worst_case['zero_shot_adaptability']:.2f})")
    base_zero = mean_results.loc['Baseline Q-learning', 'zero_shot_adaptability']
    flow_zero = mean_results.loc['Flow-guided Q-learning', 'zero_shot_adaptability']
    base_recover = mean_results.loc['Baseline Q-learning', 'episodes_to_recover']
    flow_recover = mean_results.loc['Flow-guided Q-learning', 'episodes_to_recover']
    base_div = mean_results.loc['Baseline Q-learning', 'path_diversity']
    flow_div = mean_results.loc['Flow-guided Q-learning', 'path_diversity']
    if flow_zero > base_zero:
        interpretation = 'Flow-guided improved zero-shot adaptability.'
    elif flow_recover < base_recover:
        interpretation = 'Flow-guided improved recovery speed.'
    elif flow_div > base_div:
        interpretation = 'Flow-guided improved path diversity.'
    else:
        interpretation = 'Results were mixed, but dynamic testing exposed the adaptability limitation of fixed-map Q-learning.'
    print('\nInterpretation:')
    print(interpretation)
    return interpretation


def main():
    all_results, challenge_logs, representative_case = run_multiple_maps_seeds()
    save_outputs(all_results, challenge_logs)
    results_df = pd.DataFrame(all_results)
    challenge_df = pd.DataFrame(challenge_logs)
    plot_comparison(results_df, None, challenge_df, representative_case)
    print_summary(results_df)


if __name__ == '__main__':
    main()
