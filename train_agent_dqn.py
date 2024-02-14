import gym
import csv
import numpy as np
from agent_dqn import AgentDQN

# ==================================================
# =================== FUNCTIONS ====================
# ==================================================

def run_agent(env, agent, n_episodes):
    """
    Run an agent on an enviroment.

    Parameters
    --------------------
    env: Env
        an enviroment where agent is run

    agent: AgentDQN
        an agent

    n_episodes: int
        number of episodes to run
    """

    scores = []

    for episode in range(1, n_episodes+1):
        score = 0
        done = False
        observation, _ = env.reset()

        #Agent is trained on current episode.
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            observation = next_observation
            score += reward

        #Update stats.
        scores.append(score)

        #Print current stats.
        print('- episode: {} ; score: {:.2f} ; avg score: {:.2f}'.format(episode, score, np.mean(scores[-100:])))



def train_agent(env, agent, n_episodes, path_model, path_hyperparameters, path_stats, path_model_structure):
    """
    Train an agent on an enviroment.

    Parameters
    --------------------
    env: Env
        an enviroment where agent is trained

    agent: AgentDQN
        an agent to train

    n_episodes: int
        number of episodes to train agent

    path_model: string
        path to save agent that is being trained

    path_hyperparameters: string
        path to save hyperparameters used

    path_model_structure: string
        path to save model structure
    """

    if path_model.find('.pth')==-1:
        path_model += '.pth'
    scores = []
    avg_scores = []
    total_obs_done = 0

    for episode in range(1, n_episodes+1):
        obs_count = 0
        score = 0
        done = False
        observation, _ = env.reset()

        #Agent is trained on current episode.
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(observation, action, reward, next_observation, done)
            agent.train()

            observation = next_observation
            score += reward
            obs_count += 1

        #Update stats.
        scores.append(score)
        avg_scores.append(np.mean(scores[-100:]))
        total_obs_done += obs_count

        #Save model
        agent.save_model(path_model)

        #Print current stats.
        print('- episode: {} ; score: {:.2f} ; avg score: {:.2f} ; epsilon: {:.2f} ; n° obs done: {} ; total obs: {}'.format(episode, score, avg_scores[-1], agent.epsilon, obs_count, total_obs_done))

    #Save stats of training into a csv file.
    if path_stats.find('.csv')==-1:
        path_stats = path_stats + '.csv'

    csv_stats_file = open(path_stats, 'w')
    csv_stats_writer = csv.writer(csv_stats_file, delimiter=';')

    csv_stats_writer.writerow(['episode', 'score', 'average score'])
    for i in range(n_episodes):
        csv_stats_writer.writerow([i+1, scores[i], avg_scores[i]])

    csv_stats_file.close()

    #Save hyperparameters into a csv file.
    if path_hyperparameters.find('.csv')==-1:
        path_hyperparameters = path_hyperparameters + '.csv'

    agent.save_report_hyperparameters(path_hyperparameters)

    #Save into a document how current model is built.
    if path_model_structure.find('.txt')==-1:
        path_model_structure += ".txt"

    model_struct_file = open(path_model_structure, 'w')

    model_struct_file.write("Model's structure: \n")
    for param_tensor in agent.model.state_dict():
        model_struct_file.write('{} \t {} \n'.format(param_tensor, agent.model.state_dict()[param_tensor].size()))
    
    model_struct_file.close()



def test_agent(env, agent, n_episodes, path_testing):
    """
    Testing agent and save its scores.
    
    Parameters
    --------------------
    env: Env
        an envirement to test an agent

    agent: AgentDQN
        an agent to test

    n_episodes: int
        number of episodes to test agent

    path_testing: str
        file to save testing
    """

    scores = []
    
    csv_test_file = open(path_testing, 'w')
    csv_test_writer = csv.writer(csv_test_file, delimiter=';')
    csv_test_writer.writerow(['episode', 'score', 'is terminated?'])

    print("-------------------- TESTING PHASE --------------------")

    for episode in range(1, n_episodes+1):
        observation, _ = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            score += reward
            observation = next_observation

        csv_test_writer.writerow([episode, score, terminated])

        #Update stats.
        scores.append(score)

        #Print current stats.
        print('- episode: {} ; score: {:.2f} ; avg score: {:.2f}'.format(episode, score, np.mean(scores)))
    
    csv_test_writer.writerow(['avg score:', np.mean(scores)])
    csv_test_writer.writerow(['std score:', np.std(scores)])
    csv_test_file.close()

        

# ==================================================
# ====================== MAIN ======================
# ==================================================

is_trained = False
is_tested = False
is_lunar_lander_used = True
is_cartpole_used = not is_lunar_lander_used     #Cart Pole enviroment is used to test only if agent works well.

if is_lunar_lander_used:
    path = './models/lunar_lander/'
    env = gym.make('LunarLander-v2', render_mode='human')
    
    if is_trained:
        n_episodes = 600
        agent = AgentDQN(env.observation_space.shape[0], env.action_space.n, 1000000, 64, lr=10**-3, eps_dec=5*10**-5)
        
        train_agent(env, agent, n_episodes, path+'model.pth', path+'hpmr.csv', path+'history_training.csv', path+'structure_model.txt')
    elif is_tested:
        n_episodes = 100
        agent = AgentDQN(env.observation_space.shape[0], env.action_space.n, 1, 1, is_trained=False)
        agent.load_model(path+'model.pth')

        test_agent(env, agent, n_episodes, path+'test.csv')
    else:
        n_episodes = 100
        agent = AgentDQN(env.observation_space.shape[0], env.action_space.n, 1, 1, is_trained=False)
        agent.load_model(path+'model.pth')
    
        run_agent(env, agent, n_episodes)
else:
    path = './models/cartpole/'
    env = gym.make('CartPole-v1', render_mode='human')
    
    if is_trained:
        n_episodes = 500
        agent = AgentDQN(env.observation_space.shape[0], env.action_space.n, 5000, 32, lr=10**-3, eps_dec=10**-4)
        
        train_agent(env, agent, n_episodes, path+'model.pth', path+'hpmr.csv', path+'history_training.csv', path+'structure_model.txt')
    else:
        n_episodes = 100
        agent = AgentDQN(env.observation_space.shape[0], env.action_space.n, 1, 1, is_trained=False)
        agent.load_model(path+'model.pth')
    
        run_agent(env, agent, n_episodes)