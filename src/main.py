import gym
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from collections import deque
from torch.distributions import Normal

from network import Actor, Critic
from replay_buffer import Replay_buffer

def get_action(actor,state, evaluate=False):
	with torch.no_grad():
		mean, std = actor(state)
		dist = Normal(mean, std)
		action = dist.sample()
	log_prob = dist.log_prob(action).sum(dim=-1)
	action = action.clamp(-2.0,2.0)
	
	return action.detach().cpu().numpy(), log_prob.item()


def run(center_actor, center_critic, i, seed, device, env_name, actor_lr, 
										 critic_lr, buffer_size, episodes, 
										 max_step, gamma, log_interval):
	print("Run")
	env = gym.make(env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	action_limit = env.action_space.high[0]  
	
	if device == 'cuda':
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		torch.cuda.manual_seed_all(seed+i)
	else:
		device = torch.device('cpu')   
		
	# Set a random seed
	env.seed(seed+i)
	np.random.seed(seed+i)
	torch.manual_seed(seed+i)
	
	actor = Actor(state_dim, action_dim, action_limit).to(device)
	critic = Critic(state_dim).to(device)

	actor.load_state_dict(center_actor.state_dict())
	critic.load_state_dict(center_critic.state_dict())
	
	actor_optim = optim.Adam(center_actor.parameters(), lr=actor_lr)
	critic_optim = optim.Adam(center_critic.parameters(), lr=critic_lr)
	
	RB = Replay_buffer(buffer_size)
	
	total_return = deque(maxlen=100)
	
	for episode in range(episodes):
		total_reward = 0
		n_step = 0
		
		state = env.reset()
		
		for s in range(max_step):
				
			#Run
			action, log_prob = get_action(actor,torch.Tensor(state).to(device))
							
			next_state, reward, done, _ = env.step(action)
			
			RB.save_sample(state, action, (reward+8.)/8., done, next_state, log_prob)
			
			
			if n_step == buffer_size or done:
				#Train 
				states, actions, rewards, dones, next_states, log_probs = RB.get_sample()

				states = torch.FloatTensor(list(states)).to(device) # shape: [memory size, state_dim]
				actions = torch.FloatTensor(list(actions)).view(-1,action_dim).to(device) # shape: [memory size, action_dim]
				rewards = torch.FloatTensor(list(rewards)).view(-1,1).to(device) # shape: [memory size]
				dones = torch.FloatTensor(list(dones)).view(-1,1).to(device) # shape: [memory size]
				next_states = torch.FloatTensor(list(next_states)).to(device) # shape: [memory size, state_dim]
				log_probs = torch.FloatTensor(list(log_probs)).view(-1,1).to(device) # shape: [memory size]

#				mean, std = actor(states)
#				dist = Normal(mean, std)
#				action = dist.sample()
#				log_probs = dist.log_prob(action).sum(dim=-1)
#				log_probs = log_probs.view(-1,1).to(device)
			   
				V = critic(states)
				Q = rewards + gamma*critic(next_states).detach()
				Advantage = Q - V
				
					   
				#Actor update
				actor_optim.zero_grad()
				actor_loss = 0
				actor_loss = -(Advantage*log_probs).mean()
				actor_loss.backward()
				for center_actor_param, actor_param in zip(center_actor.parameters(), actor.parameters()):
					center_actor_param._grad = actor_param.grad
				actor_optim.step()

				#Critic update
				critic_optim.zero_grad()
				critic_loss = Advantage.pow(2.).mean()
				critic_loss.backward()
				for center_critic_param, critic_param in zip(center_critic.parameters(), critic.parameters()):
					center_critic_param._grad = critic_param.grad
				critic_optim.step()
				
				actor.load_state_dict(center_actor.state_dict())
				critic.load_state_dict(center_critic.state_dict())
				
				RB.clear()
				n_step = 0 
					
			state = next_state
			total_reward += reward
				
			if done:
				break
			
		total_return.append(total_reward)
		avg_return = np.mean(total_return)
		
		if (episode+1)%log_interval ==0:
			print('---------------------------------------')
			print("Process-{}".format(i))
			print('\tEpisodes:', episode + 1)
			print('\tAverageReturn:', round(avg_return, 2))
			print('---------------------------------------')
		
	env.close()
	print("Process {} End".format(i))
		
# def main(args):
	
	
  
		  
if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description='A3C')
	# parser.add_argument('--env', type=str, default='Pendulum-v0', 
	#					  help='pendulum environment')
	# parser.add_argument('--seed', type=int, default=1, 
	#					  help='seed for random number generators')
	# parser.add_argument('--episode', type=int, default=10000, 
	#					  help='Iterations for train')
	# parser.add_argument('--log_interval', type=int, default=100, 
	#					  help='Train log interval while training')
	# parser.add_argument('--num_process', type=int, default=4, 
	#					  help='Number of process')
	# parser.add_argument('--max_step', type=int, default=200,
	#					  help='max episode step')
	# parser.add_argument('--actor_lr', type=float, default=1e-4, 
	#					  help='actor learning rate')
	# parser.add_argument('--gamma', type=float, default=0.99, 
	#					  help='Gamma(discount factor)')
	# parser.add_argument('--critic_lr', type=float, default=1e-3, 
	#					  help='critic learning rate')
	# parser.add_argument('--replay_buffer', type=int, default=5,
	#					  help='replay buffer_size')
	# parser.add_argument('--device', type=str, default='cpu')
	
	# args = parser.parse_args()	
	
	# main(args)
	
	# seed = args.seed
	# device = args.device
	# env_name = args.env	 
	# actor_lr = args.actor_lr
	# critic_lr = args.critic_lr
	# buffer_size= args.replay_buffer
	# episodes = args.episode
	# max_step = args.max_step
	# gamma = args.gamma
	# log_interval = args.log_interval
	# num_process = args.num_process
	
	seed = 0
	device = 'cpu'
	env_name = "Pendulum-v0"
	actor_lr = 1e-4
	critic_lr = 1e-3
	buffer_size= 5
	episodes = 10000
	max_step = 200
	gamma = 0.99
	log_interval = 100
	num_process = 4
	
	if device == 'cuda':
		assert torch.cuda.is_available()
		mp.set_start_method('spawn',True)
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	else:
		device = torch.device('cpu')   
		
	env = gym.make(env_name)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	action_limit = env.action_space.high[0]  
	env.close()
	
	center_actor = Actor(state_dim, action_dim, action_limit).to(device)
	center_critic = Critic(state_dim).to(device)
	
	center_actor.share_memory()
	center_critic.share_memory()
	
	processes = []	  
	
	print("Mp: {}".format(mp.get_start_method()))
	for i in range(num_process):
		print("Process-{} start".format(i))
		p = mp.Process(target=run, args=(center_actor, center_critic, i, 
										 seed, device, env_name, actor_lr, 
										 critic_lr, buffer_size, episodes, 
										 max_step, gamma, log_interval))
		p.start()
		processes.append(p)
		
	for p in processes:
		p.join()
		
