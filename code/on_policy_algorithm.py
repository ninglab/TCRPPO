import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import sys
import gym
import numpy as np
import torch as th
from config import device
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv
from data_utils import num2seq, edit_sequence
from good_buffer import GoodBuffer
from collections import deque
import random

class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        reward_model: object = None,
        good_coef: float = 0.01,
        buffer_config: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.good_coef = good_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.reward_model = reward_model
        if _init_setup_model:
            self._setup_model()
        
        self.bad_init_states = []
        self.bad_rewards = []
        if buffer_config is not None:
            self.init_buffer_size = buffer_config['init_size']
            self.bad_example_rate = buffer_config['bad_example_rate']
            self.bad_ratio = buffer_config['init_bad_ratio']
            self.bad_ratio_rate = buffer_config['bad_ratio_rate']
            self.bad_ratio_step = buffer_config['bad_ratio_step']
            self.use_tcr = buffer_config['use_tcr']
            self.max_bad_ratio = buffer_config['max_bad_ratio']
            
        if buffer_config is not None: self.good_buffer = GoodBuffer(buffer_config)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        time1 = time.time()

        good_actions = []
        good_states = []
        
        resets = [{} for _ in range(env.num_envs)]
        all_obs = [[] for _ in range(env.num_envs)]
        all_actions = [[] for _ in range(env.num_envs)]
        
        gmm_stop_criteria = env.get_attr("gmm_stop_criteria")[0]
        score_stop_criteria = env.get_attr("score_stop_criteria")[0]
        max_tcr_len = env.get_attr("max_tcr_len")[0]
         
        collect_bad_samples = []
        collect_bad_rewards = []
        remove_idxs = []
        random_integers = [i for i in range(len(self.bad_init_states))]
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            
            for i in range(env.num_envs):
                all_actions[i].append(actions[i,:])
                all_obs[i].append(self._last_obs[i,:])
            
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
            tcrs = num2seq(self._last_obs[:, :max_tcr_len])
            peptides = num2seq(self._last_obs[:, max_tcr_len:])
            new_tcrs = edit_sequence(tcrs, clipped_actions)
            try: 
                rewards = self.reward_model.reward(new_tcrs, peptides)
            except:
                pdb.set_trace()
            new_obs, rewards, dones, infos = env.step((clipped_actions, rewards, resets))
            
            
            for i, info in enumerate(infos):
                if dones[i]:
                    if info['score'] >= score_stop_criteria and info['score1'] + info['score2'] >= gmm_stop_criteria and self.good_coef != 0:
                        good_actions.extend(all_actions[i])
                        good_states.extend(all_obs[i])
                    
                    if info['score'] < 0.9 or info['score1'] + info['score2'] < gmm_stop_criteria:
                        if len(resets[i]) == 0 or random.random() < 0.5:
                            collect_bad_samples.append( (info['peptide'], info['init_tcr']) )
                            collect_bad_rewards.append( self.bad_example_rate ** (1.0 - info['rewards']) )
                        
                    if random.random() < self.bad_ratio and len(self.bad_init_states) > 1000:
                        idx = random.choices( random_integers, weights=self.bad_rewards )[0]
                        remove_idxs.append(idx)
                        if self.use_tcr: resets[i] = {'peptide': self.bad_init_states[idx][0], 'init_tcr': self.bad_init_states[idx][1]}
                        else: resets[i] = {'peptide': self.bad_init_states[idx][0], 'init_tcr': None}
                    else:
                        resets[i] = {}
                        
                all_actions[i] = []
                all_obs[i] = []
                
                for key, value in info.items():
                    if key == "episode" or key == "terminal_observation": continue 
                    #pdb.set_trace()
                    if type(value) == float:
                        print("%s: %.4f; " % (key, value), end='') 
                    else:
                        print("%s: %s; " % (key, value), end='')
                print("\n") 
            
            #if any(dones): pdb.set_trace()
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
             
            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            
            
            if len(rewards.shape) > 1:
                rewards = rewards.squeeze(1)
                values = values.squeeze(1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            
            self._last_obs = new_obs
            self._last_dones = dones

        time2 = time.time()
        print("time cost for %d envs: %s" % (env.num_envs, time2 - time1))
        if len(remove_idxs) > 0:
            self.bad_init_states = [self.bad_init_states[idx] for idx in range(len(self.bad_init_states)) if idx not in remove_idxs]
            self.bad_rewards = [self.bad_rewards[idx] for idx in range(len(self.bad_rewards)) if idx not in remove_idxs]
        
        self.bad_init_states.extend(collect_bad_samples)
        self.bad_rewards.extend(collect_bad_rewards)
        if len(self.bad_init_states) > self.init_buffer_size:
            remove_idxs = len(self.bad_init_states) - self.init_buffer_size
            self.bad_init_states = self.bad_init_states[remove_idxs:]
            self.bad_rewards = self.bad_rewards[remove_idxs:]
        
        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        if len(good_states) > 0:
            print("add %d good actions to buffer" % (len(good_states)))
            self.good_buffer.store(good_states, good_actions)
            
        callback.on_rollout_end()
        sys.stdout.flush()
        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        
        #max_tcr_len = self.env.get_attr("max_tcr_len")[0]
        #tcrs = num2seq(self._last_obs[:, :max_tcr_len])
        #peptides = num2seq(self._last_obs[:, max_tcr_len:])
        #rewards = self.reward_model.reward(tcrs, peptides)
        
        callback.on_training_start(locals(), globals())
        k = 1
        while self.num_timesteps < total_timesteps:
            
            if self.bad_ratio < self.max_bad_ratio and self.num_timesteps > self.bad_ratio_step * k:
                k += 1
                self.bad_ratio += self.bad_ratio_rate
            
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            
            #if iteration % 10 == 0 and self.good_coef != 0 and len(self.good_buffer._states) > self.batch_size: self.good_buffer.update_priority()
            
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
