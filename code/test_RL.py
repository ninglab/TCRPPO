import time
import pdb
import torch
import argparse
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common.env_util import make_vec_env

from ppo import PPO
from tcr_env import TCREnv
from data_utils import num2seq, seq2num
import torch

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
import config
from reward import Reward
from data_utils import num2seq, edit_sequence



def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data[0] , data[1])
                
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    # observation = env.reset(allele=data[1])

                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                if data[0]:
                    if type(data[1]) is str:
                        observation = env.reset(peptide=data[1])
                    else:
                        observation = env.reset(peptide=data[1][0], init_tcr=data[1][1])
                else:
                    observation = data[2]
                
                remote.send(observation)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break

class MySubprocVecEnv(VecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        self._last_obs = None
        self.reward_model = None
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_attr", "max_tcr_len"))
        self.max_tcr_len = self.remotes[0].recv()
        
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def set_reward_model(self, reward_model) -> None:
        self.reward_model = reward_model
        
    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action
        :param actions: the action
        :return: observation, reward, done, information
        """
        tcrs = num2seq(self._last_obs[:, :self.max_tcr_len])
        peptides = num2seq(self._last_obs[:, self.max_tcr_len:])
        new_tcrs = edit_sequence(tcrs, actions)
        rewards = self.reward_model.reward(new_tcrs, peptides)
         
        self.step_async((actions, rewards))
        return self.step_wait()       
        

    def step_async(self, data: Tuple) -> None:
        actions, rewards = data
        for remote, action, reward in zip(self.remotes, actions, rewards):
            remote.send(("step", (action, reward)))
        
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rew, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rew), np.stack(dones), infos
    
    
    def reset(self, dones, peptides, obs) -> VecEnvObs:
        self.reset_async(dones, peptides, obs)
        self._last_obs = self.reset_wait()
        return self._last_obs

    def reset_async(self, dones, peptides, obs):
        for remote, done, peptide, ob in zip(self.remotes, dones, peptides, obs):
            remote.send(("reset", (done, peptide, ob)))
        self.waiting = True

    def reset_wait(self):
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return _flatten_obs(obs, self.observation_space)


    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.
        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--peptide_path", type=str)
    parser.add_argument("--rollout", type=int)
    parser.add_argument("--path", type=str)
    parser.add_argument("--out", type=str)

    parser.add_argument('--ergo_model', type=str)
    parser.add_argument('--reward_type', type=str, default="game", help="select game or molecule")
    parser.add_argument('--terminal', action="store_true", help="whether using the no-modification action as termination")
    parser.add_argument('--discount_penalty', type=float, default=0.8, help="used for molecule modification-based reward design")
    
    parser.add_argument('--mod_pos_penalty', type=float, default=1, help="penalty for each step")
    parser.add_argument('--no_mod_penalty', type=float, default=-0.5, help="penalty for no modification")
    parser.add_argument('--mod_neg_penalty', type=float, default=-1, help="penalty for negative modification")
    
    parser.add_argument('--allow_imm_rew', type=int, default=None, help="whether use immediate reward or not: (None represent using imm reward; 0 represent not using imm reward")
    parser.add_argument('--allow_final_rew', action="store_false", help="whether use final reward or not")

    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--rate', type=float, default=10, help="weight of final reward")
    
    parser.add_argument('--anneal_nomod_step', type=int, default=10000)
    parser.add_argument('--anneal_nomod_rate', type=float, default=0.05)
    
    # environment
    parser.add_argument('--max_len', type=int, default=27, help="maximum number of steps")
    parser.add_argument('--use_step', action="store_true")
    parser.add_argument('--use_gmm', action="store_true")
    parser.add_argument('--score_stop_criteria', type=float, default=0.95, help="stop_criteria")
    parser.add_argument('--gmm_stop_criteria', type=float, default=1.2, help="stop_criteria")
    parser.add_argument('--num_envs', type=int, default=10, help="number of environments")
    parser.add_argument('--n_steps', type=int, default=20, help="number of roll out steps")
    parser.add_argument('--max_step', type=int, default=8, help="maximum number of steps")
    parser.add_argument('--good_sample_step', type=int, default=1, help="good sample step")
    parser.add_argument('--sample_rate', type=float, default=0.8, help='the rate of sampling from IEDB dataset')
    
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--hour', type=int, default=5)
    parser.add_argument('--max_size', type=int, default=50000)
    
    args = parser.parse_args()
    t1 = time.time()
    
    peptides = [peptide.strip() for peptide in open(args.peptide_path, 'r').readlines()]
    
    action_space = gym.spaces.multi_discrete.MultiDiscrete([args.max_len, 20])
    if args.use_step:
        observation_space = gym.spaces.MultiDiscrete([20] * (26 + args.max_len))
    else:
        observation_space = gym.spaces.MultiDiscrete([20] * (25 + args.max_len))
    
    m_env_kwargs = {"action_space":action_space, "observation_space":observation_space, \
                "args": args, "max_tcr_len": args.max_len} 
    reward_model = Reward(args.beta, args.gmm_stop_criteria, ergo_model_file=args.ergo_model)
    
    m_env = make_vec_env(TCREnv, n_envs=args.num_envs, env_kwargs=m_env_kwargs, vec_env_cls=MySubprocVecEnv)
    m_env.set_reward_model(reward_model)
    
    model = PPO.load(args.path,env=m_env)
     
    results = {peptide: [] for peptide in peptides}
    
    rollout_peptides = [peptide for peptide in peptides] * args.rollout
    #rollout_alleles = [allele for tmp in rollout_alleles for allele in tmp]
    
    batch_peptides = rollout_peptides[:args.num_envs]
    batch_idxs = np.arange(args.num_envs)
    
    
    obs = m_env.reset([True] * len(batch_peptides), batch_peptides, [None] * len(batch_peptides))
    rollout = 0

    st_time = time.time()
    num = len(peptides)
    removed_peptides = []
    while rollout < len(rollout_peptides):
        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, config.device)
            actions, values, log_probs = model.policy.forward(obs_tensor)
        
        actions = actions.cpu().numpy()

        new_obs, rewards, dones, infos = m_env.step( actions )
    
        for info in infos:
            for key, value in info.items():
                if key == "episode" or key == "terminal_observation": continue
                #pdb.set_trace()
                if type(value) == float:
                    print("%s: %.4f; " % (key, value), end='')
                else:
                    print("%s: %s; " % (key, value), end='')
            print("\n")
    
        for idx, done in enumerate(dones):
            if done:
                peptide = rollout_peptides[batch_idxs[idx]]
                #if allele != infos[idx]['allele']: pdb.set_trace()
                if max(batch_idxs) < len(rollout_peptides) - 1:
                    batch_idxs[idx] = max(batch_idxs) + 1
                    batch_peptides[idx] = rollout_peptides[batch_idxs[idx]]
                
                rollout += 1

                if rollout < len(rollout_peptides):
                    if args.sample_rate == 0:
                        results[peptide].append((infos[idx]['new_tcr'], infos[idx]['score'], infos[idx]['score1'], infos[idx]['score2']))
                    else:
                        results[peptide].append((infos[idx]['init_tcr'], infos[idx]['new_tcr'], infos[idx]['score'], infos[idx]['score1'], infos[idx]['score2']))
                else:
                    break
       
        obs = m_env.reset(dones, batch_peptides, new_obs) 
        ck_time = time.time()
        for peptide in results:
            if peptide not in removed_peptides and len(results[peptide]) >= args.max_size:
                num -= 1
                removed_peptides.append(peptide)
                print("time cost for peptide %s: %.4f" % (peptide, ck_time - st_time))

        if num == 0: break
        if ck_time - st_time >= args.hour * 3600: break
        
    m_env.close()
    output = open(args.out, 'w')
        
    for peptide in peptides:
        for result in results[peptide]:
            if args.sample_rate == 0:
                output.write("%s %s %.4f %.4f %.4f\n" % (peptide, result[0], \
                      float(result[1]), float(result[2]), float(result[3])))
            else:
                output.write("%s %s %s %.4f %.4f %.4f\n" % (peptide, result[0], result[1], \
                      float(result[2]), float(result[3]), float(result[4])))
    
    output.close()
    print("time cost: %.4f" % (time.time()-t1))
