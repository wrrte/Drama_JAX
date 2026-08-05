import os
import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
from collections import deque
from tqdm import tqdm
import colorama

import pandas as pd

from utils import seed_np_torch, WandbLogger
from replay_buffer import ReplayBuffer
import agents
from sub_models.world_models import WorldModel
from line_profiler import profile
import yaml
from envs.my_memory_maze import MemoryMaze
from envs.my_atari import Atari
from envs.my_dmc import DMControl
from eval import eval_episodes
import warnings
import ast
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval import RetrievalContextManager

@profile
def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, batch_length, logger, epoch, global_step, agent=None, retrieval_manager=None):
    epoch_reconstruction_loss_list = []
    epoch_reward_loss_list = []
    epoch_termination_loss_list = []
    epoch_dynamics_loss_list = []
    epoch_dynamics_real_kl_div_list = []
    epoch_representation_loss_list = []
    epoch_representation_real_kl_div_list = []
    epoch_total_loss_list = []
    
    num_trig_pos = 0
    num_trig_neg = 0
    
    for e in range(epoch):
        obs, action, reward, termination, indexes = replay_buffer.sample(batch_size, batch_length, imagine=False)
        reconstruction_loss, reward_loss, termination_loss, \
        dynamics_loss, dynamics_real_kl_div, representation_loss, \
        representation_real_kl_div, total_loss, latent = world_model.update(obs, action, reward, termination, global_step=global_step, epoch_step=e, logger=logger)
        
        if retrieval_manager is not None and agent is not None and retrieval_manager.enabled:
            with torch.no_grad():
                # Get agent value (v_t) from latent
                # We must use agent.value to get the decoded scalar value instead of the raw symlog logits
                v_t = agent.value(latent.detach()) # [B, T]
            
            base_indexes = indexes[:, 0]
            if isinstance(base_indexes, torch.Tensor):
                base_indexes = base_indexes.cpu().numpy()
            base_envs = np.zeros_like(base_indexes)
            
            num_trig_pos, num_trig_neg = retrieval_manager.add_batch_transitions(
                v_t, base_indexes, base_envs, replay_buffer.max_length, skip_len=8
            )

        epoch_reconstruction_loss_list.append(reconstruction_loss)
        epoch_reward_loss_list.append(reward_loss)
        epoch_termination_loss_list.append(termination_loss)
        epoch_dynamics_loss_list.append(dynamics_loss)
        epoch_dynamics_real_kl_div_list.append(dynamics_real_kl_div)
        epoch_representation_loss_list.append(representation_loss)
        epoch_representation_real_kl_div_list.append(representation_real_kl_div)
        epoch_total_loss_list.append(total_loss)
    if logger is not None:
        logger.log("WorldModel/reconstruction_loss", np.mean(epoch_reconstruction_loss_list), global_step=global_step)
        # logger.log("WorldModel/augmented_reconstruction_loss", augmented_reconstruction_loss.item(), global_step=global_step)
        logger.log("WorldModel/reward_loss",np.mean(epoch_reward_loss_list), global_step=global_step)
        logger.log("WorldModel/termination_loss", np.mean(epoch_termination_loss_list), global_step=global_step)
        logger.log("WorldModel/dynamics_loss", np.mean(epoch_dynamics_loss_list), global_step=global_step)
        logger.log("WorldModel/dynamics_real_kl_div", np.mean(epoch_dynamics_real_kl_div_list), global_step=global_step)
        logger.log("WorldModel/representation_loss", np.mean(epoch_representation_loss_list), global_step=global_step)
        logger.log("WorldModel/representation_real_kl_div", np.mean(epoch_representation_real_kl_div_list), global_step=global_step)
        logger.log("WorldModel/total_loss", np.mean(epoch_total_loss_list), global_step=global_step)
        
        if retrieval_manager is not None and retrieval_manager.enabled:
            logger.log("Retrieval/triggered_anchors_step", num_trig_pos + num_trig_neg, global_step=global_step)

    return num_trig_pos, num_trig_neg    

@profile
@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size,
                             imagine_context_length, imagine_batch_length,
                             retrieval_manager=None):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()
    
    retrieved_count = 0
    lazy_hit_rate = 0.0
    
    if retrieval_manager is not None and retrieval_manager.enabled:
        # We need max_contexts to not exceed imagine_batch_size
        max_anchors = imagine_batch_size // 4
        ret_obs, ret_action, _, lazy_hit_rate, ret_indexes = retrieval_manager.retrieve_contexts(
            replay_buffer, world_model, max_anchors, multiplier=5, target=5, max_contexts=imagine_batch_size
        )
        if ret_obs is not None:
            retrieved_count = ret_obs.shape[0]
            # Update imagined counter for retrieved frames
            if len(ret_indexes) > 0:
                ret_indexes_arr = np.array(ret_indexes)
                replay_buffer.imagined_counter[ret_indexes_arr] += 1
    
    random_batch_size = max(1, imagine_batch_size - retrieved_count)
    
    sample_obs, sample_action, sample_reward, sample_termination, sample_indexes = replay_buffer.sample(
        random_batch_size, imagine_context_length, imagine=True, fetch_future_length=0)
        
    if retrieved_count > 0:
        sample_obs = torch.cat([sample_obs, ret_obs], dim=0)
        sample_action = torch.cat([sample_action, ret_action], dim=0)

    actual_batch_size = sample_obs.shape[0]

    if world_model.model == 'Transformer':
        latent, action, old_logits, context_latent, reward_hat, termination_hat = world_model.imagine_data(
            agent, sample_obs, sample_action,
            imagine_batch_size=actual_batch_size,
            imagine_context_length=imagine_context_length,
            imagine_batch_length=imagine_batch_length
        )
    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
         latent, action, old_logits, context_latent, reward_hat, termination_hat = world_model.imagine_data2(
            agent, sample_obs, sample_action,
            imagine_batch_size=actual_batch_size,
            imagine_context_length=imagine_context_length,
            imagine_batch_length=imagine_batch_length
        )
    return latent, action, old_logits, context_latent, sample_reward, sample_termination, reward_hat, termination_hat, lazy_hit_rate

@profile
def joint_train_world_model_agent(config, logdir,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  logger):
    
    latent_dim = config.Models.WorldModel.CategoricalDim * config.Models.WorldModel.ClassDim
    retrieval_manager = RetrievalContextManager(latent_dim=latent_dim, num_envs=1, config=config.JointTrainAgent.get("Retrieval", {}), device=world_model.device)
    os.makedirs(f"{logdir}/ckpt", exist_ok=True)


    if config.BasicSettings.Env_name.startswith('ALE'):
        env = Atari(config.BasicSettings.Env_name, size=config.BasicSettings.ImageSize, seed=config.BasicSettings.Seed, noops=getattr(config.BasicSettings, 'Noops', 0))
    elif config.BasicSettings.Env_name.startswith('memory'):
        env = MemoryMaze(config.BasicSettings.Env_name, size=config.BasicSettings.ImageSize, seed=config.BasicSettings.Seed)
    elif config.BasicSettings.Env_name.startswith('dm_'):
        # Parse dm_control environment name: dm_domain_task
        # Example: dm_cheetah_run, dm_walker_walk, dm_humanoid_stand
        parts = config.BasicSettings.Env_name.split('_')
        domain_name = parts[1]
        task_name = '_'.join(parts[2:])  # Handle multi-word tasks
        env = DMControl(
            domain_name=domain_name,
            task_name=task_name,
            action_repeat=config.BasicSettings.ActionRepeat if hasattr(config.BasicSettings, 'ActionRepeat') else 2,
            size=config.BasicSettings.ImageSize,
            camera_id=config.BasicSettings.CameraId if hasattr(config.BasicSettings, 'CameraId') else 0,
            seed=config.BasicSettings.Seed,
            length=config.BasicSettings.MaxEpisodeSteps if hasattr(config.BasicSettings, 'MaxEpisodeSteps') else 1000
        )
        is_discrete = False
    else:
        assert ValueError(f'Unknown environment name: {config.BasicSettings.Env_name}')
    
    is_discrete = hasattr(env.action_space, 'discrete') and env.action_space.discrete
    if is_discrete:
        action_dim = int(env.action_space.n)
    else:
        action_dim = int(env.action_space.shape[0])
    print("Current env: " + colorama.Fore.YELLOW + f"{config.BasicSettings.Env_name}" + colorama.Style.RESET_ALL)

    # Benchmark handling (only for Atari)
    if config.BasicSettings.Env_name.startswith('ALE'):
        atari_benchmark_df = pd.read_csv("atari_performance.csv", index_col='Task', usecols=lambda column: column in ['Task', 'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'])
        atari_pure_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
        game_benchmark_df = atari_benchmark_df.get(atari_pure_name)
    else:
        game_benchmark_df = None
    
    sum_reward = 0
    current_ob, info = env.reset()
    context_obs_seq = None
    context_action_seq = None
    max_context_len = config.JointTrainAgent.RealityContextLength

    # sample and train
    for total_steps in tqdm(range(config.JointTrainAgent.SampleMaxSteps // config.JointTrainAgent.NumEnvs), desc='Training'):
        # sample part >>>
        current_latent_for_hash = None
        if replay_buffer.ready('world_model'):
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if context_action_seq is None:
                    action_numpy = env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(context_obs_seq)
                    current_latent_for_hash = context_latent[:, -1]
                    
                    if is_discrete:
                        model_context_action = rearrange(context_action_seq, "L -> 1 L")
                    else:
                        model_context_action = rearrange(context_action_seq, "L A -> 1 L A")
                    
                    if world_model.model == 'Transformer' or world_model.model == 'Mamba' or world_model.model == 'Mamba2':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                        
                    action_tensor = agent.sample_as_env_action_tensor(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )[0]

                    if is_discrete:
                        action_numpy = action_tensor.cpu().item()
                    else:
                        action_numpy = action_tensor.cpu().numpy()

            new_ob = rearrange(torch.Tensor(current_ob).to(world_model.device), "H W C -> 1 1 C H W") / 255.0
            if context_obs_seq is None:
                context_obs_seq = new_ob
            else:
                context_obs_seq = torch.cat([context_obs_seq, new_ob], dim=1)
                if context_obs_seq.shape[1] > max_context_len:
                    context_obs_seq = context_obs_seq[:, 1:]
            
            if is_discrete:
                if context_action_seq is None:
                    action_tensor_seq = torch.tensor([action_numpy], device=world_model.device, dtype=torch.long)
                else:
                    action_tensor_seq = action_tensor.view(1)
            else:
                if context_action_seq is None:
                    action_tensor_seq = torch.tensor(action_numpy, device=world_model.device, dtype=torch.float32).view(1, -1)
                else:
                    action_tensor_seq = action_tensor.view(1, -1)

            if context_action_seq is None:
                context_action_seq = action_tensor_seq
            else:
                context_action_seq = torch.cat([context_action_seq, action_tensor_seq], dim=0)
                if context_action_seq.shape[0] > max_context_len:
                    context_action_seq = context_action_seq[1:]
        else:
            action_numpy = env.action_space.sample()

        ob, reward, is_last, info = env.step(action_numpy)
        replay_buffer.append(current_ob, action_numpy, reward, info['is_terminal'])
        
        if replay_buffer.ready('world_model') and current_latent_for_hash is not None and retrieval_manager.enabled:
            retrieval_manager.add_transition(
                pointer=replay_buffer.last_pointer,
                env_idx=0,
                latent_b=current_latent_for_hash
            )

        sum_reward += reward
        current_ob = ob

        if is_last:
            logger.log(f"episode/score", sum_reward, global_step=total_steps)
            logger.log(f"episode/length", info["episode_frame_number"], global_step=total_steps)  # framskip=4
            if config.BasicSettings.Env_name.startswith('ALE'):
                logger.log(f"episode/normalised score", (sum_reward - game_benchmark_df['Random'])/(game_benchmark_df['Human'] - game_benchmark_df['Random']), global_step=total_steps)
                for algorithm in game_benchmark_df.index[2:]:
                    denominator = game_benchmark_df[algorithm] - game_benchmark_df['Random']
                    if denominator != 0:
                        normalized_score = (sum_reward - game_benchmark_df['Random']) / denominator
                        logger.log(f"benchmark/normalised {algorithm} score", normalized_score, global_step=total_steps)
            
            sum_reward = 0
            ob, info = env.reset()
            context_obs_seq = None
            context_action_seq = None



        if replay_buffer.ready('world_model') and total_steps % (config.JointTrainAgent.TrainDynamicsEverySteps // config.JointTrainAgent.NumEnvs) == 0 and total_steps <= config.JointTrainAgent.FreezeWorldModelAfterSteps:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=config.JointTrainAgent.BatchSize,
                batch_length=config.JointTrainAgent.BatchLength,
                logger=logger,
                epoch=config.JointTrainAgent.TrainDynamicsEpoch,
                global_step=total_steps,
                agent=agent,
                retrieval_manager=retrieval_manager
            )


        if replay_buffer.ready('behaviour') and total_steps % (config.JointTrainAgent.TrainAgentEverySteps // config.JointTrainAgent.NumEnvs) == 0 and total_steps <= config.JointTrainAgent.FreezeBehaviourAfterSteps:
            log_video = total_steps % (config.JointTrainAgent.SaveEverySteps // config.JointTrainAgent.NumEnvs) == 0

            if log_video:
                video_columns = getattr(config.BasicSettings, "VideoColumns", 5)
                video_total_length = getattr(config.BasicSettings, "VideoTotalLength", 64)
                video_temporal_length = getattr(config.BasicSettings, "VideoTemporalLength", 5)
                imagine_len = config.JointTrainAgent.ImagineBatchLength
                context_len = max(1, video_total_length - imagine_len)
                num_videos = video_columns * video_temporal_length
                
                openloop_obs, openloop_action, _, _, _ = replay_buffer.sample(
                    num_videos, context_len, imagine=True, fetch_future_length=imagine_len)
                
                world_model.log_openloop_video(
                    openloop_obs, openloop_action, context_len, imagine_len, logger, total_steps, video_columns=video_columns)

            imagine_latent, agent_action, old_logits, context_latent, context_reward, context_termination, imagine_reward, imagine_termination, lazy_hit_rate = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=config.JointTrainAgent.ImagineBatchSize,
                imagine_context_length=config.JointTrainAgent.ImagineContextLength,
                imagine_batch_length=config.JointTrainAgent.ImagineBatchLength,
                retrieval_manager=retrieval_manager
            )
            
            if retrieval_manager.enabled:
                logger.log("Retrieval/lazy_hit_rate", lazy_hit_rate, global_step=total_steps)

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logits=old_logits,
                context_latent=context_latent,
                context_reward=context_reward,
                context_termination=context_termination,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger,
                global_step=total_steps
            )

        eval_after_steps = getattr(config.Evaluate, 'AfterSteps', 0) // config.JointTrainAgent.NumEnvs
        if config.Evaluate.DuringTraining and total_steps >= eval_after_steps and total_steps % (config.Evaluate.EverySteps // config.JointTrainAgent.NumEnvs) == 0:
            _ = eval_episodes(config, world_model, agent, logger, total_steps)
        if config.JointTrainAgent.SaveModels and total_steps % (config.JointTrainAgent.SaveEverySteps // config.JointTrainAgent.NumEnvs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"{logdir}/ckpt/world_model.pth")
            torch.save(agent.state_dict(), f"{logdir}/ckpt/agent.pth")



def build_world_model(conf, action_dim, device):
    return WorldModel(
        action_dim = action_dim,
        config = conf, 
        device = device
    ).cuda(device)


def build_agent(conf, action_dim, device):
    if conf.Models.Agent.Policy == 'AC':
        return agents.ActorCriticAgent(
            conf = conf,
            action_dim=action_dim,
            device = device
        ).cuda(device)
    elif conf.Models.Agent.Policy == 'PPO':
        return agents.PPOAgent(
            conf=conf,
            action_dim=action_dim,
            device = device
        ).cuda(device)        


class DotDict(dict):
    """Dictionary with dot notation access."""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{item}'")
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def update_or_create(self, key_path, value):
        keys = key_path.split('.')
        d = self
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = DotDict()
            d = d[key]
        d[keys[-1]] = value

# Function to parse and update config from arguments
def parse_args_and_update_config(config, prefix=''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_files/configure.yaml', help='Path to the config file')

    # Map string dtype to torch dtype
    def dtype_mapper(dtype_str):
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        return dtype_map[dtype_str]

    def add_arguments(config, prefix=''):
        for key, value in config.items():
            if isinstance(value, dict):
                add_arguments(value, prefix + key + '.')
            elif isinstance(value, bool):
                # Special handling for boolean arguments
                parser.add_argument(f'--{prefix}{key}', type=lambda x: x.lower() in ['true', '1', 'yes'], default=value)
            elif key == 'dtype':
                # Special handling for dtype arguments
                parser.add_argument(f'--{prefix}{key}', type=dtype_mapper, default=value)
            elif isinstance(value, (list, dict)):
                # Use a custom converter for list/dict-like arguments
                parser.add_argument(f'--{prefix}{key}', type=lambda x: ast.literal_eval(x), default=value)
            else:
                parser.add_argument(f'--{prefix}{key}', type=type(value), default=value)

    def update_dict(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    add_arguments(config, prefix)
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    for arg_key, arg_value in args_dict.items():
        if arg_value is not None:
            keys = arg_key.split('.')
            update_dict(config, keys, arg_value)
    
    return config

def update_model_parameters(config, world_model, agent):
    config.update_or_create('Models.WorldModel.TotalParamNum', sum([p.numel() for p in world_model.parameters()]))
    print(f'World model total parameters: {sum([p.numel() for p in world_model.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.BackboneParamNum', sum([p.numel() for p in world_model.sequence_model.parameters()]))
    print(f'Dynamic model parameters: {sum([p.numel() for p in world_model.sequence_model.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.EncoderParamNum', sum([p.numel() for p in world_model.encoder.parameters()]))
    print(f'Encoder parameters: {sum([p.numel() for p in world_model.encoder.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.DecoderParamNum', sum([p.numel() for p in world_model.image_decoder.parameters()]))
    print(f'Decoder parameters: {sum([p.numel() for p in world_model.image_decoder.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.DiscretisationLayerParamNum', sum([p.numel() for p in world_model.dist_head.parameters()]))
    print(f'Discretisation layer parameters: {sum([p.numel() for p in world_model.dist_head.parameters()]):,}')
    
    config.update_or_create('Models.Agent.ActorParamNum', sum([p.numel() for p in agent.actor.parameters()]))
    print(f'Actor parameters: {sum([p.numel() for p in agent.actor.parameters()]):,}')
    
    config.update_or_create('Models.Agent.CriticParamNum', sum([p.numel() for p in agent.critic.parameters()]))
    print(f'Critic parameters: {sum([p.numel() for p in agent.critic.parameters()]):,}')

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    with open('config_files/configure.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config = parse_args_and_update_config(config)   
    config = DotDict(config)

    device = torch.device(config.BasicSettings.Device)
    # set seed
    seed_np_torch(seed=config.BasicSettings.Seed)

    
    # getting action_dim with dummy env
    if config.BasicSettings.Env_name.startswith('ALE'):
        dummy_env = Atari(config.BasicSettings.Env_name)
    elif config.BasicSettings.Env_name.startswith('memory'):
        dummy_env = MemoryMaze(config.BasicSettings.Env_name)
    elif config.BasicSettings.Env_name.startswith('dm_'):
        parts = config.BasicSettings.Env_name.split('_')
        domain_name = parts[1]
        task_name = '_'.join(parts[2:])
        dummy_env = DMControl(domain_name=domain_name, task_name=task_name)
    else:
        raise ValueError(f'Unknown environment name: {config.BasicSettings.Env_name}')

    action_dim = dummy_env.action_space.n if hasattr(dummy_env.action_space, 'n') else dummy_env.action_space.shape[0]
    is_discrete = hasattr(dummy_env.action_space, 'discrete') and dummy_env.action_space.discrete

    # build world model and agent
    world_model = build_world_model(config, action_dim, device=device)
    agent = build_agent(config, action_dim, device=device)
    update_model_parameters(config, world_model, agent)
    if (config.BasicSettings.Compile and os.name != "nt"):  # compilation is not supported on windows
        world_model.encoder = torch.compile(world_model.encoder, fullgraph=True, dynamic=True)
        world_model.dist_head.forward_prior = torch.compile(world_model.dist_head.forward_prior, fullgraph=True, dynamic=True)
        world_model.dist_head.forward_post = torch.compile(world_model.dist_head.forward_post, fullgraph=True, dynamic=True)
        world_model.image_decoder = torch.compile(world_model.image_decoder, fullgraph=True, dynamic=True)
        world_model.reward_decoder = torch.compile(world_model.reward_decoder, fullgraph=True, dynamic=True)
        world_model.termination_decoder = torch.compile(world_model.termination_decoder, fullgraph=True, dynamic=True)
        
        world_model.mse_loss_func = torch.compile(world_model.mse_loss_func, fullgraph=True, dynamic=True)
        world_model.symlog_twohot_loss_func = torch.compile(world_model.symlog_twohot_loss_func, fullgraph=True, dynamic=True)
        world_model.categorical_kl_div_loss = torch.compile(world_model.categorical_kl_div_loss, fullgraph=True, dynamic=True)
        world_model.bce_with_logits_loss_func = torch.compile(world_model.bce_with_logits_loss_func, fullgraph=True, dynamic=True)
        
        world_model.stright_throught_gradient = torch.compile(world_model.stright_throught_gradient, fullgraph=True, dynamic=True)
        
        # Compile agent modules
        agent.actor = torch.compile(agent.actor, fullgraph=True, dynamic=True)
        agent.critic = torch.compile(agent.critic, fullgraph=True, dynamic=True)
        agent.sample_as_env_action_tensor = torch.compile(agent.sample_as_env_action_tensor, fullgraph=True, dynamic=True)
    if config.BasicSettings.SavePath != 'None':
        print('Loading models')
        world_model.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/world_model.pth"))
        agent.load_state_dict(torch.load(f"{config.BasicSettings.SavePath}/agent.pth"))
   
    logger = WandbLogger(config=config, project=config.Wandb.Init.Project, mode=config.Wandb.Init.Mode)
    logdir = f"./saved_models/{config.n}/{config.BasicSettings.Env_name}/{logger.run.id}"
    logger.logdir = logdir
    # build replay buffer
    replay_buffer = ReplayBuffer(
        config,
        device=device,
        action_dim=action_dim,
        is_discrete=is_discrete
    )

    # train
    joint_train_world_model_agent(config, logdir, replay_buffer, world_model, agent, logger)

    logger.close()

