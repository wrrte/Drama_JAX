import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
import wandb


def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])



class WandbLogger:
    def __init__(self, config, project=None, mode='online'):
        """
        Initialize the Logger class.
        Args:
            path (str): Path to the directory where logs will be saved. This can be used to define the run name in W&B.
            project (str, optional): Name of the W&B project. Defaults to None.
        """
        # Initialize a W&B run with the given project and path as the run name
        pure_env_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
        run_id = wandb.util.generate_id()
        run_name = f"{pure_env_name}_{run_id}"
        # Initialize wandb with the complete name (including run ID will be auto-appended by wandb)
        api_key_path = os.path.join(os.path.dirname(__file__), '.wandb_api_key')
        try:
            with open(api_key_path, 'r') as f:
                wandb.login(key=f.read().strip())
        except FileNotFoundError:
            pass

        self.run = wandb.init(project=project, config=config, mode=mode, name=run_name, id=run_id)
        self.tag_step = {}

    def log(self, tag, value, global_step):
        """
        Log data to Weights & Biases.

        Args:
            tag (str): The tag or label for the data being logged.
            value: The data to be logged. It can be a scalar, image, histogram, or video.
        """
        # Log data based on the type
        if "video" in tag:
            # Log video
            wandb.log({tag: wandb.Video(value, fps=20, format='mp4')}, step=global_step)
            
            # Local MP4 Saving (DreamerV3 style)
            if hasattr(self, 'logdir') and self.logdir:
                try:
                    import imageio
                    import numpy as np
                    import os
                    
                    vid = np.transpose(value, (0, 2, 3, 1))
                    if vid.dtype != np.uint8:
                        vid = (255 * np.clip(vid, 0, 1)).astype(np.uint8)
                    
                    scale = max(1, int(np.round(512 / vid.shape[1])))
                    if scale > 1:
                        vid = np.repeat(np.repeat(vid, scale, axis=1), scale, axis=2)
                        
                    safe_name = tag.replace('/', '_')
                    filename = os.path.join(self.logdir, f"{global_step}_{safe_name}.mp4")
                    os.makedirs(self.logdir, exist_ok=True)
                    imageio.mimsave(filename, vid, fps=20, macro_block_size=1, quality=10, pixelformat='yuv444p')
                except Exception as e:
                    print(f"Failed to save video locally: {e}")
        elif "images" in tag:
            # Log images
            images = [wandb.Image(img) for img in value]  # Convert each image to a wandb.Image
            wandb.log({tag: images}, step=global_step)
        elif "hist" in tag:
            # Log histogram
            wandb.log({tag: wandb.Histogram(value)}, step=global_step)
        else:
            # Log scalar value
            wandb.log({tag: value}, step=global_step)

    def update_config(self, update_dict):
        """
        Update the configuration with the given parameters.

        Args:
            update_dict (dict): A dictionary containing scalar parameter information to update in the configuration.
        """
        # Update the configuration using wandb.config.update
        wandb.config.update(update_dict)

    def close(self):
        """
        Finalize and close the W&B run.
        """
        # Finish the run
        wandb.finish()



class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar
