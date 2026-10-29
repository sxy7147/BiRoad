import os
import time
import json

import glob
import random
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pyrep.objects import Dummy, VisionSensor
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, assert_action_shape
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError, TaskEnvironmentError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from modeling.encoder.text import fetch_tokenizers
from datasets.rlbench import strip_partial_role_language_markers
from online_evaluation_rlbench.get_stored_demos import get_stored_demos

EVAL_RESET_MAX_RETRIES = int(os.environ.get("EVAL_RESET_MAX_RETRIES", "5"))
EVAL_RESTART_ENV_ON_RESET_RETRY = os.environ.get(
    "EVAL_RESTART_ENV_ON_RESET_RETRY",
    "true",
).strip().lower() not in {"0", "false", "no", "off"}
EVAL_STATIC_POSITIONS = os.environ.get("EVAL_STATIC_POSITIONS", "auto").strip().lower()


def _resolve_eval_static_positions(task_str):
    if EVAL_STATIC_POSITIONS in {"1", "true", "yes", "on"}:
        return True
    if EVAL_STATIC_POSITIONS in {"0", "false", "no", "off"}:
        return False
    return "put_item_in_drawer" in str(task_str)

def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class TaskRecorder(object):
    def __init__(
        self, scene, fps=10, image_size=(128, 128), frame_skip=5,
        camera_name=None
    ):
        self._scene = scene
        self._fps = fps
        self._image_size = image_size
        self._frame_skip = frame_skip
        self._camera_name = camera_name
        self._step_count = 0
        self._frames = []
        self._cam = None
        self._create_recording_camera()

        if self._cam is not None:
            self._cam.set_resolution(self._image_size)

        # Monkey-patch Scene.step to support automatic recording
        if not hasattr(self._scene, '_original_step'):
            self._scene._original_step = self._scene.step
            def patched_step():
                self._scene._original_step()
                if hasattr(self._scene, '_step_callback') and self._scene._step_callback is not None:
                    self._scene._step_callback()
            self._scene.step = patched_step

    def _copy_camera_pose(self, source_camera_name):
        source_name = (
            source_camera_name
            if source_camera_name.startswith("cam_")
            else "cam_" + source_camera_name
        )
        source_cam = VisionSensor(source_name)
        self._cam = VisionSensor.create(self._image_size)
        self._cam.set_pose(source_cam.get_pose())

    def _create_recording_camera(self):
        if self._camera_name not in {None, "", "cinematic"}:
            try:
                self._copy_camera_pose(self._camera_name)
                return
            except Exception as e:
                print(
                    f"Warning: failed to create recorder camera from "
                    f"cam_{self._camera_name}: {e}. Falling back to cinematic.",
                    flush=True,
                )

        # Try to find a camera for recording, or create one like cinematic_recorder.py
        try:
            # Try to use existing cinematic placeholder if available
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._cam = VisionSensor.create(self._image_size)
            self._cam.set_pose(cam_placeholder.get_pose())

            # Fix upside-down issue: rotate the camera 180 degrees around its own Z-axis
            # or simply adjust the parent/pose relationship.
            # In RLBench, the placeholder might be oriented for V-REP's default view.
            self._cam.rotate([0, 0, np.pi])

            self._cam.set_parent(cam_placeholder)
        except Exception:
            # Fallback: copy any available camera pose without mutating that camera.
            for cam_name in ['front', 'over_shoulder_left', 'over_shoulder_right', 'overhead']:
                try:
                    self._copy_camera_pose(cam_name)
                    break
                except Exception:
                    continue

    def take_snap(self):
        self._step_count += 1
        if self._cam is not None and self._step_count % self._frame_skip == 0:
            rgb = self._cam.capture_rgb()
            rgb = (rgb * 255).astype(np.uint8)
            # PyRep returns images flipped vertically (bottom-up), flip them to top-down
            rgb = np.flip(rgb, 0)
            self._frames.append(rgb)

    def remove(self):
        if self._cam is not None:
            self._cam.remove()
            self._cam = None

    def save(self, path):
        if not self._frames:
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            import imageio
            # Use imageio to save video, which is more headless-friendly than cv2
            writer = imageio.get_writer(str(path), fps=self._fps, codec='libx264', quality=8)
            for frame in self._frames:
                writer.append_data(frame)
            writer.close()
        except ImportError:
            print("Warning: imageio not found. Falling back to saving as individual frames.")
            frame_dir = str(path).replace('.mp4', '_frames')
            os.makedirs(frame_dir, exist_ok=True)
            for i, frame in enumerate(self._frames):
                Image.fromarray(frame).save(os.path.join(frame_dir, f"{i:04d}.png"))
        
        self._frames = []


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        # action is an array (2, 8)
        obs = None
        terminate = None
        reward = 0

        # Try to reach the desired pose without changing the gripper state
        target = action.copy()
        if self._last_action is not None:
            action[:, 7] = self._last_action[:, 7].copy() # copy gripper state
        for _ in range(self._max_tries):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            # Peract2 takes (right, left) action, but we predict (left, right)
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

            # Check if we reached the desired pose (planner may be inaccurate)
            l_pos = obs.left.gripper_pose[:3]
            r_pos = obs.right.gripper_pose[:3]
            l_dist_pos = np.sqrt(np.square(target[0, :3] - l_pos).sum())
            r_dist_pos = np.sqrt(np.square(target[1, :3] - r_pos).sum())
            criteria = (l_dist_pos < 5e-3, r_dist_pos < 5e-3)

            if all(criteria) or reward == 1:
                break

        # Then execute with gripper action (open/close))
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and (  # if any gripper state has changed, re-execute
                action[0, 7] != self._last_action[0, 7]
                or action[1, 7] != self._last_action[1, 7]
            )
        ):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

        # Store the last action action for the gripper state
        self._last_action = action.copy()

        return obs, reward, terminate


class Actioner:

    def __init__(
        self, policy=None, backbone='clip', task_str=None,
        eval_instructions=None,
    ):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)
        self.task_str = task_str
        self._episode_info = {}
        self.eval_instruction_path = (
            Path(eval_instructions) if eval_instructions is not None else None
        )
        self.eval_instructions = None
        if self.eval_instruction_path is not None:
            with open(self.eval_instruction_path) as f:
                self.eval_instructions = json.load(f)

    def start_episode(self, task_str, variation, demo_id, episode_idx):
        self._episode_info = {
            "task": task_str,
            "variation": variation,
            "demo_id": demo_id,
            "episode_idx": episode_idx
        }

    def _sample_eval_instruction(self):
        task = self._episode_info.get("task", self.task_str)
        variation = self._episode_info.get("variation", 0)
        variation_key = "0" if int(variation) == -1 else str(int(variation))

        if task not in self.eval_instructions:
            raise KeyError(
                f"Task '{task}' not found in eval instructions: "
                f"{self.eval_instruction_path}"
            )
        if variation_key not in self.eval_instructions[task]:
            raise KeyError(
                f"Variation '{variation_key}' for task '{task}' not found "
                f"in eval instructions: {self.eval_instruction_path}"
            )

        instructions = self.eval_instructions[task][variation_key]
        if not instructions:
            raise ValueError(
                f"Empty instruction list for task '{task}', variation "
                f"'{variation_key}' in {self.eval_instruction_path}"
            )
        return random.choice(instructions)

    def load_episode(self, descriptions):
        if self.eval_instructions is not None:
            instr = self._sample_eval_instruction()
        else:
            instr = random.choice(descriptions)
            # Truncate instruction if it ends with specific phrases
            for suffix in [" with swapped hands", " using the other hand"]:
                if instr.endswith(suffix):
                    instr = instr[:-len(suffix)]
                    break

        if (
            self.task_str
            and self.task_str.endswith(("_random_left", "_random_right"))
        ):
            instr = strip_partial_role_language_markers(instr)

        self._instr = self.tokenizer([instr]).cuda(non_blocking=True)

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        """
        Args:
            rgbs: (1, ncam, 3, H, W)
            pcds: (1, ncam, 3, H, W)
            gripper: (1, nhist, 2*8)
            prediction_len: int

        Returns:
            trajectory: (1, nhist, nhand=2, 8)
        """
        output = self._policy(
            None,
            torch.full([1, prediction_len, 2], False).cuda(non_blocking=True),
            rgbs,
            None,
            pcds,
            self._instr,
            gripper.unflatten(-1, (2, -1)),  # (1, nhist, nhand=2, 8)
            run_inference=True,
        )
        return output


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right", "front"),
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_cameras = apply_cameras
        self.headless = headless
        self.static_positions = _resolve_eval_static_positions(task_str)
        print(
            f"[EVAL] RLBench static_positions={self.static_positions} "
            f"(EVAL_STATIC_POSITIONS={EVAL_STATIC_POSITIONS}) for task={task_str}",
            flush=True,
        )

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=BimanualEndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=HandoverDiscrete() if 'handover' in task_str else BimanualDiscrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless, static_positions=self.static_positions,
            robot_setup="dual_panda"
        )

    def _restart_env_and_get_task(self, task_str):
        print(
            f"[EVAL] Restarting RLBench environment for task={task_str}",
            flush=True,
        )
        try:
            self.env.shutdown()
        except Exception as e:
            print(f"[EVAL] env.shutdown during restart raised: {e}", flush=True)
        self.env = Environment(
            self.action_mode,
            str(self.data_path),
            self.obs_config,
            headless=self.headless,
            static_positions=self.static_positions,
            robot_setup="dual_panda",
        )
        self.env.launch()
        return self.env.get_task(task_file_to_task_class(task_str))

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        rgb = torch.stack([
            torch.tensor(obs.perception_data["{}_rgb".format(cam)]).float().permute(2, 0, 1) / 255.0
            for cam in self.apply_cameras
        ]).unsqueeze(0)  # 1, N, C, H, W
        pcd = torch.stack([
            torch.tensor(obs.perception_data["{}_point_cloud".format(cam)]).float().permute(2, 0, 1)
            for cam in self.apply_cameras
        ]).unsqueeze(0)  # 1, N, C, H, W
        # action is an array of length 16 = (7+1)*2
        gripper = torch.from_numpy(np.concatenate([
            obs.left.gripper_pose, [obs.left.gripper_open],
            obs.right.gripper_pose, [obs.right.gripper_open]
        ])).float().unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1,
        num_demos=-1,
        from_episode_number=0,
        save_video=False,
        video_dir=None,
        video_resolution=(128, 128),
        video_freq=5,
        video_camera=None
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        
        # Check for standard RLBench structure: task/variationXcd
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = [
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        ]

        # If standard structure not found, check for all_variations folder
        if len(task_variations) == 0:
            all_var_path = os.path.join(self.data_path, task_str, "all_variations")
            if os.path.exists(all_var_path):
                # If all_variations exists, we treat it as variation 0 or handle accordingly
                # For compatibility, we might assume variation 0 contains the episodes
                print(f"Found all_variations for {task_str}, assuming mixed variations.")
                task_variations = [-1] 
            else:
                print(f"Warning: No variations found for task {task_str} in {self.data_path}")
                self.env.shutdown()
                return {"mean": 0.0}

        if len(task_variations) > num_demos:
            task_variations = task_variations[:num_demos]
            num_demos_per_variation = 1 
        else:
            num_demos_per_variation = max(1, num_demos // len(task_variations))

        var_success_rates = {}
        var_num_valid_demos = {}
        
        episode_idx = 0

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            success_rate, valid, num_valid_demos, episode_idx, task = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history,
                    num_demos=num_demos_per_variation,
                    from_episode_number=from_episode_number,
                    save_video=save_video,
                    video_dir=video_dir,
                    video_resolution=video_resolution,
                    video_freq=video_freq,
                    video_camera=video_camera,
                    episode_idx=episode_idx
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,  # this is str
        task,  # this instance of TaskEnvironment
        max_steps,
        variation,
        actioner,
        max_tries=1,
        prediction_len=50,
        num_history=1,
        num_demos=-1,
        from_episode_number=0,
        save_video=False,
        video_dir=None,
        video_resolution=(128, 128),
        video_freq=5,
        video_camera=None,
        episode_idx=0
    ):
        success_rate = 0
        total_reward = 0
        start_time = time.time()
        var_demos = get_stored_demos(
            amount=-1,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=from_episode_number
        )
        if num_demos is not None and num_demos > 0:
            var_demos = var_demos[:num_demos]
        print('num demos:', len(var_demos))
        reset_failed_demos = 0

        for demo_id, demo in enumerate(var_demos):
            recorder = None
            should_save_this_video = save_video and (episode_idx % video_freq == 0)
            if should_save_this_video and video_dir is not None:
                # Use user-defined resolution and set frame_skip to 5 for faster video
                recorder = TaskRecorder(
                    self.env._scene, fps=10, image_size=video_resolution,
                    frame_skip=5, camera_name=video_camera
                )
                # Use RLBench native scene step callback for automatic capture
                self.env._scene._step_callback = recorder.take_snap
                
                # Enable higher frequency recording during gripper actions for better video quality
                self.env._scene.record_gripper_closing = True

            grippers = torch.Tensor([]).cuda(non_blocking=True)
            # descriptions, obs = task.reset_to_demo(demo)
            reset_error = None
            descriptions = obs = None
            
            demo_variation = getattr(demo, "variation_number", variation)
            for reset_attempt in range(max(1, EVAL_RESET_MAX_RETRIES)):
                try:
                    if variation == -1:
                        task.set_variation(int(demo_variation))
                    descriptions, obs = task.reset_to_demo(demo)
                    reset_error = None
                    if reset_attempt > 0:
                        print(
                            "[EVAL] RLBench reset retry succeeded. "
                            f"task={task_str} variation={demo_variation} "
                            f"demo_id={demo_id} attempt={reset_attempt + 1}/"
                            f"{max(1, EVAL_RESET_MAX_RETRIES)}",
                            flush=True,
                        )
                    break
                except TaskEnvironmentError as e:
                    reset_error = e
                    print(
                        "[EVAL] RLBench reset failed; retrying. "
                        f"task={task_str} variation={demo_variation} "
                        f"demo_id={demo_id} attempt={reset_attempt + 1}/"
                        f"{max(1, EVAL_RESET_MAX_RETRIES)} error={e}",
                        flush=True,
                    )
                    is_last_attempt = reset_attempt + 1 >= max(1, EVAL_RESET_MAX_RETRIES)
                    if EVAL_RESTART_ENV_ON_RESET_RETRY and not is_last_attempt:
                        if recorder is not None:
                            recorder.remove()
                            recorder = None
                        task = self._restart_env_and_get_task(task_str)
                        if variation != -1:
                            task.set_variation(variation)
                        if should_save_this_video and video_dir is not None:
                            recorder = TaskRecorder(
                                self.env._scene,
                                fps=10,
                                image_size=video_resolution,
                                frame_skip=5,
                                camera_name=video_camera,
                            )
                            self.env._scene._step_callback = recorder.take_snap
                            self.env._scene.record_gripper_closing = True
            if reset_error is not None:
                reset_failed_demos += 1
                print(
                    "[EVAL] RLBench reset failed; marking demo invalid and continuing. "
                    f"task={task_str} variation={demo_variation} demo_id={demo_id} "
                    f"attempts={max(1, EVAL_RESET_MAX_RETRIES)} error={reset_error}",
                    flush=True,
                )
                if recorder is not None:
                    recorder.remove()
                    self.env._scene._step_callback = None
                    self.env._scene.record_gripper_closing = False
                elapsed_time = time.time() - start_time
                h = int(elapsed_time // 3600)
                m = int((elapsed_time % 3600) // 60)
                s = int(elapsed_time % 60)
                valid_demo_count = demo_id + 1 - reset_failed_demos
                print(
                    task_str,
                    "Variation",
                    variation,
                    "Demo",
                    demo_id,
                    "Reward",
                    "0.00",
                    "max_reward",
                    "0.00",
                    f"SR: {success_rate}/{valid_demo_count}",
                    f"Total Reward: {total_reward:.2f}/{valid_demo_count}",
                    f"Elapsed: {h:02d}:{m:02d}:{s:02d}",
                    "# valid demos",
                    valid_demo_count,
                    "# reset_failed",
                    reset_failed_demos,
                    "reset_failed",
                )
                continue
            if recorder is not None:
                recorder.take_snap()
            
            if hasattr(actioner, "start_episode"):
                actioner.start_episode(task_str, variation, demo_id, episode_idx)
            actioner.load_episode(descriptions)

            move = Mover(task, max_tries=max_tries)
            max_reward = 0.0

            for step_id in range(max_steps):

                # Fetch the current observation and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.cuda(non_blocking=True)
                pcds_input = pcd.cuda(non_blocking=True)
                gripper = gripper.cuda(non_blocking=True)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], 1)

                # Prepare proprioception history
                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, npad, 0), mode='replicate'
                )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len
                )

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    actions = output[-1].cpu().numpy()
                    actions[..., -1] = actions[..., -1].round()

                    # execute
                    for action in actions:
                        obs, reward, _ = move(action, collision_checking=False)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

            if recorder is not None:
                video_name = (
                            f"{task_str}_var{variation}"
                            f"_demo{from_episode_number + demo_id}"
                            f"_reward{max_reward}.mp4"
                )
                recorder.save(os.path.join(video_dir, video_name))
                recorder.remove()
                # Unset scene callback and gripper recording
                self.env._scene._step_callback = None
                self.env._scene.record_gripper_closing = False

            total_reward += max_reward
            episode_idx += 1
            elapsed_time = time.time() - start_time
            # Format elapsed time as HH:MM:SS
            h = int(elapsed_time // 3600)
            m = int((elapsed_time % 3600) // 60)
            s = int(elapsed_time % 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"Total Reward: {total_reward:.2f}/{demo_id+1}",
                f"Elapsed: {time_str}",
                "# valid demos", demo_id + 1
            )

        # # Compensate for failed demos
        # valid = len(var_demos) > 0
        valid_demo_count = len(var_demos) - reset_failed_demos
        if reset_failed_demos:
            print(
                f"[EVAL] skipped {reset_failed_demos} reset-failed demos for "
                f"task={task_str} variation={variation}; valid_demos={valid_demo_count}",
                flush=True,
            )

        valid = valid_demo_count > 0

        # return success_rate, valid, len(var_demos), episode_idx
        return success_rate, valid, valid_demo_count, episode_idx, task

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        # Define a config for an unused camera with all applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL3,  # note OPENGL3 for Peract2!
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        cameras = {}
        for name in camera_names:
            cameras[name] = used_cams

        obs_config = ObservationConfig(
            camera_configs=cameras,
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
            record_gripper_closing=False
        )

        return obs_config


class HandoverDiscrete(BimanualDiscrete):
    """
    A custom gripper action mode for the handover task.
    It forces one gripper to release so that the other grasps.
    """

    def action(self, scene, action):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        if 0.0 > action[1] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        right_open_condition = all(
            x > 0.9 for x in scene.robot.right_gripper.get_open_amount())

        left_open_condition = all(
            x > 0.9 for x in scene.robot.left_gripper.get_open_amount())

        right_current_ee = 1.0 if right_open_condition else 0.0
        left_current_ee = 1.0 if left_open_condition else 0.0

        right_action = float(action[0] > 0.5)
        left_action = float(action[1] > 0.5)

        if right_current_ee != right_action or left_current_ee != left_action:
            if not self._detach_before_open:
                self._actuate(scene, action)

        # Move objects between grippers
        if right_current_ee != right_action:
            if right_action == 0.0 and self._attach_grasped_objects:
                left_grasped_objects = scene.robot.left_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in left_grasped_objects:
                        scene.robot.left_gripper.release()
                        scene.robot.right_gripper.grasp(g_obj)
                    else:
                        scene.robot.right_gripper.grasp(g_obj)
            else:
                scene.robot.right_gripper.release()
        if left_current_ee != left_action:
            if left_action == 0.0 and self._attach_grasped_objects:
                right_grasped_objects = scene.robot.right_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in right_grasped_objects:
                        scene.robot.right_gripper.release()
                        scene.robot.left_gripper.grasp(g_obj)
                    else:
                        scene.robot.left_gripper.grasp(g_obj)
            else:
                scene.robot.left_gripper.release()

        if right_current_ee != right_action or left_current_ee != left_action:
            if self._detach_before_open:
                self._actuate(scene, action)
            if right_action == 1.0 or left_action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()
