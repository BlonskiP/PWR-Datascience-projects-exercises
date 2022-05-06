import torch
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2 as cv

class EnvManager:

    def __init__(self, env, device="cpu",gym=False):
        self.device = device
        self.env = env
        self.current_screen = None
        self.done = False
        self.frame_memory = []
        self.frame_memory_max_size = 4
        self.gym=gym

    def add_frame_to_memory(self,frame):
        while(len(self.frame_memory)<self.frame_memory_max_size):
            zeros_frame = torch.zeros_like(frame)
            self.frame_memory.append(zeros_frame),
        if len(self.frame_memory)>= self.frame_memory_max_size:
            self.frame_memory.pop(0)
            self.frame_memory.append(frame)

    def reset(self):
        self.env.reset()
        self.frame_memory = []
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_avaiable(self):
        if self.gym:
            return self.env.action_space.n
        else:
            return self.env.action_space

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            self.current_screen = black_screen
        else:
            self.current_screen = self.get_processed_screen()
        self.add_frame_to_memory(self.current_screen)
        state = self.stack_memories()
        return state

    def stack_memories(self):
        frame = torch.cat(self.frame_memory, 1)
        return frame

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render("rgb_array")
        screen=screen.transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def human_input(self):
        key_char = self.env.BotController.human_pressed_key.char
        action = self.env.get_human_action(key_char)
        return torch.tensor([action])

    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height * 0)
        bottom = int(screen_height * 1)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        resize = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor()
        ])
        return resize(screen).unsqueeze(0).to(self.device)

    def plot_processed_screen(self, screen):
        plt.figure()
        plt.imshow(screen.squeeze(0).permute(1, 2, 0), interpolation='none')
        plt.title('Screen example')
        plt.show()
