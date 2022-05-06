import time
from collections import namedtuple
from pynput.keyboard import Key
import numpy as np
from gym import spaces
from BotController import BotController
from ImageCapture import ImageCapture
import cv2 as cv
from PIL import Image, ImageEnhance, ImageOps
from Question_awserer import Question_anwserer


def get_human_action(key_char):
    map = {
        's':3,
        'w':0,
        'a':2,
        'd':1,
        'f':4,
        None : 5,
    }
    action = map[key_char]
    print("human key override: ", action,key_char)
    return action


class Enviornment:
    """
    Enviornment in which agent will be taking actions.
    Resposible for reward signal as change in pixels in score section of image
    """
    action_space=namedtuple('action_space',('n'))
    def __init__(self):
        self.screen_capture = ImageCapture()
        self.last_state = None
        self.question_mode = False
        self.Rewards={
            'episode_done_punish':-100, #punisment for ending/dying
            'Step_reward':0.1, #reward for new step
            'Score_reward':100,
            'Ammo_reward':0.2,
            'Health_reward':0.2,
            'Health_punish':-1
        }
        self.ready=False
        self.last_score_bar = None
        self.last_health = 870
        self.last_ammo = 1025
        self.pause = False
        self.BotController = BotController()
        self.Question_anwserer = Question_anwserer(self.BotController)
        self.action_dic = \
            {
                0: self.BotController.jump,
                1: self.BotController.down,
                2: self.BotController.go,
                3: self.BotController.go,
                4: self.BotController.shoot,
                5: None
            }
        self.dir_action = {2: 'left', 3: 'right'}
        self.action_space = 6
        self.last_action = None
        self.action_log = []
        self.done = False

    def pause_env(self,reason="no reason pause"):
        '''Pauses env'''
        self.pause=not self.pause
        if self.BotController.activate:
            self.BotController.pause()
    def wait_for_hp_frame(self):
        hp = 0
        print('Waiting for full hp frame')
        while (hp != self.last_health):
            new_frame, frame_states = self.get_frame_and_state()
            bars = self.get_bars(new_frame)
            hp = self.check_health_change(bars[1])
            self.handle_states(new_frame, frame_states)
        self.done=False
    def reset(self):
       '''Reset env states'''
       self.last_health = 870

       self.last_score_bar = None
       self.last_ammo = 1025
       self.last_action = None
       self.action_log = []
       self.question_mode=False
       self.pause=False
       if self.done:
           self.wait_for_hp_frame()




    def perform_action(self, action,hold):
        """performs action by using pyinput to press or hold keys on keyboard"""
        if action in [2,3]: #left right
            dir = self.dir_action[action]
            self.action_dic[action](hold_button=hold,direction=dir)
        else: #everything else
            action = self.action_dic[action]
            if action:
                action(hold_button=hold)
            else:
                pass #do nothing



    def wait_for_activation(self):
        while (self.pause):
            time.sleep(0.1)

    def get_frame_and_state(self):
        new_frame = self.screen_capture.get_new_frame()
        frame_states = self.screen_capture.get_state_img(new_frame)
        return new_frame, frame_states
    def get_hold(self,action):
        if len(self.action_log)>1:
            hold = self.action_log[-1] == action
        else:
            hold = False
        return hold
    def step(self, action):
        """
        Perform action inside Enviornment and get new states
        """
        while(self.pause):
            print('cant handle action, env is paused')
            self.wait_for_activation()
        #normal env actions
        else:
            hold = self.get_hold(action)
            self.perform_action(action,hold)
            reward, done = self.handle_step()

        return self.last_state, reward, done, None #none for gym api consistency

    def handle_step(self):
        new_frame, frame_states = self.get_frame_and_state()
        new_frame, frame_states = self.handle_states(new_frame, frame_states)
        self.last_state = new_frame
        bars = self.get_bars(new_frame)
        self.done = self.check_done(bars[1])
        reward = self.get_rewards(bars, self.done)
        self.update_states(bars)
        return reward, self.done

    def close(self):
        """end evn work"""

    def render(self,mode):
        """get new img from env"""
        frame = self.screen_capture.get_new_frame()
        screen_state = self.screen_capture.get_state_img(frame)
        if screen_state["running_game"][0]:
            return self.last_state
        else:
            self.last_state=frame
            return frame

    def get_state(self) -> np.array:
        """
        :return: current env state as np.array
        """
        return self.last_state

    def get_rewards(self, bars,done) -> int:
        """
        Sums all rewards signals
        :param img: np.array of actual img
        :return: sum of rewards
        """
        reward = 0
        if done:
            reward = self.Rewards["episode_done_punish"]
            return reward
        step_reward = self.Rewards["Step_reward"]
        score_reward = self._get_score_reward(bars[0])
        health_reward = self._get_health_reward(bars[1])
        ammo_reward = self._get_ammo_reward(bars[2])
        reward += score_reward + step_reward + ammo_reward + health_reward
        return reward

    def check_done(self, health_bar) -> bool:
        """
        checks if episode is done
        :param img: np.array of actual img
        :return: False if not done. True is episode is finished (no hp)
        """
        if self.check_health_change(health_bar) == 0:
            print("Done 0 health detected")
            return True
        return False

    def _get_score_reward(self, score_bar):
        """
        Score can only grow. If new score_bar is diffrent than last one, it means that agent got score.
        :param score_bar: np array of pixels of score_bar from ImageCapture
        :return: reward signal for score
        """

        if self.last_score_bar is None:
            self.last_score_bar = score_bar
            return 0
        compare = (self.last_score_bar == score_bar)
        count = np.size(compare) - np.count_nonzero(compare)
        thresh = 40
        # cv.imshow("scorebar", score_bar)
        if count >= thresh:
            print("score reward")
            return self.Rewards["Score_reward"]
        return 0

    def _get_health_reward(self, health_bar):
        actual_health = self.check_health_change(health_bar)
        if self.last_health < actual_health:
            print("health reward")
            return self.Rewards["Health_reward"]
        if self.last_health > actual_health:
            print("health punish reward")
            return self.Rewards["Health_punish"]
        return 0

    def check_health_change(self, health_bar):
        health = np.count_nonzero(health_bar == 255)
        if self.last_health is None:
            self.last_health = health
        return health

    def _get_ammo_reward(self, ammo_bar):
        actual_ammo = self.check_ammo_change(ammo_bar)
        if actual_ammo > self.last_ammo:
            print("Ammo reward")
            return self.Rewards["Ammo_reward"]
        return 0

    def check_ammo_change(self, ammo_bar):
        ammo = np.count_nonzero(ammo_bar == 255)  # 24 is constant light pixels of ammo
        if self.last_ammo is None:
            self.last_ammo = ammo
        return ammo

    def get_bars(self,img):
        bars = self.screen_capture.get_bars(img)
        return bars

    def update_states(self, bars):
        '''updates bar states'''

        self.last_score_bar = bars[0]
        self.last_health = self.check_health_change(bars[1])
        self.last_ammo = self.check_ammo_change(bars[2])

    def handle_states(self,img,screen_state):
        '''Handle special states like questions, menu, pause etc.'''
        #While game is running question mode can appear
        if screen_state["question_mode_detected"]:
             img, screen_state = self.handle_questions(img, img)
        #while game is not running menu,next_level screen etc can appear
        elif  not screen_state["running_game"][0]:
            print("!Special case!")
            while(not screen_state["running_game"][0]):
                if screen_state["menu"][0]:
                    self.handle_button(screen_state["menu"],"menu")
                    time.sleep(10)

                if screen_state["next_level"][0]:
                    self.handle_button(screen_state["next_level"],"next_level")

                elif screen_state["forward"][0]:
                    self.handle_button(screen_state["forward"],"forward")

                elif screen_state["endless_loading"][0]:
                    self.handle_endless_level_loading()

                elif screen_state["finish_menu"][0]:
                    #Special finish menu - can be skiped by pressing Pause
                    #self.handle_button(screen_state["finish_menu"],"finish_menu")
                    self.pause=True
                    time.sleep(1.5)
                    self.pause_env(reason="Pausing to skip finish menu")
                img, screen_state = self.get_frame_and_state()
        return img, screen_state

    def handle_questions(self, img,states):
        question_mode_detected = True
        while question_mode_detected:
            if self.BotController.activate: #if Bot can use keyboard
                id, lookup_image = self.screen_capture.get_question_id(img)
                self.BotController.realse_buttons()
                self.Question_anwserer.answer_question(id)
                img , states = self.get_frame_and_state()
                question_mode_detected = states["question_mode_detected"]
        #position agent i better spot automaticly
        time.sleep(1)
        self.BotController.go(direction="right",hold=1.3)
        return img, states

    def handle_button(self,state,key):
        waited = 0
        while(True):
            img, state=self.get_frame_and_state()
            state = state[key]
            if waited % 5==0:
                self.BotController.click_button(key)
            if not state[0]:
                break
            time.sleep(0.25)
            waited += 0.25

    def handle_endless_level_loading(self):
        self.BotController.click_button("endless_loading_next")


