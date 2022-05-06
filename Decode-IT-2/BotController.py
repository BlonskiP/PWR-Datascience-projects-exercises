from pynput import keyboard
from pynput.keyboard import Key, Controller, HotKey
from pynput.mouse import Button, Controller as mouseController
import time
from sys import exit
import cv2 as cv
import time

class BotController():

    def __init__(self, DEBUG_MODE=False):
        self.DEBUG_MODE = DEBUG_MODE
        self.Key_listen = False
        self.Controller = Controller()
        self.mouse = mouseController()
        self._TIME_DELAY = 0.12
        self.UP_KEY = Key.up
        self.DOWN_KEY = Key.down
        self.LEFT_KEY = Key.left
        self.RIGHT_KEY = Key.right
        self.SHOOT_KEY = Key.space
        self.ENTER_KEY = Key.enter
        self.PAUSE_KEY = 'p'
        self.activate = False
        self.Holded_button = []
        self.ghk = keyboard.GlobalHotKeys(
            {"q": self.turn_activation, "e": self.turn_down}
        )
        self.ghk.start()
        self.end = False
        self.buttons = {
            "next_level": (575, 535),
            "forward": (890, 639),
            "endless_loading_next":(510,555),
            "finish_menu":(592, 543),
            "menu":(511, 552)
        }
        self.last_key_pressed = None
        self.human_keyboard_listen = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.human_keyboard_listen.start()
        self.human_pressed_key=None

    def on_press(self,key):
        print('pressed: ',key,'last',self.human_pressed_key)
        if not self.activate: #in human mode
            if key.char in ['w','s','a','d','f']:
                self.human_pressed_key=key
                return key
            else:
                return False

    def on_release(self,key):
        print('human key realsed',self.human_pressed_key)
        if not self.activate: #in human mode
            if key.char in ['w','s','a','d','f']:
                self.human_pressed_key=None
                return key
            else:
                return False

    def turn_down(self)g:
        self.end = True

    def realse_buttons(self):
        if len(self.Holded_button)>=1:
            print('holded buttons',self.Holded_button)
        for button in self.Holded_button:
            self.Controller.release(button)
        self.Holded_button=[]

    def __del__(self):
        self.ghk.join()
        self.human_keyboard_listen.join()
        cv.destroyAllWindows()

    def turn_activation(self,delay=0.1):
        self.activate = not self.activate
        print(f"Bot activation: {self.activate}")
        time.sleep(delay)

    def enter(self,delay=0.1):
        self.Controller.press(self.ENTER_KEY)
        self.Controller.release(self.ENTER_KEY)
        time.sleep(delay)

    def jump(self,hold_button=False,delay=0.1):
        if self.DEBUG_MODE:
            print("Jump pressed")
        self.Controller.press(self.UP_KEY)
        self.Controller.release(self.UP_KEY)
        time.sleep(delay)

    def down(self,delay=0.1,hold_button=False):
        if self.DOWN_KEY not in self.Holded_button:
            self.Controller.press(self.DOWN_KEY)
            self.add_holded_button(self.DOWN_KEY)
        if not hold_button:
            self.realse_buttons()
        time.sleep(delay)

    def pause(self,delay=0.1):
        self.Controller.press(self.PAUSE_KEY)
        self.Controller.release(self.PAUSE_KEY)
        time.sleep(delay)

    def shoot(self,delay=0.1,hold_button=False):
        if self.DEBUG_MODE:
            print("Shoot (space) pressed")
        self.Controller.press(self.SHOOT_KEY)
        self.Controller.release(self.SHOOT_KEY)
        time.sleep(delay)

    def go(self, direction,delay=0.1,hold=0.0,hold_button=False):

        if direction == "left":
            if self.LEFT_KEY not in self.Holded_button:
                self.Controller.press(self.LEFT_KEY)
                self.add_holded_button(self.LEFT_KEY)
            time.sleep(hold)
            if not hold_button:
                self.realse_buttons()


        if direction == "right":
            if self.RIGHT_KEY not in self.Holded_button:
                self.Controller.press(self.RIGHT_KEY)
                self.add_holded_button(self.RIGHT_KEY)
            time.sleep(hold)
            if not hold_button:
                #self.Controller.release(self.RIGHT_KEY)
                self.realse_buttons()
        time.sleep(delay)

    def add_holded_button(self,key):
        #print('Holding',key)
        if key not in self.Holded_button:
            self.Holded_button.append(key)

    def click_position(self, position,buffor,delay=0.25,hold=0.0):
        new_x = position[0]+buffor
        new_y = position[1]+buffor
        old_x = self.mouse.position[0]
        old_y = self.mouse.position[1]
        print('mousepos',self.mouse.position)
        print('need to click',position)
        self.mouse.move(-old_x+new_x,-old_y+new_y)
        print('mousepos', self.mouse.position)
        self.mouse.press(Button.left)
        time.sleep(hold)
        self.mouse.release(Button.left)
        time.sleep(delay)
        pass

    def click_button(self, key):
        print('mousepos', self.mouse.position)
        self.click_position(self.buttons[key],buffor=0,hold=0.3,delay=0.25)
        pass
