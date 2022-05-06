from time import sleep

import numpy as np
import win32gui, win32ui, win32con
import cv2 as cv
from PIL import Image

question_template_path = f"question_template_low.PNG"
questions_template_path = f"quests_templates/quest_"
menu_template_path = f"templates/start.bmp"
forward_template_path =  f"templates/forward.bmp"
play_next_path = f"templates/contiune_game.png"
endless_loading = f"templates/endless_loading.bmp"
endless_loading_next = f"templates/endless_loading_next.bmp"
finish_menu = f"templates/finish_menu.bmp"
running_game = f"templates/running_game.bmp"
class ImageCapture:
    def __init__(self, DEBUG_MODE=False, IMG_DEBUG_SAVE=False):


        self.width = 1024
        self.height = 768

        self.hwnd = win32gui.GetDesktopWindow()
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        self.cropped_x = 0
        self.cropped_y = 0
        self.offset_x = 0
        self.offset_y = 0
        self.IMG_DEBUG_SAVE = IMG_DEBUG_SAVE
        self.DEBUG_MODE = DEBUG_MODE
        self.forward_template = cv.imread(forward_template_path,cv.IMREAD_GRAYSCALE)
        self.question_template = cv.imread(question_template_path, cv.IMREAD_GRAYSCALE)
        self.menu_template = cv.imread(menu_template_path, cv.IMREAD_GRAYSCALE)
        self.play_next_template = cv.imread(play_next_path,cv.IMREAD_GRAYSCALE)
        self.endless_loading_template = cv.imread(endless_loading,cv.IMREAD_GRAYSCALE)
        self.endless_loading_next_template = cv.imread(endless_loading_next,cv.IMREAD_GRAYSCALE)
        self.finish_menu_template = cv.imread(finish_menu,cv.IMREAD_GRAYSCALE)
        self.running_game_template = cv.imread(running_game,cv.IMREAD_GRAYSCALE)

        self.img_states = {
            "menu":False,
            "question_mode_detected":False,
            "loading":False,
            "next_level":False,
            "forward":False,
            "endless_loading":False,
            "finish_menu":False,
            "running_game":False,
        }

        self.score_bar = {
            "left_top": (350, 38),
            "width": 100,
            "height": 18,
            "color": (122, 122, 0),
        }
        self.health_bar = {
            "left_top": (25, 30),
            "width": 100,
            "height": 5,
            "color": (255, 0, 0),
        }

        self.ammo_bar = {
            "left_top": (675, 30),
            "width": 100,
            "height": 5,
            "color": (0, 122, 122),
        }

        self.next_level_area = {
            "x": 400,
            "y": 435,
            "width": 125,
            "height":45,
            "color": (255, 255, 0)
        }

        self.menu_area = {
            "x": 325,
            "y": 440,
            "width": 150,
            "height": 50,
            "color": (255, 255, 0)
        }
        self.running_game_area={
            "x": 90,
            "y": 45,
            "width": 40,
            "height": 20,
            "color": (255, 255, 0)
        }

        self.forward_click = {
            "x": 645,
            "y": 520,
            "width": 105,
            "height":30,
            "color": (255, 255, 0)
        }
        self.quest_template_area = {
            "x":90,
            "y":110,
            "width": 70,
            "height":60,
            "color":(255,255,0)
        }
        self.endless_loading_area = {
            "x": 50,
            "y": 80,
            "width": 200,
            "height": 200,
            "color": (255, 255, 0)
        }

        self.endless_loading_area_next = {
            "x": 350,
            "y": 450,
            "width": 100,
            "height": 30,
            "color": (255, 255, 0)
        }
        self.finish_menu_area = {
            "x": 400,
            "y": 440,
            "width": 120,
            "height": 30,
            "color": (255, 255, 0)
        }
        self.quest_text_area = {
            "x":190,
            "y":50,
            "width":405,
            "height":340,
            "color":(125,125,125)
        }
        self.question_tempates =[
            cv.imread(questions_template_path+f"{x}"+"_title.PNG",cv.IMREAD_GRAYSCALE)
            for x in range(1,16)]

    def reset_states(self):
        self.img_states = {
            "menu": (False,0),
            "question_mode_detected": (False,0),
            "loading": (False,0),
            "next_level": (False,0),
            "forward": (False,0),
            "endless_loading": (False,0),
            "endless_loading_next":(False,0),
            "finish_menu":(False,0),
            "running_game": (False,0)
        }

    def get_screen(self):
        '''
        Using windows gui makes screenshot in high speed
        returns img
        '''
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt(
            (0, 0),
            (self.w, self.h),
            dcObj,
            (self.cropped_x, self.cropped_y),
            win32con.SRCCOPY,
        )
        #dataBitMap.SaveBitmapFile(cDC, 'debug_win.bmp')
        signed_ints_array = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signed_ints_array, dtype="uint8")
        img.shape = (self.h, self.w, 4)
        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        #
        return img

    def update_states(self,img):
        '''
        gets states from processed image.
        Img MUST be exacly 800x640
        '''
        self.img_states['menu'] = self.detect_icon(template=self.menu_template, img=img, area=self.menu_area)
        self.img_states["forward"] = self.detect_icon(template=self.forward_template, img=img,
                                                      area=self.forward_click)
        # self.img_states["loading"] = self.detect_icon(template=loading_template,img=lookup_image,area=self.loading_area)
        self.img_states["next_level"] = self.detect_icon(template=self.play_next_template, img=img,
                                                         area=self.next_level_area)
        self.img_states['question_mode_detected'], lookup_image = self.detect_question_template(img)
        self.img_states["endless_loading"] = self.detect_icon(template=self.endless_loading_template, img=lookup_image,  area=self.endless_loading_area)
        if self.img_states["endless_loading"][0]:
            self.img_states["endless_loading_next"] = self.detect_icon(template=self.endless_loading_next_template,
                                                                       img=lookup_image,
                                                                       area=self.endless_loading_area_next)
        self.img_states["finish_menu"] = self.detect_icon(template=self.finish_menu_template,
                                                          area=self.finish_menu_area, img=lookup_image)
        self.img_states["running_game"] = self.detect_icon(template=self.running_game_template,
                                                           area=self.running_game_area, img=lookup_image)

    def get_state_img(self,img):
        '''
        generates new state for img
        '''
        self.update_states(img)
        return self.img_states

    def get_bars(self,img):
        '''
        Crops health/ammo/score bars from img 800x640
        '''
        score_bar = self.get_bar(img, self.score_bar)
        health_bar = self.get_bar(img, self.health_bar)
        ammo_bar = self.get_bar(img, self.ammo_bar)

        return score_bar,health_bar,ammo_bar
    def get_new_frame(self):
        frame = self.get_screen()
        return self.process_img(frame)

    def process_img(self, img):
        '''process_img'''
        img = img[..., :3]
        img = np.ascontiguousarray(img)
        resized = cv.resize(img, (800, 640), interpolation=cv.INTER_AREA)
        return resized

    def draw_rectangles(self, img, bar):
        left_top = bar["left_top"]
        right_bottom = (
            bar["left_top"][0] + bar["width"],
            bar["left_top"][1] + bar["height"],
        )
        return cv.rectangle(img, left_top, right_bottom, bar["color"], 2)

    def get_bar(self, resize_img, bar_type: dict):
        width = bar_type["left_top"][0] + bar_type["width"]
        height = bar_type["left_top"][1] + bar_type["height"]
        bar = resize_img[
            bar_type["left_top"][1] : height, bar_type["left_top"][0] : width
        ]
        if bar_type == self.score_bar:
            binary_color = self.preprocess_bar(bar,220)
        else:
            binary_color = self.preprocess_bar(bar)
        return binary_color

    def preprocess_bar(self, bar,thresh=128):
        return ((bar > thresh) * 255).astype(np.uint8)

    def detect_question_template(self, lookup_image):
        start_x = self.quest_template_area["x"]
        end_x = self.quest_template_area["x"]+ self.quest_template_area["width"]
        start_y = self.quest_template_area["y"]
        end_y = self.quest_template_area["y"] + self.quest_template_area["height"]
        new_img = lookup_image[start_y:end_y,start_x:end_x]
        #cv.imwrite("test.png", new_img)
        grey = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(grey, self.question_template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        threshold = 0.9
        detected = False
        if max_val > threshold:
            detected = True
        if self.DEBUG_MODE and detected:
            top_left = (max_loc[0]+start_x,max_loc[1]+start_y)
            w, h = self.question_template.shape[::-1]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(lookup_image, top_left, bottom_right, self.quest_template_area['color'], 2)

        return detected, lookup_image

    def detect_quest(self,template,img):
        start_x = self.quest_text_area["x"]
        end_x = self.quest_text_area["x"] + self.quest_text_area["width"]
        start_y = self.quest_text_area["y"]
        end_y = self.quest_text_area["y"] + self.quest_text_area["height"]
        new_img = img[start_y:end_y, start_x:end_x]
       # cv.imwrite("new.png", new_img)
        grey = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(grey, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        threshold = 0.9
        detected = False
        if max_val > threshold:
            detected = True
        return max_val

    def get_question_id(self,img):
        detection = [self.detect_quest(temp,img) for temp in self.question_tempates]
        id = np.argmax(detection)
        print("Choosen question:",id)
        return id, img
        pass

    def detect_icon(self, template,img,area):

        start_x = area["x"]
        end_x = area["x"] + area["width"]
        start_y = area["y"]
        end_y = area["y"] + area["height"]

        new_img = img[start_y:end_y,start_x:end_x]
       # cv.imwrite("newa.png", img)
       # cv.imwrite("new.png", new_img)
        grey = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(grey, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        threshold = 0.9
        detected = False

        if max_val > threshold:
            detected = True
        if self.DEBUG_MODE and detected:
            top_left = max_loc
            w, h = template.shape[::-1]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img, top_left, bottom_right, 255, 2)
        max_loc = (max_loc[0]+start_x,max_loc[1]+start_y)
        return detected , max_loc
