"""
1. LEFT ENTER
2. LEFT LEFT ENTER (do sprawdzenia ikonka)
3. RIGHT RIGHT RIGHT DOWN RIGHT DOWN DOWN RIGHT ENTER
4. RIGHT DOWN RIGHT ENTER
5. DOWN RIGHT x13 ENTER
lvl 2
6. 72 -> 8x RIGHT DOWN RIGHT ENTER
7.  31x RIGHT DOWN 30x RIGHT DOWN  38x LEFT ENTER 12 19 31 50 81 131 212
8.  18x LEFT ENTER  2D = 45 73-45=28
9. RIGHT ENTER
10. RIGHT DOWN RIGHT ENTER
lvl3
11. 54 2x RIGHT ENTER
12. RIGHT ENTER
13. DOWN LEFT DOWN RIGHT ENTER
14. 31 RGIHT DOWN LEFT ENTER 3,12,24,33,66, 75?
15. 2x LEFT DOWN 2x RIGHT DOWN RIGHT ENTER - index 0 lenght 3
"""
import time

from BotController import BotController


class Question_anwserer:
    def __init__(self,bt:BotController):
        self.delay = 0.33
        self.answers = {
            0: self.question1,
            1: self.question2,
            2: self.question3,
            3: self.question4,
            4: self.question5,
            5: self.question6,
            6: self.question7,
            7: self.question8,
            8: self.question9,
            9: self.question10,
            10: self.question11,
            11: self.question12,
            12: self.question13,
            13: self.question14,
            14: self.question15,
        }
        self.controller=bt

    def choose_question(self, question_id):
        return self.answers[question_id]

    def answer_question(self, question_id: int):
        time.sleep(1)
        anwser_fun = self.choose_question(question_id)
        anwser_fun()
        self.controller.enter()

    def question1(self):
        self.controller.go(direction="right",delay=self.delay)
        time.sleep(0.5)
        pass

    def question2(self):
        '''fan question'''
        self.controller.go(direction="right",delay=self.delay)
        pass

    def question3(self):
        self.controller.go(direction="right",delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)

        pass

    def question4(self):
        self.controller.go(direction="right",delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)
        pass

    def question5(self):
        self.controller.down(delay=self.delay)
        for i in range(13):
            self.controller.go(direction="left",delay=self.delay)
        pass

    def question6(self):
        for i in range(8):
            self.controller.go(direction="right",delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)

    def question7(self):
        for i in range(31):
            self.controller.go(direction="right",delay=0.25)
        self.controller.down(delay=self.delay)
        for i in range(30):
            self.controller.go(direction="right",delay=0.25)
        self.controller.down(delay=self.delay)
        for i in range(38):
            self.controller.go(direction="left",delay=0.25)

    def question8(self):
        for i in range(18):
            self.controller.go(direction="left",delay=self.delay)

    def question9(self):
        self.controller.go(direction="right",delay=self.delay)
        pass

    def question10(self):
        self.controller.go(direction="right",delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)
        pass

    def question11(self):
        self.controller.go(direction="right",delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)

        pass

    def question12(self):
        self.controller.go(direction="right",delay=self.delay)
        pass

    def question13(self):
        self.controller.down(delay=self.delay)
        self.controller.go(direction="left",delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)
        pass

    def question14(self):
        for i in range(31):
            self.controller.go(direction="right",delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="left",delay=self.delay)
        pass

    def question15(self):
        self.controller.go(direction="left",delay=1)
        self.controller.go(direction="left",delay=1)
        self.controller.go(direction="left", delay=1)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)
        self.controller.go(direction="right", delay=self.delay)
        self.controller.down(delay=self.delay)
        self.controller.go(direction="right",delay=self.delay)
        pass
