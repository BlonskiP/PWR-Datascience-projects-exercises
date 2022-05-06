from BotController import BotController

controller = BotController(DEBUG_MODE=True)

while not controller.end:
    if controller.activate:
        controller.jump()
        controller.shoot()
        controller.go(direction="left")
        controller.go(direction="right")
