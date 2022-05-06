from Enviornment import Enviornment
import cv2 as cv

env = Enviornment(DEBUG_MODE=True)

while not env.pause:
    result = env.step(0)
    if result:
        new_state, reward, done, img, lookup_image =result
    #cv.imshow("RF", lookup_image)
    if cv.waitKey(1) == ord("e"):
        cv.destroyAllWindows()
        break
