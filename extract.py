import cv2
import os


def videoToFrames(cam, startFrame, endFrame, folderName):
    try:
        # creating a folder named data
        if not os.path.exists(f'frames/{folderName}'):
            os.makedirs(f'frames/{folderName}')
        # if not created then raise error
    except OSError:
        print(f'Error: Creating directory of {folderName}')

    while (True):
        cam.set(cv2.CAP_PROP_POS_MSEC, (startFrame * 1000))
        # reading from frame
        hasFrame, frame = cam.read()

        if hasFrame and startFrame <= endFrame:
            # if video is still left continue creating images
            name = f'./frames/{folderName}/frame' + str(startFrame) + '.jpg'
            print('Creating...' + name)
            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            startFrame += 1
        else:
            break
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
