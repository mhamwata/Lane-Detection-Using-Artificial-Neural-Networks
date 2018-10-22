import cv2
import os

def extractFrames(pathIn, pathOut):
    cap = cv2.VideoCapture(pathIn)
    count = 0
 
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
 
        if (ret == True):
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "frame{:d}.png".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break
 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
 
def main():
    extractFrames('Youtube Video Videocaps/tunnel_drive.MKV', 'Youtube Video Videocaps/tunnel/vid_caps')
 
if __name__=="__main__":
    main()

cv2.waitKey(0)