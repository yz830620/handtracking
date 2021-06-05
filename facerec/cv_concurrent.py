import cv2
import numpy as np
import concurrent.futures

def display_frames(name):
    cap = cv2.VideoCapture(name)

    while True:
        ret, frames = cap.read()

        if ret is False:
            break

        # Show frames for testing:
        cv2.imshow(str(cap), frames)
        cv2.waitKey(100)

    cap.release()

    return name


def main():
    names = ['Videos/1copy.mp4', 'Videos/2copy.mp4']

    # Generate two synthetic video files to be used as input:
    ###############################################################################
    width, height, n_frames = 640, 480, 30  # 30 frames, resolution 640x480

    intput_filename1 = names[0]
    intput_filename2 = names[1]

    # Use MPEG4 codec (for testing)
    synthetic_out = cv2.VideoCapture(intput_filename1, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

    for i in range(n_frames):
        img = np.full((height, width, 3), 60, np.uint8)
        cv2.putText(img, str(i+1), (width//2-100*len(str(i+1)), height//2+100), cv2.FONT_HERSHEY_DUPLEX, 10, (30, 255, 30), 20)  # Green number
        synthetic_out.write(img)

    synthetic_out.release()

    width, height, n_frames = 320, 240, 20 # 20 frames, resolution 320x240
    synthetic_out = cv2.VideoCapture(intput_filename2, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

    for i in range(n_frames):
        img = np.full((height, width, 3), 60, np.uint8)
        cv2.putText(img, str(i+1), (width//2-50*len(str(i+1)), height//2+50), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 30, 30), 10)  # Blue number
        synthetic_out.write(img)

    synthetic_out.release()
    ###############################################################################


    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for name in executor.map(display_frames, names):
            print(name)

    cv2.destroyAllWindows() # For testing


# Using Python 3.6 there is an error: "TypeError: can't pickle cv2.VideoCapture objects"
if __name__ == '__main__':
    main()