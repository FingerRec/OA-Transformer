import cv2

video_path = "MSVD/YouTubeClips/fVWUaH2mCt4_1_7.avi"
cap = cv2.VideoCapture(video_path)
vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(vlen)